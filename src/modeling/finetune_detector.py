"""
Step 3 - Fine-tuning of the error detector (Token Classification: CORRECT / ERROR).

Starts from the Sardinian MLM and trains on synthetic errors. Training sentences are
corrupted on the fly, so the model sees different errors at every epoch; validation
sentences are corrupted once, so that runs can be compared.

Usage:
    python -m src.modeling.finetune_detector
"""

import argparse
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import concatenate_datasets, load_dataset
from sklearn.metrics import accuracy_score, fbeta_score, precision_recall_fscore_support
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

from src.config import (DETECTOR_DEFAULTS, DETECTOR_DIR, ERROR_RATE, MLM_DIR, PLOTS_DIR, RUNS_DIR, SEED,
                        TRAIN_CORPUS, VAL_CORPUS)
from src.data_prep.synth_errors import SyntheticErrorGenerator
from src.utils import append_record, relative_to_root, set_seed, timestamp, use_bf16


def parse_args():
    d = DETECTOR_DEFAULTS
    parser = argparse.ArgumentParser(description="Fine-tune the Sardinian spelling error detector.")
    parser.add_argument("--base-model", default=str(MLM_DIR))
    parser.add_argument("--output-dir", default=str(DETECTOR_DIR))
    parser.add_argument("--epochs", type=int, default=d["epochs"])
    parser.add_argument("--batch-size", type=int, default=d["batch_size"])
    parser.add_argument("--grad-accum", type=int, default=d["grad_accum"])
    parser.add_argument("--learning-rate", type=float, default=d["learning_rate"])
    parser.add_argument("--error-rate", type=float, default=ERROR_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a handful of steps to check that everything works; nothing is recorded.")
    return parser.parse_args()


def make_corrupt_and_tokenize(tokenizer, corruptor, max_seq_length):
    """ Builds the function that corrupts a batch of sentences and tokenizes the result. """

    def corrupt_and_tokenize(batch):
        corrupted_words = []
        error_labels = []

        for text in batch["text"]:
            res = corruptor.corrupt_sentence(text)
            corrupted_words.append(res["corrupted_words_list"])
            error_labels.append(res["error_labels"])

        tokenized_inputs = tokenizer(
            corrupted_words,
            truncation=True,
            is_split_into_words=True,
            max_length=max_seq_length
        )

        # Align the word labels with the sub-tokens produced by the tokenizer
        labels = []
        for i, label_list in enumerate(error_labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            # -100 for special tokens, which are ignored by the loss
            labels.append([-100 if word_idx is None else label_list[word_idx] for word_idx in word_ids])

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return corrupt_and_tokenize


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=2)

    # Leave out the -100 tokens (padding and special tokens)
    flat_true = [l for label in labels for l in label if l != -100]
    flat_preds = [p for prediction, label in zip(predictions, labels) for p, l in zip(prediction, label) if l != -100]

    acc = accuracy_score(flat_true, flat_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(flat_true, flat_preds, average='binary', zero_division=0)
    # F0.5 weights precision above recall
    f0_5 = fbeta_score(flat_true, flat_preds, beta=0.5, average='binary', zero_division=0)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "f0.5": f0_5}


def main():
    args = parse_args()
    set_seed(args.seed)
    max_seq_length = DETECTOR_DEFAULTS["max_seq_length"]
    checkpoints_dir = f"{args.output_dir}-checkpoints"

    # ==========================================
    # DATA
    # ==========================================
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    raw_datasets = load_dataset("text", data_files={"train": str(TRAIN_CORPUS), "validation": str(VAL_CORPUS)})

    print(f"Training lines:       {len(raw_datasets['train'])}")
    print(f"Validation sentences: {len(raw_datasets['validation'])}")

    # Two independent generators: the validation errors do not depend on how long training lasts
    train_corruptor = SyntheticErrorGenerator(error_rate=args.error_rate, seed=args.seed)
    val_corruptor = SyntheticErrorGenerator(error_rate=args.error_rate, seed=args.seed + 1)

    # DYNAMIC corruption of the training set: new errors every time a sentence is read
    raw_datasets["train"].set_transform(make_corrupt_and_tokenize(tokenizer, train_corruptor, max_seq_length))

    # STATIC corruption of the validation set (repeated, with different errors in each copy)
    expanded_val = concatenate_datasets([raw_datasets["validation"]] * DETECTOR_DEFAULTS["val_copies"])
    static_val_dataset = expanded_val.map(
        make_corrupt_and_tokenize(tokenizer, val_corruptor, max_seq_length),
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=False
    )

    # ==========================================
    # TRAINING
    # ==========================================
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "CORRECT", 1: "ERROR"},
        label2id={"CORRECT": 0, "ERROR": 1}
    )

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        learning_rate=args.learning_rate,
        weight_decay=DETECTOR_DEFAULTS["weight_decay"],
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=5 if args.smoke_test else -1,
        remove_unused_columns=False,

        # Evaluation
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f0.5",  # The best checkpoint is chosen on F0.5

        # Reproducibility: data is loaded in the main process so that the
        # corruption sequence is fully determined by the seed
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,

        # Efficiency
        bf16=use_bf16(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=static_val_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=DETECTOR_DEFAULTS["patience"])]
    )

    print(f"Starting fine-tuning from '{args.base_model}':")
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    shutil.rmtree(checkpoints_dir, ignore_errors=True)
    print(f"Model (best checkpoint) and tokenizer saved to: {args.output_dir}")

    if args.smoke_test:
        metrics = trainer.evaluate()
        print(f"Smoke test completed. Validation F0.5: {metrics['eval_f0.5']:.4f}")
        return

    # ==========================================
    # RUN RECORD
    # ==========================================
    history = trainer.state.log_history

    train_loss = [x['loss'] for x in history if 'loss' in x]
    train_epochs = [x['epoch'] for x in history if 'loss' in x]
    eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
    eval_epochs = [x['epoch'] for x in history if 'eval_loss' in x]
    eval_f05 = [x['eval_f0.5'] for x in history if 'eval_f0.5' in x]
    eval_precision = [x['eval_precision'] for x in history if 'eval_precision' in x]
    eval_recall = [x['eval_recall'] for x in history if 'eval_recall' in x]
    eval_f1 = [x['eval_f1'] for x in history if 'eval_f1' in x]

    # Actual figures, taking early stopping into account
    best_f05 = max(eval_f05)
    best_idx = eval_f05.index(best_f05)

    run_id = timestamp()
    plot_path = PLOTS_DIR / "finetune" / f"finetune_{run_id}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_epochs, train_loss, label="Training Loss")
    ax1.plot(eval_epochs, eval_loss, label="Validation Loss", marker="o")
    ax1.set_title("Fine-tune Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2.plot(eval_epochs, eval_f05, label="F0.5 Score", marker="o", color="green")
    ax2.plot(eval_epochs, eval_precision, label="Precision", marker="x", color="blue")
    ax2.plot(eval_epochs, eval_recall, label="Recall", marker="s", color="orange")
    ax2.set_title("Performance Metrics")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)

    append_record(RUNS_DIR / "finetune_records.csv", {
        "Timestamp": run_id,
        "Base Model Name": args.base_model.replace("\\", "/").split("/")[-1],
        "Seed": args.seed,
        "Error Rate": args.error_rate,
        "Max Epochs": args.epochs,
        "Stopped Epoch": len(eval_loss),
        "Best Epoch": best_idx + 1,
        "Batch Size": args.batch_size,
        "Grad Accumulation": args.grad_accum,
        "Learning Rate": args.learning_rate,
        "Final Train Loss": train_loss[-1] if train_loss else None,
        "Best F0.5 Score": best_f05,
        "Associated Precision": eval_precision[best_idx],
        "Associated Recall": eval_recall[best_idx],
        "Associated F1": eval_f1[best_idx],
        "Associated Val Loss": eval_loss[best_idx],
        "Plot Path": relative_to_root(plot_path)
    })
    print(f"Best validation F0.5: {best_f05:.4f} (epoch {best_idx + 1} of {len(eval_loss)})")
    print(f"Run recorded in {RUNS_DIR / 'finetune_records.csv'}")


if __name__ == "__main__":
    main()
