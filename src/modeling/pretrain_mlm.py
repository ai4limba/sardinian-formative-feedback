"""
Step 1 - Continued pre-training of a BERT masked language model on the Sardinian corpus.

The adapted model is used at inference time to rank correction candidates in context,
and as the starting checkpoint for the error detector.

Usage:
    python -m src.modeling.pretrain_mlm
"""

import argparse
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

from src.config import MLM_DEFAULTS, MLM_DIR, PLOTS_DIR, RUNS_DIR, SEED, TRAIN_CORPUS, VAL_CORPUS
from src.utils import append_record, relative_to_root, set_seed, timestamp, use_bf16


def parse_args():
    d = MLM_DEFAULTS
    parser = argparse.ArgumentParser(description="Continued MLM pre-training on the Sardinian corpus.")
    parser.add_argument("--base-model", default=d["base_model"])
    parser.add_argument("--output-dir", default=str(MLM_DIR))
    parser.add_argument("--epochs", type=int, default=d["epochs"])
    parser.add_argument("--batch-size", type=int, default=d["batch_size"])
    parser.add_argument("--grad-accum", type=int, default=d["grad_accum"])
    parser.add_argument("--learning-rate", type=float, default=d["learning_rate"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a handful of steps to check that everything works; nothing is recorded.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    max_seq_length = MLM_DEFAULTS["max_seq_length"]
    checkpoints_dir = f"{args.output_dir}-checkpoints"

    # ==========================================
    # DATA
    # ==========================================
    # Uncased BERT tokenizers strip accents by default, which would make accented and
    # unaccented Sardinian words identical. The setting is saved with the model, so the
    # detector and the inference pipeline inherit it.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, strip_accents=False)
    raw_datasets = load_dataset("text", data_files={"train": str(TRAIN_CORPUS), "validation": str(VAL_CORPUS)})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_seq_length)

    def group_texts(examples):
        # Join all texts and cut them into blocks of max_seq_length tokens
        concatenated = {key: sum(examples[key], []) for key in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])

        # Drop the tail so there is no short last block
        # (e.g. 1060 tokens -> 1024, two blocks of 512, the last 36 tokens are discarded)
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length

        return {
            key: [tokens[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for key, tokens in concatenated.items()
        }

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    print(f"Training blocks:   {len(lm_datasets['train'])}")
    print(f"Validation blocks: {len(lm_datasets['validation'])}")

    # ==========================================
    # TRAINING
    # ==========================================
    model = AutoModelForMaskedLM.from_pretrained(args.base_model)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_DEFAULTS["mlm_probability"]
    )

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        learning_rate=args.learning_rate,
        weight_decay=MLM_DEFAULTS["weight_decay"],
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=5 if args.smoke_test else -1,
        warmup_ratio=MLM_DEFAULTS["warmup_ratio"],

        # Evaluation
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,

        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,

        # Efficiency
        bf16=use_bf16(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=MLM_DEFAULTS["patience"])]
    )

    print(f"Starting pre-training from '{args.base_model}':")
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    shutil.rmtree(checkpoints_dir, ignore_errors=True)
    print(f"Model and tokenizer saved to: {args.output_dir}")

    if args.smoke_test:
        print(f"Smoke test completed. Validation loss: {trainer.evaluate()['eval_loss']:.4f}")
        return

    # ==========================================
    # RUN RECORD
    # ==========================================
    history = trainer.state.log_history

    train_loss = [x['loss'] for x in history if 'loss' in x]
    train_epochs = [x['epoch'] for x in history if 'loss' in x]
    eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
    eval_epochs = [x['epoch'] for x in history if 'eval_loss' in x]

    # Actual figures, taking early stopping into account
    best_val_loss = min(eval_loss)
    best_epoch = eval_loss.index(best_val_loss) + 1

    run_id = timestamp()
    model_name = args.base_model.split("/")[-1]
    plot_path = PLOTS_DIR / "pretrain" / f"pretrain_loss_{run_id}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_epochs, train_loss, label="Training Loss")
    plt.plot(eval_epochs, eval_loss, label="Validation Loss", marker="o")
    plt.title(f"{model_name} - {run_id}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)

    append_record(RUNS_DIR / "pretraining_records.csv", {
        "Timestamp": run_id,
        "Base Model Name": model_name,
        "Seed": args.seed,
        "Max Epochs": args.epochs,
        "Stopped Epoch": len(eval_loss),
        "Best Epoch": best_epoch,
        "Batch Size": args.batch_size,
        "Grad Accumulation": args.grad_accum,
        "Learning Rate": args.learning_rate,
        "MLM Probability": MLM_DEFAULTS["mlm_probability"],
        "Final Train Loss": train_loss[-1],
        "Best Val Loss": best_val_loss,
        "Plot Path": relative_to_root(plot_path)
    })
    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch} of {len(eval_loss)})")
    print(f"Run recorded in {RUNS_DIR / 'pretraining_records.csv'}")


if __name__ == "__main__":
    main()
