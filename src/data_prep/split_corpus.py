"""
Splits the cleaned corpus into train, validation and test sets.

    - test (5% of the corpus): held out, used only for the final evaluation
    - validation (10% of the remainder): early stopping, model selection and threshold tuning
    - train: everything else

Validation and test lines are further split into single sentences.

Usage:
    python -m src.data_prep.split_corpus
"""

import re

from datasets import load_dataset

from src.config import CLEANED_DATASET, SEED, TEST_CORPUS, TRAIN_CORPUS, VAL_CORPUS

TEST_SIZE = 0.05
VAL_SIZE = 0.10


def split_into_sentences(lines):
    """ Splits each line after ., ! or ? and drops fragments of 10 characters or fewer. """
    sentences = []
    for text in lines:
        sentences.extend(s.strip() for s in re.split(r'(?<=[.!?])', text) if len(s.strip()) > 10)
    return sentences


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    dataset = load_dataset("text", data_files=str(CLEANED_DATASET), split="train")

    # Fixed seeds so the three sets are always the same
    held_out = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    train_val = held_out["train"].train_test_split(test_size=VAL_SIZE, seed=SEED)

    train_lines = train_val["train"]["text"]
    val_sentences = split_into_sentences(train_val["test"]["text"])
    test_sentences = split_into_sentences(held_out["test"]["text"])

    write_lines(TRAIN_CORPUS, train_lines)
    write_lines(VAL_CORPUS, val_sentences)
    write_lines(TEST_CORPUS, test_sentences)

    print(f"Training lines:       {len(train_lines)}")
    print(f"Validation sentences: {len(val_sentences)}")
    print(f"Test sentences:       {len(test_sentences)}")


if __name__ == "__main__":
    main()
