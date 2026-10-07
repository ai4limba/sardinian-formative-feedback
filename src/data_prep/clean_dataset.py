"""
Cleans the raw Campidanese Sardinian corpus.

Reads the parallel Sardinian/Italian TSV, keeps the Sardinian side, normalises it and
writes one sentence per line.

Usage:
    python -m src.data_prep.clean_dataset
"""

import re

import pandas as pd

from src.config import CLEANED_DATASET, RAW_DATASET


def clean_text(text):
    """ Normalises one corpus line. """
    # Lowercase: the models are uncased, which also helps in a low-resource setting
    text = str(text).lower()

    # Normalise single and double quotes
    text = text.replace('’', "'").replace('‘', "'").replace('`', "'")
    text = text.replace('“', '"').replace('”', '"')

    # Anomalies found in the corpus: en dash and ellipsis character
    text = text.replace('–', '-')
    text = text.replace('…', '...')

    # Drop any other "non standard" character
    text = re.sub(r"[^a-z0-9àèìòùáéíóú\'\-\.,!?;:()\" ]", " ", text)

    # Collapse the multiple spaces this may have created
    text = re.sub(r'\s+', ' ', text).strip()

    # Drop quotes or parentheses that are opened without being closed (and vice versa)
    if text.count('"') % 2 != 0:
        text = text.replace('"', '')
    if text.count('(') != text.count(')'):
        text = text.replace('(', '').replace(')', '')

    # Acronyms (such as "a.n.c" and "a.n.m.i.g."): remove the dots so they do not break tokenisation
    text = re.sub(r'\b(?:[a-z]\.){2,}', lambda m: m.group(0).replace('.', ''), text)

    # Tidy spacing and punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([!?])\.', r'\1', text)
    text = re.sub(r'([!?])"\.', r'\1"', text)
    text = re.sub(r'\.{2,}', '...', text)
    text = re.sub(r'\,+', ',', text)
    text = re.sub(r'^([\'"\(]*\s*)[\.,!?;:\-]+\s*', r'\1', text)

    return text


def main():
    df = pd.read_csv(RAW_DATASET, sep='\t')

    # Keep only the Sardinian side of the parallel corpus
    sentences = df['Sardo'].drop_duplicates().apply(clean_text)

    # Keep sentences with 3 or more words (\w+ so that elisions count as separate words)
    word_counts = sentences.apply(lambda sentence: len(re.findall(r'\w+', str(sentence))))
    sentences = sentences[word_counts >= 3]

    with open(CLEANED_DATASET, 'w', encoding='utf-8', newline='\n') as f:
        for text in sentences:
            f.write(f"{text}\n")

    print(f"Raw rows: {len(df)} | Cleaned sentences: {len(sentences)}")
    print(f"Saved to: {CLEANED_DATASET}")


if __name__ == "__main__":
    main()
