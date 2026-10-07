"""
Builds the Sardinian word lists stored in data/dictionaries.

    corpus       words of the training corpus                       -> corpus_dict.txt
    sardu-wiki   Campidanese lemmas scraped from sardu.wiki         -> campidanese_standard.txt,
                                                                       campidanese_non_standard.txt
    ditzionariu  entries scraped from ditzionariu.nor-web.eu        -> ditzionariu_in_linia.txt
    wikipedia    words of the Sardinian Wikipedia (1 Nov 2023 dump) -> wikipedia_dict.txt
    stats        size and overlap of the dictionaries

Only "corpus" has to be re-run when the corpus split changes. The scraped lists are already
included in the repository: re-running those sources takes a long time and depends on the
current content of the websites.

Usage:
    python -m src.data_prep.build_dictionaries corpus
"""

import argparse
import re
import string
import time
import unicodedata

from src.config import (CORPUS_DICT, DITZIONARIU_DICT, NON_STANDARD_DICT, STANDARD_DICT,
                        TRAIN_CORPUS, WIKIPEDIA_DICT, WORD_PATTERN)

HEADERS = {'User-Agent': 'Mozilla/5.0'}
REQUEST_DELAY = 0.5  # Seconds between requests, to be polite with the websites


def save_dictionary(path, words):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for word in sorted(words):
            f.write(word + "\n")
    print(f"Saved '{path.name}' with {len(words)} words.")


def load_dictionary(path):
    with open(path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())


# ==========================================
# TRAINING CORPUS
# ==========================================

def build_corpus_dictionary():
    """ Vocabulary of the training corpus, tokenised exactly as the spellchecker does. """
    # Words only: the shared pattern without its number and punctuation alternatives
    word_pattern = re.compile(WORD_PATTERN.split("|")[0])
    vocab = set()

    with open(TRAIN_CORPUS, "r", encoding="utf-8") as f:
        for line in f:
            words = word_pattern.findall(line.lower())
            # Keep purely alphabetic tokens
            vocab.update(w for w in words if w.replace("'", "").replace("-", "").isalpha())

    save_dictionary(CORPUS_DICT, vocab)


# ==========================================
# SARDINIAN WIKIPEDIA
# ==========================================

def build_wikipedia_dictionary():
    """ Vocabulary of the Sardinian Wikipedia (Hugging Face dump of 1 November 2023, ~7,600 pages). """
    from datasets import load_dataset

    dataset = load_dataset("wikimedia/wikipedia", "20231101.sc", split="train")

    # Letters only, no apostrophes or hyphens
    entry_pattern = re.compile(r"\b[a-zàèìòùáéíóú]+\b")
    vocab = set()
    for text in dataset["text"]:
        # Keep words longer than one character
        vocab.update(word for word in entry_pattern.findall(text.lower()) if len(word) > 1)

    save_dictionary(WIKIPEDIA_DICT, vocab)


# ==========================================
# DITZIONARIU IN LINIA (ditzionariu.nor-web.eu)
# ==========================================

def scrape_ditzionariu_page(letter, page):
    import requests
    from bs4 import BeautifulSoup

    url = f"https://ditzionariu.nor-web.eu/it/leghe/{letter}/{page}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return set()

    soup = BeautifulSoup(response.text, 'html.parser')

    entries = set()
    for entry in soup.find_all('div', class_='shadow-z-1'):
        original_text = entry.find('strong').text.strip()

        # An entry may list several comma-separated forms
        for word in original_text.split(','):
            # Lowercase, drop digits and punctuation
            clean_word = re.sub(r"[^a-zàèìòùáéíóú']+", '', word.lower()).strip()
            if len(clean_word) > 1:
                entries.add(clean_word)

    return entries


def build_ditzionariu_dictionary():
    # Whole alphabet, plus TZ which has its own section
    letters = list(string.ascii_uppercase) + ["TZ"]
    dictionary = set()

    for letter in letters:
        page = 1
        while True:
            print(f"Scraping {letter} - page {page}...", end="\r")
            words_in_page = scrape_ditzionariu_page(letter, page)

            # An empty page means the letter is finished
            if not words_in_page:
                print(f"\nLetter {letter} done: {page - 1} pages.")
                break

            dictionary.update(words_in_page)
            page += 1
            time.sleep(REQUEST_DELAY)

    save_dictionary(DITZIONARIU_DICT, dictionary)


# ==========================================
# SARDU.WIKI (Campidanese lemmas)
# ==========================================

def extract_lemmas(soup, css_class):
    lemmas = []
    for element in soup.find_all(class_=css_class):
        # Split on apostrophes and spaces
        for part in re.split(r"[\s']+", element.text.lower()):
            clean_word = re.sub(r"[^a-zàèìòùáéíóú\-]+", '', part).strip(' -')
            if len(clean_word) > 1:
                lemmas.append(clean_word)
    return lemmas


def scrape_sardu_wiki_lemma(url, session):
    from bs4 import BeautifulSoup

    response = session.get(url, headers=HEADERS)
    if response.status_code != 200:
        return [], []

    soup = BeautifulSoup(response.text, 'html.parser')
    return (extract_lemmas(soup, "standard-lemma campidanese"),
            extract_lemmas(soup, "nonstandard-lemma campidanese"))


def build_sardu_wiki_dictionaries():
    import requests
    from bs4 import BeautifulSoup

    base_url = "https://sardu.wiki"
    category_url = f"{base_url}/index.php/Category:Lemas"
    session = requests.Session()

    # Standard and non-standard Campidanese are kept apart
    standard, non_standard = set(), set()
    page_num = 1

    # Follow the "next page" link for as long as there is one
    while category_url:
        print(f"\nScraping category page {page_num}...")
        response = session.get(category_url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Could not load category page: {category_url}")
            break

        soup = BeautifulSoup(response.text, 'html.parser')

        # Links to the lemma pages (internal links only)
        lemma_links = []
        for group in soup.find_all('div', class_='mw-category-group'):
            for a in group.find_all('a', href=True):
                if a['href'].startswith('/index.php/'):
                    lemma_links.append(base_url + a['href'])

        for i, url in enumerate(lemma_links, 1):
            print(f"Scraping lemma {i}/{len(lemma_links)}...", end="\r")
            std_words, non_std_words = scrape_sardu_wiki_lemma(url, session)
            standard.update(std_words)
            non_standard.update(non_std_words)
            time.sleep(REQUEST_DELAY)

        category_url = None
        for a in soup.find_all('a', href=True):
            if 'pagefrom=' in a['href'] and 'next' in a.text.lower():
                category_url = base_url + a['href']
                break
        page_num += 1

    save_dictionary(STANDARD_DICT, standard)
    save_dictionary(NON_STANDARD_DICT, non_standard)


# ==========================================
# STATISTICS
# ==========================================

def remove_accents(word):
    nfkd_form = unicodedata.normalize('NFKD', word)
    return "".join(c for c in nfkd_form if not unicodedata.combining(c))


def print_statistics():
    wikipedia = load_dictionary(WIKIPEDIA_DICT)
    ditzionariu = load_dictionary(DITZIONARIU_DICT)
    standard = load_dictionary(STANDARD_DICT)
    non_standard = load_dictionary(NON_STANDARD_DICT)
    corpus = load_dictionary(CORPUS_DICT)

    # Dictionaries coming from lexicographic sources
    reference = ditzionariu | standard | non_standard

    print("--- DICTIONARY SIZES ---")
    print(f"Wikipedia:                {len(wikipedia)} words")
    print(f"Ditzionariu in linia:     {len(ditzionariu)} words")
    print(f"Campidanese standard:     {len(standard)} words")
    print(f"Campidanese non standard: {len(non_standard)} words")
    print(f"Training corpus:          {len(corpus)} words\n")

    print("--- COMBINED ---")
    print(f"Unique words in the reference dictionaries: {len(reference)}")
    print(f"Unique words overall:                       {len(reference | wikipedia | corpus)}\n")

    for name, words in (("Wikipedia", wikipedia), ("Training corpus", corpus)):
        covered = words & reference
        print(f"--- {name.upper()} COVERAGE ---")
        print(f"Words found in the reference dictionaries:     {len(covered)} ({len(covered) / len(words):.2%})")
        print(f"Words not found in the reference dictionaries: {len(words - reference)}")

        # Same comparison ignoring accents
        words_no_acc = {remove_accents(w) for w in words}
        reference_no_acc = {remove_accents(w) for w in reference}
        covered_no_acc = words_no_acc & reference_no_acc
        print(f"Ignoring accents:                              {len(covered_no_acc)} ({len(covered_no_acc) / len(words_no_acc):.2%})\n")


SOURCES = {
    "corpus": build_corpus_dictionary,
    "sardu-wiki": build_sardu_wiki_dictionaries,
    "ditzionariu": build_ditzionariu_dictionary,
    "wikipedia": build_wikipedia_dictionary,
    "stats": print_statistics,
}


def main():
    parser = argparse.ArgumentParser(description="Build the Sardinian dictionaries.")
    parser.add_argument("source", choices=SOURCES.keys())
    args = parser.parse_args()
    SOURCES[args.source]()


if __name__ == "__main__":
    main()
