"""
Synthetic spelling error generator for Campidanese Sardinian.

Usage (prints a few corrupted versions of a sentence):
    python -m src.data_prep.synth_errors "su trenu 'e casteddu arribàda a mesudì"
"""

import argparse
import random
import re
import unicodedata

from src.config import ERROR_RATE, SEED, WORD_PATTERN
from src.rules import GRAPHEMATIC_RULES, KEYBOARD_NEIGHBOURS


class SyntheticErrorGenerator:
    def __init__(self, error_rate=ERROR_RATE, seed=None):
        self.error_rate = error_rate
        # Own random generator, so results depend only on the seed given here
        self.rng = random.Random(seed)

        ###
        """
            Custom error type distribution:
            - phonetic: applies phonetic confusion rules based on the Sardinian graphematic repertoire.
            - typo:     simulates keyboard typing errors.
            - accent:   removes the accent from the word.
        """
        self.error_types = ['phonetic', 'typo', 'accent']
        self.error_weights = [0.45, 0.35, 0.20]
        ###

        self.typos = KEYBOARD_NEIGHBOURS
        self.phonetic_rules = GRAPHEMATIC_RULES

        self.word_pattern = re.compile(WORD_PATTERN)

    def strip_accents(self, word: str) -> str:
        """ Removes the accent. """

        nfkd = unicodedata.normalize('NFKD', word)
        return "".join([char for char in nfkd if not unicodedata.combining(char)])

    def inject_phonetic_error(self, word: str) -> str:
        """ Replaces a grapheme using the rules based on the graphematic repertoire. """

        # Longer patterns first (a word containing 'tz' also contains 'z', so 'tz' gets priority)
        possible_targets = sorted(
            [t for t in self.phonetic_rules.keys() if t in word],
            key=len, reverse=True
        )
        if not possible_targets:
            return word

        weights = [len(t)**2 for t in possible_targets]
        chosen_target = self.rng.choices(possible_targets, weights=weights, k=1)[0]
        replacement = self.rng.choice(self.phonetic_rules[chosen_target])
        return word.replace(chosen_target, replacement, 1)

    def inject_keyboard_typo(self, word: str) -> str:
        """ Substitutes, inserts, deletes or swaps a character, simulating the keyboard. """

        oper = self.rng.choice(['sub', 'ins', 'del', 'inv'])
        idx = self.rng.randint(0, len(word) - 1)

        char = word[idx]

        if oper == 'sub' and char in self.typos:
            typo_char = self.rng.choice(self.typos[char])
            return word[:idx] + typo_char + word[idx+1:]

        elif oper == 'ins' and char in self.typos:
            typo_char = self.rng.choice(self.typos[char])
            return word[:idx] + typo_char + word[idx:]

        elif oper == 'del' and len(word) > 3:
            return word[:idx] + word[idx+1:]

        elif oper == 'inv' and idx < len(word) - 1:
            return word[:idx] + word[idx+1] + word[idx] + word[idx+2:]

        return word

    def corrupt_word(self, word: str) -> tuple:
        """ Applies one of the error types to a single word. Returns (corrupted word, error type). """

        choice = self.rng.choices(
            self.error_types,
            weights=self.error_weights,
            k=1
        )[0]

        if choice == 'phonetic':
            return self.inject_phonetic_error(word), choice
        elif choice == 'typo':
            return self.inject_keyboard_typo(word), choice
        else:
            return self.strip_accents(word), choice

    def corrupt_sentence(self, sentence: str) -> dict:
        """
        Processes a whole sentence, returning its corrupted version, the list of its
        words and, for each word, the Token Classification label and the error type.
        """
        corrupted_words = []
        labels = []
        error_types = []
        modifications = []

        for match in self.word_pattern.finditer(sentence):
            word = match.group(0)
            is_word = word.replace("'", "").replace("-", "").isalpha()

            noisy_word = word
            label = 0
            error_type = None

            if is_word and len(word) > 2 and self.rng.random() < self.error_rate:
                # Retry up to 5 times to "guarantee" that the word actually gets corrupted
                for _ in range(5):
                    candidate, candidate_type = self.corrupt_word(word)
                    if candidate != word:
                        noisy_word = candidate
                        error_type = candidate_type
                        break

                if noisy_word != word:
                    label = 1
                    modifications.append({
                        "start": match.start(),
                        "end": match.end(),
                        "new_word": noisy_word
                    })

            corrupted_words.append(noisy_word)
            labels.append(label)
            error_types.append(error_type)

        # Rebuild the string using the original indices (in reverse order)
        corrupted_text = sentence
        for mod in reversed(modifications):
            start = mod["start"]
            end = mod["end"]
            new_word = mod["new_word"]
            corrupted_text = corrupted_text[:start] + new_word + corrupted_text[end:]

        return {
            "original_text": sentence,
            "corrupted_text": corrupted_text,
            "corrupted_words_list": corrupted_words,
            "error_labels": labels,
            "error_types": error_types
        }


def main():
    parser = argparse.ArgumentParser(description="Show synthetic corruptions of a sentence.")
    parser.add_argument("sentence")
    parser.add_argument("--error-rate", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    corruptor = SyntheticErrorGenerator(error_rate=args.error_rate, seed=args.seed)
    original_words = corruptor.word_pattern.findall(args.sentence)

    print(f"Original:  {args.sentence}")
    for _ in range(args.samples):
        result = corruptor.corrupt_sentence(args.sentence)
        print(f"\nCorrupted: {result['corrupted_text']}")
        for original, corrupted, error_type in zip(original_words, result['corrupted_words_list'], result['error_types']):
            if error_type:
                print(f"  - '{original}' -> '{corrupted}' ({error_type})")


if __name__ == "__main__":
    main()
