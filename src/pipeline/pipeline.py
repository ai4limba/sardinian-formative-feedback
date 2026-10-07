import re

import jellyfish
import torch
from symspellpy import SymSpell, Verbosity

from src.config import WORD_PATTERN


class SardinianSpellchecker:
    """
    Hybrid spelling correction system for the Sardinian language.
    Combines a BERT error detector, SymSpell for candidate generation (structural)
    and a BERT MLM for contextual ranking.
    """

    def __init__(self, tokenizer, mlm, detector, vocabulary, device,
                 detection_threshold, correction_threshold,
                 max_levenshtein_distance, max_candidates, debug=False):
        """ Initialises the correction pipeline. """
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.detector = detector
        self.vocabulary = vocabulary
        self.device = device
        self.detection_threshold = detection_threshold
        self.correction_threshold = correction_threshold
        self.max_levenshtein_distance = max_levenshtein_distance
        self.max_candidates = max_candidates
        self.debug = debug

        # SymSpell dictionary (built with max distance 3 so that the distance can be changed
        # afterwards, e.g. during a grid search; lookups still use the value of the parameter)
        symspell_capacity = max(3, self.max_levenshtein_distance)
        self.sym_spell = SymSpell(max_dictionary_edit_distance=symspell_capacity, prefix_length=15)
        for word in self.vocabulary:
            self.sym_spell.create_dictionary_entry(word, 1)

        # Regular expression identifying Sardinian words (including apostrophes and internal hyphens)
        self.word_pattern = re.compile(WORD_PATTERN)

    def split_text(self, text):
        """ Splits the text into words and tokens, keeping punctuation intact. """
        return self.word_pattern.findall(text)

    def detect_errors(self, words, threshold):
        """ Token Classification to detect misspelled words in context. """
        inputs = self.tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        # The detector labels every token as (0: CORRECT, 1: ERROR)
        with torch.no_grad():
            outputs = self.detector(**inputs)
            # Softmax probabilities from the logits
            probs = torch.softmax(outputs.logits, dim=-1)
            # Probability of the ERROR class (index 1)
            error_probs = probs[0, :, 1].tolist()

        word_ids = inputs.word_ids()
        error_flags = [False] * len(words)

        # A word is flagged as soon as one of its sub-tokens goes above the threshold
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and error_probs[idx] > threshold:
                error_flags[word_idx] = True

        return error_flags

    def generate_candidates(self, words, idx, original_word):
        """ Candidate generation with SymSpell (O(1) lookup). """
        original_word = original_word.lower()

        possible_targets = self.sym_spell.lookup(original_word, Verbosity.ALL, max_edit_distance=self.max_levenshtein_distance)

        # Suggested terms with their distance, closest first. Ties are broken alphabetically:
        # SymSpell returns them in an order that changes from run to run, and with the cap
        # below that would make the set of candidates (and the results) non-reproducible
        candidates_with_dist = [(t.term, t.distance) for t in possible_targets]
        candidates_with_dist.sort(key=lambda x: (x[1], x[0]))

        # Drop the original word (already lowercase) from the candidates, if present
        all_candidates_with_dist = [(word, dist) for word, dist in candidates_with_dist if word != original_word]

        if not all_candidates_with_dist:
            return []

        # Keep only the candidates at the minimum distance found, or at most one more (min_dist + 1)
        min_dist = all_candidates_with_dist[0][1]  # Already sorted, so the first one is the minimum
        valid_cands = [
            (word, dist) for word, dist in all_candidates_with_dist
            if dist <= min_dist + 1
        ]

        # Remove duplicates and cap the number of candidates to preserve MLM performance
        seen = set()
        final_candidates = []
        for word, dist in valid_cands:
            if word not in seen:
                seen.add(word)
                final_candidates.append(word)
                if len(final_candidates) >= self.max_candidates:
                    break

        return final_candidates

    def rank_candidates(self, words, error_idx, candidates, original_word):
        """
        Contextual ranking with the Masked Language Model.
        Computes the direct log-likelihood of each candidate and returns the candidates
        sorted with a hybrid logic: (1) Damerau-Levenshtein distance, (2) MLM score.
        """
        if not candidates:
            return float('-inf'), []

        # One sentence per target: the original word first, then each candidate
        all_targets = [original_word] + candidates
        batch_sentences = []
        for target in all_targets:
            temp_words = words.copy()
            temp_words[error_idx] = target
            batch_sentences.append(temp_words)

        inputs = self.tokenizer(
            batch_sentences,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.mlm(**inputs)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)

        scores = []

        # Score of each target: mean log-probability of the sub-tokens that make up the word
        for batch_idx, target in enumerate(all_targets):
            word_ids = inputs.word_ids(batch_index=batch_idx)
            token_scores = []

            for i, word_idx in enumerate(word_ids):
                if word_idx == error_idx:
                    token_id = inputs["input_ids"][batch_idx, i]
                    token_scores.append(log_probs[batch_idx, i, token_id].item())

            score = sum(token_scores) / len(token_scores) if token_scores else float('-inf')
            scores.append(score)

        original_score = scores[0]

        # Final hybrid ordering
        scored_candidates = []
        for i, cand in enumerate(candidates):
            cand_score = scores[i + 1]
            dist = jellyfish.damerau_levenshtein_distance(original_word.lower(), cand.lower())

            ###
            if self.debug:
                print(f"    * Candidate score '{cand}' (dist {dist}): {cand_score:.4f}")
            ###

            scored_candidates.append({"word": cand, "score": cand_score, "dist": dist})

        scored_candidates.sort(key=lambda x: (x["dist"], -x["score"]))

        return original_score, scored_candidates

    def process_text(self, input_text):
        """
        Core method: runs the whole analysis and extracts the complete "chunks".
        Returns a list of dictionaries that represents 100% of the original string.
        """
        # finditer also gives the exact (start, end) indices
        matches = list(self.word_pattern.finditer(input_text))
        words = [m.group() for m in matches]

        # Error detection
        error_flags = self.detect_errors(words, threshold=self.detection_threshold)

        analysis = []
        last_end = 0

        for idx, match in enumerate(matches):
            word = match.group()
            start, end = match.span()

            # Whitespace between tokens
            if start > last_end:
                whitespace_chunk = input_text[last_end:start]
                analysis.append({
                    "word": whitespace_chunk,
                    "is_error": False,
                    "status": "ok",
                    "candidates": [],
                    "start": last_end,
                    "end": start
                })

            # The analysed token. Status: "ok", "suggestion" (flagged, with candidates)
            # or "unverified" (doubtful for the detector, but without an accepted candidate)
            is_error = error_flags[idx]
            word_data = {
                "word": word,
                "is_error": False,
                "status": "ok",
                "candidates": [],
                "start": start,
                "end": end
            }

            # Basic filter: only words made of letters are corrected
            if is_error and word.replace("'", "").replace("-", "").isalpha():
                word_data["status"] = "unverified"

                ###
                if self.debug:
                    print(f"\n[DEBUG] Analysing error on: '{word}'")
                ###

                # Candidate generation
                candidates = self.generate_candidates(words, idx, word)

                # Contextual ranking and final decision through the threshold
                if candidates:

                    ###
                    if self.debug:
                        print(f"  - Candidates:    {candidates}")
                    ###

                    original_score, scored_candidates = self.rank_candidates(words, idx, candidates, word)

                    if scored_candidates:
                        best_score = scored_candidates[0]["score"]
                        diff = best_score - original_score

                        ###
                        if self.debug:
                            print(f"  - Final comparison: '{scored_candidates[0]['word']}' ({best_score:.4f}) vs '{word}' ({original_score:.4f})")
                            print(f"  - Difference: {diff:.4f} (Threshold: {self.correction_threshold})")
                        ###

                        # The word is marked as an error if the gain exceeds the correction threshold
                        if diff > self.correction_threshold:
                            word_data["is_error"] = True
                            word_data["status"] = "suggestion"
                            # How much better each candidate fits the context than the written word
                            word_data["context_gains"] = [c["score"] - original_score for c in scored_candidates[:self.max_candidates]]

                            formatted_cands = []
                            for c in scored_candidates[:self.max_candidates]:
                                c_word = c["word"]
                                # Restore the original casing
                                if word.istitle():
                                    c_word = c_word.capitalize()
                                elif word.isupper():
                                    c_word = c_word.upper()
                                formatted_cands.append(c_word)

                            word_data["candidates"] = formatted_cands

                            ###
                            if self.debug:
                                print(f"  >>> CORRECTED TO: '{formatted_cands[0]}'\n")
                            ###

                        ###
                        elif self.debug:
                            print(f"  >>> DISCARDED (insufficient improvement)\n")
                        ###

            analysis.append(word_data)
            last_end = end

        # Any characters left at the end of the string
        if last_end < len(input_text):
            final_chunk = input_text[last_end:]
            analysis.append({
                "word": final_chunk,
                "is_error": False,
                "status": "ok",
                "candidates": [],
                "start": last_end,
                "end": len(input_text)
            })

        return analysis

    def correct_text(self, input_text):
        """ Full correction pipeline, returning only the corrected string. """
        return self.apply_corrections(input_text, self.process_text(input_text))

    @staticmethod
    def apply_corrections(input_text, analysis):
        """ Replaces every flagged word of an analysis with its best candidate. """
        output_text = input_text
        # Go backwards so replacements of a different length do not shift the indices
        for chunk in reversed(analysis):
            if chunk["is_error"] and chunk["candidates"]:
                best_match = chunk["candidates"][0]
                start = chunk["start"]
                end = chunk["end"]
                output_text = output_text[:start] + best_match + output_text[end:]

        return output_text
