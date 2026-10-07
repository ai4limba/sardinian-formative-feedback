"""
Step 5 - End-to-end evaluation of the spellchecker on synthetic errors.

Corrupts the held-out sentences with a fixed seed, runs the full pipeline and reports
detection metrics, correction metrics (precision, recall, F0.5), suggestion metrics
(is the right word among the candidates offered?), WER, CER and latency.

Usage:
    python -m src.evaluation.evaluate                  # test set
    python -m src.evaluation.evaluate --split val      # validation set
"""

import argparse
import difflib
import time

import jiwer
import pandas as pd
from sklearn.metrics import fbeta_score, precision_recall_fscore_support
from tqdm import tqdm

from src.config import DETECTOR_DIR, ERROR_RATE, MLM_DIR, RUNS_DIR, SEED, TEST_CORPUS, VAL_CORPUS
from src.data_prep.synth_errors import SyntheticErrorGenerator
from src.feedback.categorize import categorize
from src.pipeline.loader import load_spellchecker
from src.utils import append_record, read_lines, timestamp

SPLITS = {"val": VAL_CORPUS, "test": TEST_CORPUS}

# Sizes of the suggestion list for which the suggestion recall is reported
SUGGESTION_KS = (1, 3, 5)

# Feedback category expected for each synthetic error type
EXPECTED_CATEGORY = {"phonetic": "graphematic", "typo": "typing", "accent": "accent"}
FEEDBACK_CATEGORIES = ("accent", "graphematic", "typing", "contextual")


def build_parallel_set(sentences, error_rate=ERROR_RATE, seed=SEED):
    """ Corrupts every sentence, making sure that each one contains at least one error. """
    corruptor = SyntheticErrorGenerator(error_rate=error_rate, seed=seed)

    rows = []
    for clean_text in sentences:
        res = corruptor.corrupt_sentence(clean_text)
        attempts = 1
        while res["corrupted_text"] == clean_text and attempts < 20:
            res = corruptor.corrupt_sentence(clean_text)
            attempts += 1
        rows.append(res)
    return rows


def edit_distance(a, b):
    """ Levenshtein distance between two lists of words. """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for k in range(n + 1): dp[k][0] = k
    for k in range(m + 1): dp[0][k] = k
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]


def compute_correction_metrics(rows, predicted_texts, split_text):
    """ Word-level confusion matrix of the corrections and the resulting precision, recall and F0.5. """
    tp, fp, fn, tn = 0, 0, 0, 0

    for row, predicted_text in zip(rows, predicted_texts):
        orig_words = split_text(row["original_text"])
        corr_words = split_text(row["corrupted_text"])
        pred_words = split_text(predicted_text)

        # Word-by-word alignment (the three versions normally have the same number of tokens)
        if len(orig_words) == len(corr_words) == len(pred_words):
            for o, c, p in zip(orig_words, corr_words, pred_words):
                if c != o:  # The word was an error
                    if p == o:
                        tp += 1  # Corrected to the right word
                    elif p != c:
                        fp += 1; fn += 1  # Error noticed, but corrected to the wrong word
                    else:
                        fn += 1  # Error not detected, or correction discarded by the threshold
                else:  # The word was NOT an error
                    if p != o:
                        fp += 1  # Correct word handled as an error (over-correction)
                    else:
                        tn += 1  # Correct word left untouched
        else:
            # Fallback when the number of tokens differs: align original and prediction with difflib
            matcher = difflib.SequenceMatcher(None, orig_words, pred_words)

            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    for i in range(i1, i2):
                        if i < len(corr_words) and corr_words[i] != orig_words[i]:
                            tp += 1
                        else:
                            tn += 1
                else:
                    for i in range(i1, i2):
                        if i < len(corr_words) and corr_words[i] != orig_words[i]:
                            fn += 1

                    sub_corr = corr_words[i1:i2] if i2 <= len(corr_words) else []
                    sub_pred = pred_words[j1:j2]
                    fp += edit_distance(sub_corr, sub_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05 = (1 + 0.5**2) * precision * recall / ((0.5**2 * precision) + recall) if (precision + recall) > 0 else 0

    return {"Correction Precision": precision, "Correction Recall": recall, "Correction F0.5": f05,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def compute_detection_metrics(spellchecker, rows):
    """ Word-level metrics of the detector alone (before candidate generation and ranking). """
    true_labels, predicted_labels, flags_per_row = [], [], []

    for row in rows:
        flags = spellchecker.detect_errors(row["corrupted_words_list"], threshold=spellchecker.detection_threshold)
        true_labels.extend(row["error_labels"])
        predicted_labels.extend(int(flag) for flag in flags)
        flags_per_row.append(flags)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary', zero_division=0)
    f05 = fbeta_score(true_labels, predicted_labels, beta=0.5, average='binary', zero_division=0)

    metrics = {"Detection Precision": precision, "Detection Recall": recall, "Detection F1": f1, "Detection F0.5": f05}
    return metrics, flags_per_row


def compute_suggestion_metrics(rows, analyses, split_text, ks=SUGGESTION_KS):
    """
    Measures the spellchecker as a feedback tool rather than as an automatic corrector:
    an injected error counts as "flagged" when the pipeline marks the word and offers
    candidates, and as a hit at k when the right word is among the first k candidates.
    Returns the overall figures and, for every injected error, the rank of the right word.
    """
    errors, flagged, hits_any = 0, 0, 0
    hits = {k: 0 for k in ks}
    ranks_per_row = []

    for row, analysis in zip(rows, analyses):
        orig_words = split_text(row["original_text"])
        # The analysis also contains the whitespace between tokens
        token_chunks = [chunk for chunk in analysis if not chunk["word"].isspace()]
        aligned = len(orig_words) == len(token_chunks) == len(row["error_labels"])

        # None: error not flagged, 0: flagged but the right word is not offered, n: offered in position n
        ranks = {}
        for idx, label in enumerate(row["error_labels"]):
            if label == 0:
                continue
            errors += 1
            ranks[idx] = None
            if not aligned or not token_chunks[idx]["is_error"]:
                continue

            flagged += 1
            candidates = token_chunks[idx]["candidates"]
            rank = candidates.index(orig_words[idx]) + 1 if orig_words[idx] in candidates else 0
            ranks[idx] = rank
            if rank > 0:
                hits_any += 1
                for k in ks:
                    hits[k] += int(rank <= k)
        ranks_per_row.append(ranks)

    metrics = {"Errors": errors, "Flagged": flagged, "Flagged Rate": flagged / errors if errors else 0}
    for k in ks:
        metrics[f"Suggestion Recall@{k}"] = hits[k] / errors if errors else 0
    metrics["Suggestion Recall@All"] = hits_any / errors if errors else 0
    return metrics, ranks_per_row


def compute_error_type_breakdown(rows, flags_per_row, ranks_per_row, k=max(SUGGESTION_KS)):
    """
    For each synthetic error type: how many errors were injected, detected by the detector,
    automatically corrected (right word first) and suggested (right word in the first k).
    """
    breakdown = {}

    for row, flags, ranks in zip(rows, flags_per_row, ranks_per_row):
        for idx, error_type in enumerate(row["error_types"]):
            if error_type is None:
                continue
            counts = breakdown.setdefault(error_type, {"total": 0, "detected": 0, "corrected": 0, "suggested": 0})
            rank = ranks[idx]
            counts["total"] += 1
            counts["detected"] += int(flags[idx])
            counts["corrected"] += int(rank == 1)
            counts["suggested"] += int(rank is not None and 0 < rank <= k)

    return breakdown


def compute_category_agreement(rows, split_text, vocabulary):
    """
    Checks the feedback engine on its own: every injected error is paired with its right
    word, and the category assigned to the pair is compared with the type of error injected.
    Returns the agreement and the table {error type: {category: count}}.
    """
    table = {error_type: dict.fromkeys(FEEDBACK_CATEGORIES, 0) for error_type in EXPECTED_CATEGORY}
    agreed, total = 0, 0

    for row in rows:
        orig_words = split_text(row["original_text"])
        for orig_word, corrupted_word, error_type in zip(orig_words, row["corrupted_words_list"], row["error_types"]):
            if error_type is None:
                continue
            category = categorize(corrupted_word, orig_word, vocabulary)["category"]
            table[error_type][category] += 1
            agreed += int(category == EXPECTED_CATEGORY[error_type])
            total += 1

    return (agreed / total if total else 0), table


def predict(spellchecker, rows, show_progress=True):
    """ Runs the pipeline on the corrupted sentences. Returns predictions, analyses and latencies (ms). """
    predicted_texts, analyses, latencies = [], [], []
    for row in tqdm(rows, disable=not show_progress):
        start = time.perf_counter()
        analysis = spellchecker.process_text(row["corrupted_text"])
        predicted_texts.append(spellchecker.apply_corrections(row["corrupted_text"], analysis))
        latencies.append((time.perf_counter() - start) * 1000)
        analyses.append(analysis)
    return predicted_texts, analyses, latencies


def main():
    parser = argparse.ArgumentParser(description="Evaluate the spellchecker on synthetic errors.")
    parser.add_argument("--split", choices=SPLITS.keys(), default="test")
    parser.add_argument("--mlm-dir", default=str(MLM_DIR))
    parser.add_argument("--detector-dir", default=str(DETECTOR_DIR))
    parser.add_argument("--error-rate", type=float, default=ERROR_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N sentences.")
    parser.add_argument("--no-record", action="store_true", help="Do not write anything to the runs folder.")
    args = parser.parse_args()

    spellchecker = load_spellchecker(args.mlm_dir, args.detector_dir)

    sentences = read_lines(SPLITS[args.split])[:args.limit]
    rows = build_parallel_set(sentences, error_rate=args.error_rate, seed=args.seed)

    # Warm-up, so the first latency does not include GPU initialisation
    spellchecker.correct_text(rows[0]["corrupted_text"])

    predicted_texts, analyses, latencies = predict(spellchecker, rows)
    avg_latency_ms = sum(latencies) / len(latencies)

    correction = compute_correction_metrics(rows, predicted_texts, spellchecker.split_text)
    detection, flags_per_row = compute_detection_metrics(spellchecker, rows)
    suggestion, ranks_per_row = compute_suggestion_metrics(rows, analyses, spellchecker.split_text)
    breakdown = compute_error_type_breakdown(rows, flags_per_row, ranks_per_row)
    top_k = max(SUGGESTION_KS)
    category_agreement, category_table = compute_category_agreement(rows, spellchecker.split_text, spellchecker.vocabulary)

    # Word Error Rate (WER) and Character Error Rate (CER), with the corrupted text as baseline
    references = [row["original_text"] for row in rows]
    corrupted = [row["corrupted_text"] for row in rows]
    wer_score = jiwer.wer(references, predicted_texts)
    cer_score = jiwer.cer(references, predicted_texts)
    baseline_wer = jiwer.wer(references, corrupted)
    baseline_cer = jiwer.cer(references, corrupted)

    print(f"\n=== [{args.split.upper()} SET: {len(rows)} sentences] ===")
    print(f"Average latency: {avg_latency_ms:.2f} ms / sentence\n")

    print("=== [DETECTION] ===")
    for name, value in detection.items():
        print(f"{name + ':':<22}{value:.4f}")

    print("\n=== [END-TO-END CORRECTION] ===")
    print(f"Correction Precision: {correction['Correction Precision']:.4f}")
    print(f"Correction Recall:    {correction['Correction Recall']:.4f}")
    print(f"Correction F0.5:      {correction['Correction F0.5']:.4f}\n")

    print("=== [CONFUSION MATRIX] ===")
    print(f"{'':<16} | {'Predicted: ERR':<14} | {'Predicted: OK':<14} |")
    print("-" * 52)
    print(f"{'Actual: ERR':<16} | TP: {correction['TP']:<10} | FN: {correction['FN']:<10} |")
    print(f"{'Actual: OK':<16} | FP: {correction['FP']:<10} | TN: {correction['TN']:<10} |\n")

    print("=== [SUGGESTIONS] (share of the injected errors) ===")
    print(f"{'Flagged with candidates:':<26}{suggestion['Flagged Rate']:.4f}  ({suggestion['Flagged']} of {suggestion['Errors']})")
    for k in SUGGESTION_KS:
        print(f"{f'Right word in top {k}:':<26}{suggestion[f'Suggestion Recall@{k}']:.4f}")
    print(f"{'Right word in the list:':<26}{suggestion['Suggestion Recall@All']:.4f}\n")

    print("=== [BY ERROR TYPE] ===")
    for error_type, counts in breakdown.items():
        print(f"{error_type:<10} injected: {counts['total']:<5} "
              f"detected: {counts['detected'] / counts['total']:<8.2%} "
              f"corrected: {counts['corrected'] / counts['total']:<8.2%} "
              f"in top {top_k}: {counts['suggested'] / counts['total']:.2%}")

    print("\n=== [FEEDBACK CATEGORIES] (rows: injected error type, columns: category assigned) ===")
    print(f"{'':<10}" + "".join(f"{category:<13}" for category in FEEDBACK_CATEGORIES))
    for error_type, counts in category_table.items():
        print(f"{error_type:<10}" + "".join(f"{counts[category]:<13}" for category in FEEDBACK_CATEGORIES))
    print(f"Agreement with the injected type: {category_agreement:.4f}")

    print(f"\nWord Error Rate (WER):      {wer_score:.4f} (corrupted baseline: {baseline_wer:.4f})")
    print(f"Character Error Rate (CER):  {cer_score:.4f} (corrupted baseline: {baseline_cer:.4f})")

    if args.no_record:
        return

    run_id = timestamp()

    # Sentence-level predictions, for error analysis
    predictions_path = RUNS_DIR / f"predictions_{args.split}.csv"
    pd.DataFrame({
        "original_text": references,
        "corrupted_text": corrupted,
        "predicted_text": predicted_texts,
        "success": [ref == pred for ref, pred in zip(references, predicted_texts)],
        "latency_ms": [round(latency, 2) for latency in latencies]
    }).to_csv(predictions_path, index=False, encoding="utf-8")

    record = {
        "Timestamp": run_id,
        "Split": args.split,
        "Sentences": len(rows),
        "Seed": args.seed,
        "Error Rate": args.error_rate,
        "Detection Threshold": spellchecker.detection_threshold,
        "Correction Threshold": spellchecker.correction_threshold,
        "Max Levenshtein Distance": spellchecker.max_levenshtein_distance,
        "Max Candidates": spellchecker.max_candidates,
        "Avg Latency (ms)": round(avg_latency_ms, 2),
        **{name: round(value, 4) for name, value in detection.items()},
        "Correction F0.5": round(correction['Correction F0.5'], 4),
        "Correction Precision": round(correction['Correction Precision'], 4),
        "Correction Recall": round(correction['Correction Recall'], 4),
        "Flagged Rate": round(suggestion["Flagged Rate"], 4),
        **{name: round(value, 4) for name, value in suggestion.items() if name.startswith("Suggestion")},
        "Category Agreement": round(category_agreement, 4),
        "WER": round(wer_score, 4),
        "CER": round(cer_score, 4),
        "Baseline WER": round(baseline_wer, 4),
        "Baseline CER": round(baseline_cer, 4),
        "TP": correction['TP'],
        "FP": correction['FP'],
        "FN": correction['FN'],
        "TN": correction['TN'],
    }
    for error_type in ("phonetic", "typo", "accent"):
        counts = breakdown.get(error_type, {"total": 0, "detected": 0, "corrected": 0, "suggested": 0})
        record[f"{error_type.capitalize()} Errors"] = counts["total"]
        record[f"{error_type.capitalize()} Detected"] = counts["detected"]
        record[f"{error_type.capitalize()} Corrected"] = counts["corrected"]
        record[f"{error_type.capitalize()} In Top {top_k}"] = counts["suggested"]

    append_record(RUNS_DIR / "evaluation_records.csv", record)
    print(f"\nPredictions saved to {predictions_path}")
    print(f"Run recorded in {RUNS_DIR / 'evaluation_records.csv'}")


if __name__ == "__main__":
    main()
