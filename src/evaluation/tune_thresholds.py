"""
Step 4 - Grid search of the inference parameters on the validation set.

Tries every combination of detection threshold, correction threshold and maximum edit
distance, and keeps the one with the highest correction F0.5. The best parameters are
saved and used from then on by the evaluation, the command line tool and the GUI.

Usage:
    python -m src.evaluation.tune_thresholds
"""

import argparse
import itertools
import json

import pandas as pd
from tqdm import tqdm

from src.config import DETECTOR_DIR, ERROR_RATE, GRID, MLM_DIR, PIPELINE_PARAMS_FILE, RUNS_DIR, SEED, VAL_CORPUS
from src.evaluation.evaluate import build_parallel_set, compute_correction_metrics, predict
from src.pipeline.loader import load_spellchecker
from src.utils import read_lines


def main():
    parser = argparse.ArgumentParser(description="Tune the spellchecker thresholds on the validation set.")
    parser.add_argument("--mlm-dir", default=str(MLM_DIR))
    parser.add_argument("--detector-dir", default=str(DETECTOR_DIR))
    parser.add_argument("--error-rate", type=float, default=ERROR_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N sentences.")
    parser.add_argument("--no-record", action="store_true", help="Do not write anything to the runs folder.")
    args = parser.parse_args()

    spellchecker = load_spellchecker(args.mlm_dir, args.detector_dir)

    sentences = read_lines(VAL_CORPUS)[:args.limit]
    rows = build_parallel_set(sentences, error_rate=args.error_rate, seed=args.seed)

    names = list(GRID.keys())
    grid = list(itertools.product(*GRID.values()))
    print(f"Grid search over {len(grid)} combinations on {len(rows)} validation sentences\n")

    results = []
    for values in tqdm(grid, desc="Grid search"):
        params = dict(zip(names, values))
        for name, value in params.items():
            setattr(spellchecker, name, value)

        predicted_texts, _, _ = predict(spellchecker, rows, show_progress=False)
        metrics = compute_correction_metrics(rows, predicted_texts, spellchecker.split_text)

        results.append({
            **params,
            "F0.5": round(metrics["Correction F0.5"], 4),
            "Precision": round(metrics["Correction Precision"], 4),
            "Recall": round(metrics["Correction Recall"], 4),
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "FN": metrics["FN"]
        })

    df_results = pd.DataFrame(results).sort_values(by="F0.5", ascending=False).reset_index(drop=True)
    print("\n[TOP 10 COMBINATIONS]")
    print(df_results.head(10).to_string())

    best = df_results.iloc[0]
    best_params = {
        "detection_threshold": float(best["detection_threshold"]),
        "correction_threshold": float(best["correction_threshold"]),
        "max_levenshtein_distance": int(best["max_levenshtein_distance"]),
    }
    print(f"\nBest parameters: {best_params}")
    print(f"Validation F0.5: {best['F0.5']:.4f} (Precision: {best['Precision']:.4f}, Recall: {best['Recall']:.4f})")

    if args.no_record:
        return

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(RUNS_DIR / "grid_search.csv", index=False)
    with open(PIPELINE_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"Full results saved to {RUNS_DIR / 'grid_search.csv'}")
    print(f"Best parameters saved to {PIPELINE_PARAMS_FILE}")


if __name__ == "__main__":
    main()
