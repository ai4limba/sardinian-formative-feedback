"""
Corrects one or more sentences from the command line.

Usage:
    python -m src.pipeline.correct "su trenu 'e casteddu arribàda a msesudì"
    python -m src.pipeline.correct --debug "..."       # shows candidates and scores
    python -m src.pipeline.correct --benchmark "..."   # also times each stage of the pipeline
"""

import argparse
import time

from src.pipeline.loader import load_spellchecker

EXAMPLE_SENTENCES = [
    "su trenu 'e casteddu arribàda a msesudì",                # 'msesudì' for 'mesudì'
    "va bene bènghiu cras e melì, candu bessu 'e traballai",  # 'bènghiu' for 'bèngiu', 'melì' for 'merì'
]


def benchmark_pipeline(pipeline, sentence, num_runs=5):
    """ Average execution time of the whole pipeline and of each of its stages. """
    orig_debug = pipeline.debug
    pipeline.debug = False

    # Total time
    start = time.perf_counter()
    for _ in range(num_runs):
        pipeline.correct_text(sentence)
    avg_total_time = (time.perf_counter() - start) / num_runs

    words = pipeline.split_text(sentence)

    # 1) Error detection
    start = time.perf_counter()
    for _ in range(num_runs):
        error_flags = pipeline.detect_errors(words, threshold=pipeline.detection_threshold)
    avg_detect_time = (time.perf_counter() - start) / num_runs

    avg_cand_time = 0.0
    avg_rank_time = 0.0
    candidates = []
    error_indices = [idx for idx, (word, is_error) in enumerate(zip(words, error_flags))
                     if is_error and word.replace("'", '').replace('-', '').isalpha()]

    # Stages 2 and 3 are timed on the first flagged word
    if error_indices:
        target_idx = error_indices[0]
        target_word = words[target_idx]

        # 2) Candidate generation
        start = time.perf_counter()
        for _ in range(num_runs):
            candidates = pipeline.generate_candidates(words, target_idx, target_word)
        avg_cand_time = (time.perf_counter() - start) / num_runs

        # 3) Candidate ranking
        if candidates:
            start = time.perf_counter()
            for _ in range(num_runs):
                pipeline.rank_candidates(words, target_idx, candidates, target_word)
            avg_rank_time = (time.perf_counter() - start) / num_runs

    pipeline.debug = orig_debug

    print(f"[BENCHMARK] '{sentence}'")
    print(f"{'Total time:':<30}{avg_total_time*1000:>8.2f} ms  (average over {num_runs} runs)")
    print(f"{'  - Error detection:':<30}{avg_detect_time*1000:>8.2f} ms")
    if error_indices:
        print(f"{'  - Candidate generation:':<30}{avg_cand_time*1000:>8.2f} ms")
        if candidates:
            print(f"{'  - Contextual ranking:':<30}{avg_rank_time*1000:>8.2f} ms  (candidates: {len(candidates)})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Correct Campidanese Sardinian sentences.")
    parser.add_argument("sentences", nargs="*", default=EXAMPLE_SENTENCES)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    pipeline = load_spellchecker(debug=args.debug)

    for sentence in args.sentences:
        print(f"Original:  {sentence}")
        print(f"Corrected: {pipeline.correct_text(sentence)}\n")

    if args.benchmark:
        # Warm-up, so the first measurement does not include GPU initialisation
        pipeline.correct_text(args.sentences[0])
        for sentence in args.sentences:
            benchmark_pipeline(pipeline, sentence)


if __name__ == "__main__":
    main()
