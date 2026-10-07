"""Central configuration: project paths, shared constants and default hyperparameters."""

from pathlib import Path

# ==========================================
# PATHS
# ==========================================
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DATASET = DATA_DIR / "datasets" / "campidanese.tsv"
CLEANED_DATASET = DATA_DIR / "datasets" / "campidanese_cleaned.txt"

TRAIN_CORPUS = DATA_DIR / "training" / "train_corpus.txt"
VAL_CORPUS = DATA_DIR / "training" / "val_corpus.txt"
TEST_CORPUS = DATA_DIR / "training" / "test_corpus.txt"

DICT_DIR = DATA_DIR / "dictionaries"
STANDARD_DICT = DICT_DIR / "campidanese_standard.txt"
NON_STANDARD_DICT = DICT_DIR / "campidanese_non_standard.txt"
CORPUS_DICT = DICT_DIR / "corpus_dict.txt"
DITZIONARIU_DICT = DICT_DIR / "ditzionariu_in_linia.txt"
WIKIPEDIA_DICT = DICT_DIR / "wikipedia_dict.txt"

# Dictionaries used for candidate generation. The two larger resources (Ditzionariu in linia
# and Wikipedia) mix all Sardinian varieties and are left out to keep candidates Campidanese.
PIPELINE_DICTIONARIES = [STANDARD_DICT, NON_STANDARD_DICT, CORPUS_DICT]

MODELS_DIR = ROOT / "models"
MLM_DIR = MODELS_DIR / "bert-sardinian-mlm"
DETECTOR_DIR = MODELS_DIR / "bert-sardinian-detector"

RUNS_DIR = ROOT / "runs"
PLOTS_DIR = RUNS_DIR / "plots"
PIPELINE_PARAMS_FILE = RUNS_DIR / "pipeline_params.json"

# Writing sessions saved by the application (local only, not versioned)
SESSIONS_DIR = ROOT / "sessions"

# ==========================================
# SHARED CONSTANTS
# ==========================================
SEED = 42

# Sardinian words (including leading/trailing and internal apostrophes and hyphens),
# numbers, and single punctuation marks
WORD_PATTERN = r"['\-]?[a-zA-Zàèìòùáéíóú]+(?:['\-][a-zA-Zàèìòùáéíóú]+)*['\-]?|\d+|[^a-zA-Z0-9àèìòùáéíóú\s]"

# Share of words corrupted when building synthetic parallel data
ERROR_RATE = 0.15

# ==========================================
# DEFAULT HYPERPARAMETERS
# ==========================================
MLM_DEFAULTS = {
    "base_model": "dbmdz/bert-base-italian-uncased",
    "epochs": 50,
    "batch_size": 16,
    "grad_accum": 1,          # Effective batch size = batch_size * grad_accum
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "mlm_probability": 0.15,
    "max_seq_length": 512,    # BERT limit
    "patience": 4,
}

DETECTOR_DEFAULTS = {
    "epochs": 50,
    "batch_size": 16,
    "grad_accum": 1,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "max_seq_length": 512,
    "patience": 4,
    "val_copies": 6,          # Validation sentences are corrupted this many times with different errors
}

# Inference parameters, overridden by PIPELINE_PARAMS_FILE once the thresholds have been tuned
PIPELINE_DEFAULTS = {
    "detection_threshold": 0.95,
    "correction_threshold": -1.5,
    "max_levenshtein_distance": 1,
    "max_candidates": 20,
}

# ==========================================
# FEEDBACK AND APPLICATION
# ==========================================
# Suggestions shown for each flagged word
MAX_SUGGESTIONS_SHOWN = 5

# Language model gain (log-probability of the suggestion minus that of the written word)
# from which a suggestion is presented as fitting the sentence "better" or "much better"
CONTEXT_FIT_GOOD = 0.0
CONTEXT_FIT_STRONG = 2.0

APP_HOST = "127.0.0.1"
APP_PORT = 8000

# Search space for src/evaluation/tune_thresholds.py
GRID = {
    "detection_threshold": [0.75, 0.8, 0.85, 0.9, 0.95],
    "correction_threshold": [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
    "max_levenshtein_distance": [1, 2, 3],
}
