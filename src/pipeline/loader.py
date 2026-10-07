"""Single entry point to build the spellchecker from the trained models and the dictionaries."""

import json

from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification, AutoTokenizer

from src.config import DETECTOR_DIR, MLM_DIR, PIPELINE_DEFAULTS, PIPELINE_DICTIONARIES, PIPELINE_PARAMS_FILE
from src.pipeline.pipeline import SardinianSpellchecker
from src.utils import get_device


def load_vocabulary(paths=PIPELINE_DICTIONARIES) -> set:
    """ Merges the given dictionaries into the vocabulary used for candidate generation. """
    vocabulary = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            vocabulary.update(f.read().split())
    return vocabulary


def load_pipeline_params() -> dict:
    """ Inference parameters: the defaults, replaced by the tuned values when available. """
    params = dict(PIPELINE_DEFAULTS)
    if PIPELINE_PARAMS_FILE.exists():
        with open(PIPELINE_PARAMS_FILE, "r", encoding="utf-8") as f:
            params.update(json.load(f))
    return params


def load_spellchecker(mlm_dir=MLM_DIR, detector_dir=DETECTOR_DIR, debug=False, **overrides) -> SardinianSpellchecker:
    """ Loads tokenizer (shared by the two models), MLM, detector and vocabulary. """
    device = get_device()
    params = {**load_pipeline_params(), **overrides}

    tokenizer = AutoTokenizer.from_pretrained(mlm_dir)
    mlm = AutoModelForMaskedLM.from_pretrained(mlm_dir).to(device).eval()
    detector = AutoModelForTokenClassification.from_pretrained(detector_dir).to(device).eval()

    return SardinianSpellchecker(tokenizer, mlm, detector, load_vocabulary(), device, debug=debug, **params)
