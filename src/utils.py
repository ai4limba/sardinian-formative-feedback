"""Small helpers shared by the training and evaluation scripts."""

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import ROOT


def set_seed(seed: int):
    """ Seeds every random number generator involved in training and evaluation. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def use_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def relative_to_root(path: Path) -> str:
    """ Path as stored in the run records: relative to the project root, forward slashes. """
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_lines(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def append_record(csv_path: Path, record: dict):
    """ Appends one row to a run-history CSV, creating the file with its header if needed. """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([record])
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
