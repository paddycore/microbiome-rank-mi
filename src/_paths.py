"""Shared helpers: project paths and loading src/encodings.py without clashing with stdlib `encodings`."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def load_encodings():
    spec = importlib.util.spec_from_file_location("abundance_encodings", Path(__file__).with_name("encodings.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
