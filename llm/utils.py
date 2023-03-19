from typing import Any
import joblib


def join_sentences(sentences):
    return " ".join(sentences)


def save_bin(obj: Any, path: str) -> None:
    joblib.dump(obj, path)


def load_bin(path: str) -> Any:
    return joblib.load(path)
