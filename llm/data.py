import os
from typing import Tuple, Union, List

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from basix import files

from llm.config import config, logger
from llm.tokenize import SentencesTokenizer


def load_corpus(
    force_download: bool = False, sample: Union[None, int] = None
) -> Tuple[List[str]]:
    if (not os.path.exists(config.CORPUS_LOCAL_PATH)) or force_download:
        logger.info("Downloading corpus")
        _download_corpus()
    else:
        logger.debug(
            f"Corpus already exists in {config.CORPUS_LOCAL_PATH}. Skipping "
            "downloading corpus."
        )

    logger.debug(f"Importing news from {config.CORPUS_LOCAL_PATH}")
    df = _get_news()

    if sample is not None:
        logger.debug(f"Using a sample of size {sample}")
        df = _sample_news(df, size=sample)
    else:
        logger.debug("Using full dataset")

    logger.debug(f"Importing news titles from {config.TITLE_DATA_PATH}")
    titles = _get_titles(df)

    logger.debug(f"Importing news texts from {config.TEXT_DATA_PATH}")
    texts = _get_texts(df)

    return texts, titles


def tokenize_text_and_title_sentences(
    sent_tokenizer: SentencesTokenizer, texts: List[str], titles: List[str]
) -> List[List[str]]:
    text_sent_tokens = sent_tokenizer.encode(texts)
    title_sent_tokens = sent_tokenizer.encode(titles)
    sent_tokens = text_sent_tokens + title_sent_tokens
    return sent_tokens


def _download_corpus() -> None:
    files.make_directory(config.DATASET_RAW_PATH)
    files.download_file(config.CORPUS_URL, config.CORPUS_LOCAL_PATH)


def _get_news() -> pd.DataFrame:

    df = pd.read_parquet(
        config.CORPUS_LOCAL_PATH,
        columns=["title", "text"],
    )

    has_nan = df[["title", "text"]].isna().any(axis=1)

    return df[~has_nan]


def _sample_news(df, size: int = 1000) -> pd.DataFrame:
    return df.sample(size)


def _get_texts(df: pd.DataFrame) -> List[str]:
    return df["text"].to_list()


def _get_titles(df: pd.DataFrame) -> List[str]:
    return df["title"].to_list()
