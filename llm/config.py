import os

from loguru import logger
from pydantic import BaseSettings


class GlobalConfig(BaseSettings):
    DATASET_PATH: str = "datasets"
    MODEL_PATH: str = "models"
    MODEL_VERSION: str = "0.1.0"
    DATASET_RAW_PATH: str = os.path.join(DATASET_PATH, "raw")
    DATASET_INTERIM_PATH: str = os.path.join(DATASET_PATH, "interim")
    CORPUS_URL: str = "https://github.com/emdemor/News-of-the-Brazilian-Newspaper/blob/main/data/brazilian-news.parquet?raw=true"
    CORPUS_LOCAL_PATH: str = os.path.join(DATASET_RAW_PATH, "brazilian-news.parquet")
    TEXT_DATA_PATH: str = os.path.join(DATASET_RAW_PATH, "text.parquet")
    TITLE_DATA_PATH: str = os.path.join(DATASET_RAW_PATH, "title.parquet")
    TEXT_LANGUAGE: str = "portuguese"
    TOKENIZER_LEVEL: str = "subword"
    TOKENIZER_LOCAL_PATH: str = os.path.join(MODEL_PATH, f"version={MODEL_VERSION}", "tokenizer.bin")
    EMBEDDER_LOCAL_PATH: str = os.path.join(MODEL_PATH, f"version={MODEL_VERSION}", "embedder.bin")
    W2V_LOCAL_PATH: str = os.path.join(MODEL_PATH, f"version={MODEL_VERSION}", "w2v.bin")

    CBOW_MIN_COUNT: int = 10
    CBOW_VECTOR_SIZE: int = 50
    CBOW_WIDOWS: int = 5
    CBOW_ALPHA: float = 0.025
    CBOW_MIN_ALPHA: float = 0.0001
    CBOW_COMPUTE_LOSS: bool = True
    CBOW_FIT_EPOCHS: int = 100
    CBOW_FIT_WORD_COUNT: int = 0
    CBOW_FIT_START_ALPHA: int = None
    CBOW_FIT_END_ALPHA: int = None

    BPEMB_VS: int = 200000
    BPEMB_DIM: int = 300

    class Config:
        env_file: str = ".env"


config = GlobalConfig()
