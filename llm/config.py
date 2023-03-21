import os

from loguru import logger
from pydantic import BaseSettings


class GlobalConfig(BaseSettings):
    DATASET_PATH: str = "datasets"
    MODEL_PATH: str = "models"
    MODEL_VERSION: str = "0.0.1dev1"
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
    BPEMB_VS: int = 200000
    BPEM_DIM: int = 300
    MODEL_DIMENSION: int = 100
    CBOW_WINDOW: int = 5

    class Config:
        env_file: str = ".env"


config = GlobalConfig()
