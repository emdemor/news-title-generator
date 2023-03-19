from typing import Union
import nltk

from llm import data, models


def train(sample: Union[str, int]):

    texts, titles = data.load_corpus(sample=50000)
    models.train_embedding(texts + titles)
    embedder = models.load_embedder()


if __name__ == "__main__":
    nltk.download("punkt", download_dir="datasets/nltk_data")
    train()
