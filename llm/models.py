import os

from basix import files
import numpy as np

from llm.config import config
from llm.embed import CBOWEmbedder
from llm.tokenize import SentencesTokenizer


class TextProcesser:
    def __init__(self):
        self.tokenizer = load_tokenizer()
        self.embedder = load_embedder()

    def tokenize(self, sentences):
        return self.tokenizer.encode(sentences)

    def tokenize_sentences(self, sentences):
        return self.tokenizer.encode_sentences(sentences)

    def get_vectors(self, sentences_tokens):
        return [
            np.array([self.embedder.get_vector(token) for token in sent_tokens])
            for sent_tokens in sentences_tokens
        ]

    def transform(self, sentences):
        tokens = self.tokenize(sentences)
        embeddings = self.get_vectors(tokens)
        return embeddings


def train_embedding(sentences):
    files.make_directory(config.MODEL_PATH)

    sent_tokenizer = SentencesTokenizer(
        tokenizer_level=config.TOKENIZER_LEVEL, lang=config.TEXT_LANGUAGE
    ).save(config.TOKENIZER_LOCAL_PATH)

    sent_tokens = sent_tokenizer.encode_sentences(sentences)

    embedder = (
        CBOWEmbedder(
            min_count=10,
            vector_size=50,
            window=5,
            alpha=0.025,
            min_alpha=0.0001,
            compute_loss=True,
        )
        .fit(
            sent_tokens, total_examples=len(sent_tokens), epochs=100, compute_loss=True
        )
        .save(config.EMBEDDER_LOCAL_PATH)
        .save_w2v(config.W2V_LOCAL_PATH)
    )


def load_tokenizer(path: str = config.TOKENIZER_LOCAL_PATH):
    if os.path.exists(path):
        return SentencesTokenizer.load(path)
    else:
        raise FileNotFoundError(
            f"Tokenizer not found in {path}. Experiment pass other path "
            "or training the model again."
        )


def load_embedder(path: str = config.EMBEDDER_LOCAL_PATH):
    if os.path.exists(path):
        return CBOWEmbedder.load(path)
    else:
        raise FileNotFoundError(
            f"Embedder not found in {path}. Experiment pass other path "
            "or training the model again."
        )


def load_w2v(path: str = config.W2V_LOCAL_PATH):
    if os.path.exists(path):
        return CBOWEmbedder.load_w2v(path)
    else:
        raise FileNotFoundError(
            f"Word2Vec bnary not found in {path}. Experiment pass other path "
            "or training the model again."
        )
