from ast import Tuple
import os
from typing import List

from basix import files
import numpy as np
from tqdm import tqdm
from loguru import logger
from tensorflow.keras.preprocessing.sequence import pad_sequences

from llm.config import config
from llm.embed import CBOWEmbedder
from llm.tokenize import SentencesTokenizer


class TextProcesser:
    def __init__(self, verbose=0):
        self.tokenizer = load_tokenizer()
        self.embedder = load_embedder()
        self.verbose = verbose

    def tokenize(self, sentences):
        return self.tokenizer.encode(sentences)

    def tokenize_sentences(self, sentences):
        return self.tokenizer.encode_sentences(sentences)

    def get_vectors(self, sentences_tokens):
        return [
            np.array([self.embedder.get_vector(token) for token in sent_tokens])
            for sent_tokens in self._get_iterator(sentences_tokens)
        ]

    def transform(
        self, sentences, add_bos=False, add_eos=False, padding="post", dtype="float32", maxlen=None
    ) -> np.ndarray:
        logger.debug("Tokenizing sentences")
        tokens = self.tokenize(sentences)

        if add_bos:
            tokens = self._add_bos(tokens)

        if add_eos:
            tokens = self._add_eos(tokens)

        logger.debug("Getting embedding vectors")
        embeddings = self.get_vectors(tokens)
        emb_padded = pad_sequences(embeddings, padding=padding, dtype=dtype, maxlen=maxlen)
        return emb_padded

    def get_most_similar_token(self, vector: np.array) -> str:
        return self.embedder.wv.similar_by_vector(vector, 1)[0][0]

    def get_tokens_from_vectors(self, vectors) -> List[List[str]]:
        if isinstance(vectors, np.ndarray):
            vectors = [vectors]

        return [[self.get_most_similar_token(vector) for vector in sent_vector] for sent_vector in vectors]

    def _get_iterator(self, iterator):
        if self.verbose == 0:
            return iterator
        else:
            return tqdm(iterator)

    def _add_bos(self, tokenized_sent_list: List[List[str]]) -> List[List[str]]:
        return [[CBOWEmbedder.BOS_TOKEN] + sent_tokens for sent_tokens in tokenized_sent_list]

    def _add_eos(self, tokenized_sent_list: List[List[str]]) -> List[List[str]]:
        return [sent_tokens + [CBOWEmbedder.EOS_TOKEN] for sent_tokens in tokenized_sent_list]


def train_embedding(sentences):
    
    files.make_directory(config.MODEL_PATH)

    sent_tokenizer = SentencesTokenizer(
        tokenizer_level=config.TOKENIZER_LEVEL, lang=config.TEXT_LANGUAGE
    ).save(config.TOKENIZER_LOCAL_PATH)

    sent_tokens = sent_tokenizer.encode_sentences(sentences)

    embedder = (
        CBOWEmbedder(
            min_count=config.CBOW_MIN_COUNT,
            vector_size=config.CBOW_VECTOR_SIZE,
            window=config.CBOW_WIDOWS,
            alpha=config.CBOW_ALPHA,
            min_alpha=config.CBOW_MIN_ALPHA,
            compute_loss=config.CBOW_COMPUTE_LOSS,
        )
        .fit(
            sent_tokens,
            total_examples=len(sent_tokens),
            epochs=config.CBOW_FIT_EPOCHS,
            compute_loss=config.CBOW_COMPUTE_LOSS,
            word_count=config.CBOW_FIT_WORD_COUNT,
        )
        .save(config.EMBEDDER_LOCAL_PATH)
        .save_w2v(config.W2V_LOCAL_PATH)
    )


def load_tokenizer(path: str = config.TOKENIZER_LOCAL_PATH):
    if os.path.exists(path):
        return SentencesTokenizer.load(path)
    else:
        raise FileNotFoundError(
            f"Tokenizer not found in {path}. Experiment pass other path " "or training the model again."
        )


def load_embedder(path: str = config.EMBEDDER_LOCAL_PATH):
    if os.path.exists(path):
        return CBOWEmbedder.load(path)
    else:
        raise FileNotFoundError(
            f"Embedder not found in {path}. Experiment pass other path " "or training the model again."
        )


def load_w2v(path: str = config.W2V_LOCAL_PATH):
    if os.path.exists(path):
        return CBOWEmbedder.load_w2v(path)
    else:
        raise FileNotFoundError(
            f"Word2Vec binary not found in {path}. Experiment pass other path " "or training the model again."
        )
