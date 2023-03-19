from __future__ import annotations
from collections import Counter, defaultdict
from functools import reduce
from typing import List

import gensim
from gensim.models.callbacks import CallbackAny2Vec
from loguru import logger

from llm.utils import load_bin, save_bin
from llm.config import config


def get_embedder():
    pass


class CBOWEmbedder(gensim.models.Word2Vec):
    OOV_TOKEN = "<oov>"
    PADDING_TOKEN = "<pad>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    LP_TOKEN = "<lang>"

    def fit(
        self,
        corpus_iterable=None,
        corpus_file=None,
        total_examples=None,
        total_words=None,
        epochs=None,
        start_alpha=None,
        end_alpha=None,
        word_count=0,
        queue_factor=2,
        report_delay=1.0,
        compute_loss=False,
        callbacks=(),
        **kwargs,
    ):
        """Update the model's neural weights from a sequence of sentences.

        Notes
        -----
        To support linear learning-rate decay from (initial) `alpha` to `min_alpha`, and accurate
        progress-percentage logging, either `total_examples` (count of sentences) or `total_words` (count of
        raw words in sentences) **MUST** be provided. If `sentences` is the same corpus
        that was provided to :meth:`~gensim.models.word2vec.Word2Vec.build_vocab` earlier,
        you can simply use `total_examples=self.corpus_count`.

        Warnings
        --------
        To avoid common mistakes around the model's ability to do multiple training passes itself, an
        explicit `epochs` argument **MUST** be provided. In the common and recommended case
        where :meth:`~gensim.models.word2vec.Word2Vec.train` is only called once, you can set `epochs=self.epochs`.

        Parameters
        ----------
        corpus_iterable : iterable of list of str
            The ``corpus_iterable`` can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network, to limit RAM usage.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
            See also the `tutorial on data streaming in Python
            <https://rare-technologies.com/data-streaming-in-python-generators-iterators-iterables/>`_.
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `sentences` to get performance boost. Only one of `sentences` or
            `corpus_file` arguments need to be passed (not both of them).
        total_examples : int
            Count of sentences.
        total_words : int
            Count of raw words in sentences.
        epochs : int
            Number of iterations (epochs) over the corpus.
        start_alpha : float, optional
            Initial learning rate. If supplied, replaces the starting `alpha` from the constructor,
            for this one call to`train()`.
            Use only if making multiple calls to `train()`, when you want to manage the alpha learning-rate yourself
            (not recommended).
        end_alpha : float, optional
            Final learning rate. Drops linearly from `start_alpha`.
            If supplied, this replaces the final `min_alpha` from the constructor, for this one call to `train()`.
            Use only if making multiple calls to `train()`, when you want to manage the alpha learning-rate yourself
            (not recommended).
        word_count : int, optional
            Count of words already trained. Set this to 0 for the usual
            case of training on all words in sentences.
        queue_factor : int, optional
            Multiplier for size of queue (number of workers * queue_factor).
        report_delay : float, optional
            Seconds to wait before reporting progress.
        compute_loss: bool, optional
            If True, computes and stores loss value which can be retrieved using
            :meth:`~gensim.models.word2vec.Word2Vec.get_latest_training_loss`.
        callbacks : iterable of :class:`~gensim.models.callbacks.CallbackAny2Vec`, optional
            Sequence of callbacks to be executed at specific stages during training.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.models import Word2Vec
            >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
            >>>
            >>> model = Word2Vec(min_count=1)
            >>> model.build_vocab(sentences)  # prepare the model vocabulary
            >>> model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)  # train word vectors
            (1, 30)
        """

        self.vocab_freq = self.get_vocab(corpus_iterable, self.min_count)

        self._validate_oov()

        logger.debug("Building vocabulary from token frequencies")
        self.build_vocab_from_freq(self.vocab_freq)

        if callbacks == ():
            callbacks = (MonitorLossCallback(),)
        elif callbacks is None:
            callbacks = ()

        logger.debug("Training model")
        self.train(
            corpus_iterable=corpus_iterable,
            corpus_file=corpus_file,
            total_examples=total_examples,
            total_words=total_words,
            epochs=epochs,
            start_alpha=start_alpha,
            end_alpha=end_alpha,
            word_count=word_count,
            queue_factor=queue_factor,
            report_delay=report_delay,
            compute_loss=compute_loss,
            callbacks=callbacks,
            **kwargs,
        )
        return self

    def get_vector(self, key, norm=False):
        """Get the key's vector, as a 1D numpy array.

        Parameters
        ----------

        key : str
            Key for vector to return.
        norm : bool, optional
            If True, the resulting vector will be L2-normalized (unit Euclidean length).

        Returns
        -------

        numpy.ndarray
            Vector for the specified key.

        Raises
        ------

        KeyError
            If the given key doesn't exist.

        """
        try:
            return self.wv.get_vector(key, norm)

        except KeyError as exception:
            logger.warning(f"Returning OOV token: {self.OOV_TOKEN}. " + str(exception))
            return self.wv.get_vector(self.OOV_TOKEN, norm)

    def _validate_oov(self):
        if self.vocab_freq[self.OOV_TOKEN] == 0:
            raise ModelWithoutOOVError(
                "The model is very general, and there is no examples of "
                "out-of-vocabulary tokens. Try to increase the parameter "
                "min_count."
            )


    def save(self, path: str = config.EMBEDDER_LOCAL_PATH) -> None:
        logger.debug(f"Saving object {self.__class__.__name__} to {path}")
        save_bin(self, path)
        return self


    def save_w2v(self, path: str = config.W2V_LOCAL_PATH) -> None:
        logger.debug(f"Saving object {self.wv.__class__.__name__} to {path}")
        self.wv.save(path)
        return self


    @classmethod
    def load(cls, path: str = config.EMBEDDER_LOCAL_PATH) -> CBOWEmbedder:
        return load_bin(path)


    @classmethod
    def load_w2v(cls, path: str = config.W2V_LOCAL_PATH) -> gensim.models.KeyedVectors:
        return gensim.models.KeyedVectors.load(config.W2V_LOCAL_PATH)

    @classmethod
    def count_tokens(cls, corpus_iterable: List[List[str]], min_count: int):
        logger.debug("Counting tokens")
        counter_words = defaultdict(int)
        oov_value = 0
        for words in corpus_iterable:
            for word in words:
                counter_words[word] += 1
        for word, count in list(counter_words.items()):
            if count < min_count:
                oov_value += count
                del counter_words[word]
        counter_words[cls.OOV_TOKEN] = oov_value
        return counter_words

    @classmethod
    def get_vocab(cls, corpus_iterable: List[List[str]], min_count: int):
        logger.debug("Get frequency of tokens")
        vocab = {
            cls.OOV_TOKEN: -1,
            cls.PADDING_TOKEN: 0,
            cls.BOS_TOKEN: 1,
            cls.EOS_TOKEN: 2,
            cls.LP_TOKEN: 3,
        }
        counter_words = cls.count_tokens(corpus_iterable, min_count)
        vocab.update(counter_words)
        return vocab


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        ...

    def on_epoch_end(self, model):
        if self.epoch % 10 == 0:
            logger.debug("Epoch #{} end".format(self.epoch))
        self.epoch += 1


class MonitorLossCallback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        if self.epoch % 10 == 0:
            loss = model.get_latest_training_loss()
            loss_diff = loss - self.loss_to_be_subed
            self.loss_to_be_subed = loss
            logger.debug("Epoch #{} end. Loss: {}".format(self.epoch, loss_diff))
        self.epoch += 1


class ModelWithoutOOVError(Exception):
    pass
