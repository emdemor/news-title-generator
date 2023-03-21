from __future__ import annotations
from typing import List

from nltk import tokenize
from bpemb import BPEmb

from llm.utils import join_sentences, save_bin, load_bin
from llm.config import config, logger


class SentencesTokenizer:
    """ """

    allowed_languages = ["portugese"]

    def __init__(self, tokenizer_level, lang):
        self.tokenizer_level = tokenizer_level
        self.lang = lang
        self._tokenizer = self._set_tokenizer()

    
    def encode(self, text_list: List[str]):
        if isinstance(text_list, str):
            text_list = [text_list]

        return [self._tokenizer(text.lower()) for text in text_list]

    def encode_sentences(self, text_list):
        if isinstance(text_list, str):
            text_list = [text_list]

        corpus = join_sentences(text_list)
        data = []
        for i in tokenize.sent_tokenize(corpus.replace("\n", " "), language=self.lang):
            data.append([j.lower() for j in self._tokenizer(i)])
        return data


    def save(self, path: str = config.TOKENIZER_LOCAL_PATH) -> None:
        logger.debug(f"Saving object {self.__class__.__name__} to {path}")
        save_bin(self, path)
        return self


    @classmethod
    def load(cls, path: str = config.TOKENIZER_LOCAL_PATH) -> SentencesTokenizer:
        return load_bin(path)

    def _set_tokenizer(self):
        return self._get_bpemp_tokenizer(self.lang)

    def _get_bpemp_tokenizer(self, lang):
        if lang == "portuguese":
            _lang = "pt"
        else:
            raise Exception(
                f"Language {lang} not allowed for BPEmb tokenizer. "
                f"Allowed languages are: {self.allowed_languages}"
            )
        self._bpemb = BPEmb(lang=_lang, vs=config.BPEMB_VS, dim=config.BPEM_DIM)
        return self._bpemb.encode
