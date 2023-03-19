import os

import click

from llm.config import config
from llm import data

def main():
    corpus = data.load_corpus()
    print(len(corpus))
    print(config)

if __name__ == "__main__":
    main()
