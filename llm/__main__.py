import click

from llm import data, models

@click.command()
@click.option(
    "-s",
    "--sample",
    help="Sample size",
    type=str,
    required=True,
)

def train(sample: str):
    texts, titles = data.load_corpus(sample=sample)
    texts = models.add_sentences_bounders(texts)
    titles = models.add_sentences_bounders(titles)
    models.train_embedding(texts + titles)
    embedder = models.load_embedder()


if __name__ == "__main__":
    train()
