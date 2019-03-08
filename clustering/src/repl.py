import pymagnitude as m
import numpy as np


def embeddings():
    return m.Magnitude('data/cc.nl.300.magnitude')


def main():
    vecs = embeddings()
    print("Number of words: {}".format(len(vecs)))
    print("Dimension: {}".format(vecs.dim))
    embed(vecs, 'uitvaartverzekering')


def embed(vecs, sentence):
    """Compute the embedding for a sentence."""
    embedding = np.zeros(300)
    for w in sentence:
        embedding = np.add(embedding, vecs.query(w))
    return embedding


def test(vecs, sentence, linear_map=lambda e: e[0]):
    sentence_embedding = embed(vecs, sentence)
    x0 = 0
    for w in sentence:
        e = vecs.query(w)
        x0 += linear_map(e)
    print(x0, linear_map(sentence_embedding))


def repl(vecs):
    print("Number of words: {}".format(len(vecs)))
    print("Dimension: {}".format(vecs.dim))
    cmd = None
    while True:
        cmd = input('>> ')
        if cmd == ":q":
            break
        print(embed(vecs, cmd))


if __name__ == "__main__":
    vecs = embeddings()
    repl(vecs)
