import logging
from cluster.embedding import EmbedSentence
from cluster.terms import read_terms, Embeddings

log = logging.getLogger(__name__)


def repl(embed, embeddings):
    print("Number of words: {}".format(len(embed)))
    print("Dimension: {}".format(embed.dim))
    print("Number of search terms: {}".format(len(embeddings)))
    cmd = None
    while True:
        cmd = input('>> ')
        if cmd == ":q":
            break
        print(embeddings.most_similar_cosine(cmd)[:10])


if __name__ == "__main__":
    embed = EmbedSentence('data/cc.nl.300.magnitude')
    terms = read_terms("data/segmentatie_all_segmentation_files_v3.csv")
    sample = terms.sample(5000)
    embeddings = Embeddings.compute(sample, embed)

    repl(embed, embeddings)
