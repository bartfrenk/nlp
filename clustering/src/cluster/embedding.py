from pymagnitude import Magnitude
import numpy as np


class EmbedSentence:
    def __init__(self, path):
        """Create an instance.
        :param path: The location of the .magnitude file
        """
        self._path = path
        self.__magnitude = None

    @property
    def _magnitude(self):
        if self.__magnitude is None:
            self.__magnitude = Magnitude(self._path)
        return self.__magnitude

    @property
    def dim(self):
        return self._magnitude.dim

    def __len__(self):
        return len(self._magnitude)

    def query(self, sentence):
        """Compute the embedding for a sentence.
        :param sentence: A list of words
        """
        embedding = np.zeros(300)
        for w in sentence.split():
            embedding = np.add(embedding, self._magnitude.query(w))
        return embedding

    def __call__(self, sentence):
        return self.query(sentence)
