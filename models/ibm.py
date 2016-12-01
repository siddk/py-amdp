"""
ibm.py

Abstract Class definition for IBM Models. Instantiates all the necessary shared parameters between
IBM Models (i.e. base word-to-word translation dictionary), vocabulary preprocessing.
"""
from collections import defaultdict

# Tolerance Probability - prevents underflow
MIN_PROB = 1.0e-12


class IBMModel():
    def __init__(self, parallel_corpus):
        """
        Instantiate an abstract IBM Model, with the given parallel corpus.

        :param parallel_corpus: List of tuples, where each tuple contains the following elements:
                    1) List of source sentence tokens.
                    2) List of target sentence tokens.
        """
        self.build_vocab(parallel_corpus)
        self.build_dictionaries()

    def build_vocab(self, parallel_corpus):
        """
        Builds the source and target vocabularies, using the given parallel corpus.

        :param parallel_corpus: List of tuples, where each tuple contains the following elements:
                    1) List of source sentence tokens.
                    2) List of target sentence tokens.
        """
        # Walk through Parallel Corpus and add to Vocabulary
        self.src_vocab, self.trg_vocab = set(), set()
        for (src, trg) in parallel_corpus:
            self.src_vocab.update(src)
            self.trg_vocab.update(trg)

        # Add Null (None) Word to Source Vocabulary
        self.src_vocab.add(None)

    def build_dictionaries(self):
        """
        Builds the core IBM Model count dictionaries, for ML estimation during EM. Builds two
        dictionaries, with the following format:

            - tau: dict[str][str]: float. Probability(target word | source word).
                   Values accessed as ``tau[target_word][source_word]``
            - delta: dict[int][int][int][int]: float. Probability(i | j,l,m).
                     Values accessed as ``delta[i][j][l][m]``.
                        i: Position in the source sentence
                           Valid values are 0 (for NULL), 1, 2, ..., length of source sentence
                        j: Position in the target sentence
                           Valid values are 1, 2, ..., length of target sentence
                        l: Number of words in the source sentence, excluding NULL
                        m: Number of words in the target sentence
        """
        self.tau = defaultdict(lambda: defaultdict(lambda: MIN_PROB))
        self.delta = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:
                                                                         defaultdict(lambda:
                                                                                     MIN_PROB))))


class Counts():
    """
    Auxiliary Object to store counts of various parameters during training.
    """
    def __init__(self):
        self.t_given_s = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.any_t_given_s = defaultdict(lambda: 0.0)

    def update_lexical_translation(self, count, alignment_info, j):
        i = alignment_info.alignment[j]
        t = alignment_info.trg_sentence[j]
        s = alignment_info.src_sentence[i]
        self.t_given_s[t][s] += count
        self.any_t_given_s[s] += count