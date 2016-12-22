"""
neural_classifier.py

Core model definition code for Neural-Network Classifier Approach - Takes in a parallel corpus,
and trains/evaluates the model.

Architecture:
    - Embedding Layer
    - RNN (LSTM) => Fixed State Vector
    - ReLU Layer
    - Softmax Output over Commands (Enumerated 1 - N)
"""
import tensorflow as tf
UNK = "<<UNK>>"


class NNClassifier():
    def __init__(self, parallel_corpus, commands):
        """
        Instantiates and Trains Model using the given parallel corpus.

        :param parallel_corpus: List of Tuples, where each tuple has two elements:
                    1) List of source sentence tokens
                    2) List of target sentence tokens
        :param commands: List of Lists, where each element is one of the possible commands (labels)
        """
        self.commands, self.labels = commands, {x: i for (i, x) in enumerate(commands)}
        self.pc = parallel_corpus

        # Build Vocabulary
        self.word2id, self.id2word = self.build_vocabulary()
        self.max_len, self.lengths = self.get_lengths()

        # Setup Placeholders
        self.X = tf.placeholder(tf.int32, shape=[None, self.max_len])

    def build_vocabulary(self):
        """
        Builds the vocabulary from the parallel corpus, adding the UNK ID.

        :return: Tuple of Word2Id, Id2Word Dictionaries.
        """
        vocab = {UNK}
        for n, _ in self.pc:
            for word in n:
                vocab.add(word)

        id2word = list(vocab)
        word2id = {id2word[i]: i for i in range(len(id2word))}
        return word2id, id2word

    def get_lengths(self):
        """
        Read through the parallel corpus, get maximum lengths, and individual command lengths.

        :return: Tuple of max_length, sequence lengths.
        """
        max_len, lengths = 0, []
        for n, _ in self.pc:
            if len(n) > max_len:
                max_len = len(n)
            lengths.append(len(n))
        return max_len, lengths