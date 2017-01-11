"""
ibm1.py

Core implementation of IBM 1 Translation Model, used for burn-in translation probability training.
Learns probability of target word given source word, via EM and ML Estimation.
"""
from collections import defaultdict
from ibm import IBMModel, Counts, MIN_PROB


class IBM1(IBMModel):
    def __init__(self, parallel_corpus, iters):
        """
        Instantiate an IBM 1 Model, with the corpus to train on, as well as the number of EM
        training iterations.

        :param parallel_corpus: List of Tuples, where each tuple has two elements:
                    1) List of source sentence tokens
                    2) List of target sentence tokens
        :param iters: Number of EM Iterations for IBM 1 EM Training.
        """
        super(IBM1, self).__init__(parallel_corpus)

        # Set all probabilities as uniform to start.
        self.set_uniform()

        # Train for # iters EM iterations.
        for _ in range(iters):
            self.train()

    def set_uniform(self):
        """
        Set P(target | source translation) probabilities to be uniform --> 1 / len(trg_vocab).
        """
        initial = 1.0 / len(self.trg_vocab)
        for t in self.trg_vocab:
            self.tau[t] = defaultdict(lambda: initial)

    def train(self):
        """
        Perform EM Training for learning word-to-word translation probabilities.
            - Note that source and target get flipped (parallel corpus should be of the form
              language you are translating from, language you are translating to).
              IBM Translation models translation as noisy channel P(target | source).
        """
        counts = Counts()
        for pair in self.pc:
            src_sentence, trg_sentence = [None] + pair[1], pair[0]

            # E-Step (a) - Get Total Counts (for alignment normalization)
            total_count = self.alignment_marginals(src_sentence, trg_sentence)

            # E-Step (b) - Collect Normalized Counts
            for t in trg_sentence:
                for s in src_sentence:
                    count = self.tau[t][s]
                    normalized_count = count / float(total_count[t])
                    counts.t_given_s[t][s] += normalized_count
                    counts.any_t_given_s[s] += normalized_count

            # M-Step - Update Probabilities with ML Estimate
            for t, src_words in counts.t_given_s.items():
                for s in src_words:
                    estimate = counts.t_given_s[t][s] / counts.any_t_given_s[s]
                    self.tau[t][s] = max(estimate, MIN_PROB)

    def alignment_marginals(self, src_sentence, trg_sentence):
        """
        Compute the probability of all possible word alignments given target word t.

        :return: Probability of t for all s in ``src_sentence``
        """
        alignment_t = defaultdict(lambda: 0.0)
        for t in trg_sentence:
            for s in src_sentence:
                alignment_t[t] += self.tau[t][s]
        return alignment_t

