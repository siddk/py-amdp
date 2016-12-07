"""
ibm2.py

Core implementation of IBM Model 2 Machine Translation code. First instantiates an IBM 2 Model
with the necessary Tau and Delta parameters, then instantiates an IBM 1 Model. Performs Burn-in
EM Inference via IBM 1, then performs IBM 2 EM to learn alignments.
"""
import math
import random
from collections import defaultdict
from ibm import IBMModel, Counts, MIN_PROB
from ibm1 import IBM1


class IBM2(IBMModel):
    def __init__(self, parallel_corpus, iters, burn_in_iters=10):
        """
        Instantiate an IBM 2 Model, with the corpus to train on, as well as the number of training
        and burn-in EM iterations.

        :param parallel_corpus: List of Tuples, where each tuple has two elements:
                    1) List of source sentence tokens
                    2) List of target sentence tokens
        :param iters: Number of EM Iterations for IBM 2 EM Training.
        :param burn_in_iters: Number of EM Iterations for IBM 1 EM Training.
        """
        super(IBM2, self).__init__(parallel_corpus)

        # Burn-In Translation Probabilities via IBM 1 Training, reset current probability matrices
        ibm1 = IBM1(parallel_corpus, burn_in_iters)
        self.tau = ibm1.tau

        self.build_eta()

        # Set alignment probabilities to be uniform to start
        self.set_uniform()

        # Perform IBM 2 EM Training
        for _ in range(iters):
            self.train()

    def set_uniform(self):
        """
        Set P(source position i | target position j, length of source l, length of target m)
        probabilities to be uniform -> delta(i | j,l,m) = 1 / (l+1) for all i, j, l, m.
        """
        l_m_combinations = set()
        for pair in self.pc:
            src_sentence, trg_sentence = pair[1], pair[0]
            l, m = len(src_sentence), len(trg_sentence)
            if (l, m) not in l_m_combinations:
                l_m_combinations.add((l, m))
                initial_prob = 1 / float(l + 1)

                for i in range(0, l + 1):
                    for j in range(1, m + 1):
                        self.delta[i][j][l][m] = initial_prob

    def train(self):
        """
        Perform EM Training for joint learning of word-to-word translation probabilities and
        alignment probabilities.
            - Note that source and target get flipped (parallel corpus should be of the form
              language you are translating from, language you are translating to).
              IBM Translation models translation as noisy channel P(target | source).
        """
        counts = Model2Counts()
        for pair in self.pc:
            # Target Sentence is 1-Indexed, Source Sentence get NULL word slot
            src_sentence, trg_sentence = [None] + pair[1], ['UNUSED'] + pair[0]
            l, m = len(pair[1]), len(pair[0])

            # E-Step (a) - Get Total Counts (for alignment normalization)
            total_count = self.alignment_marginals(src_sentence, trg_sentence)

            # E-Step (b) - Collect counts
            for j in range(1, m + 1):
                t = trg_sentence[j]
                for i in range(0, l + 1):
                    s = src_sentence[i]
                    count = self.tau[trg_sentence[j]][src_sentence[i]] * \
                            self.delta[i][j][len(src_sentence) - 1][len(trg_sentence) - 1]
                    normalized_count = count / total_count[t]
                    counts.update_lexical_translation(normalized_count, s, t)
                    counts.update_alignment(normalized_count, i, j, l, m)

            # M-Step - Update probabilities with maximum likelihood estimates
            # Maximize Tau Translation Probabilities
            for t, src_words in counts.t_given_s.items():
                for s in src_words:
                    estimate = counts.t_given_s[t][s] / counts.any_t_given_s[s]
                    self.tau[t][s] = max(estimate, MIN_PROB)

            # Maximize Delta Alignment Probabilities
            for i, j_s in counts.alignment.items():
                for j, src_sentence_lengths in j_s.items():
                    for l, trg_sentence_lengths in src_sentence_lengths.items():
                        for m in trg_sentence_lengths:
                            estimate = (counts.alignment[i][j][l][m] /
                                        counts.alignment_for_any_i[j][l][m])
                            self.delta[i][j][l][m] = max(estimate, MIN_PROB)

    def alignment_marginals(self, src_sentence, trg_sentence):
        """
        Computes the probability of all possible word alignments (comprised of product of
        word-to-word translation probability and alignment probability), expressed as a marginal
        distribution over target words t.

        :return: Probability of t for all s in ``src_sentence``
        """
        alignment_t = defaultdict(lambda: 0.0)
        for j in range(1, len(trg_sentence)):
            t = trg_sentence[j]
            for i in range(len(src_sentence)):
                alignment_t[t] += self.tau[trg_sentence[j]][src_sentence[i]] * \
                                  self.delta[i][j][len(src_sentence) - 1][len(trg_sentence) - 1]
        return alignment_t

    def build_eta(self):
        """
        Builds the eta dictionary for storing length priors based on parallel corpus data
        """
        lm_counts = defaultdict(lambda: defaultdict(int))
        l_counts = defaultdict(int)

        for pair in self.pc:
            src_sentence, trg_sentence = pair[1], pair[0]
            l, m = len(src_sentence), len(trg_sentence)
            lm_counts[l][m] += 1
            l_counts[l] += 1

        for l in l_counts.keys():
            norm = l_counts[l]
            for m in lm_counts[l].keys():
                joint = lm_counts[l][m]
                p = joint / norm
                self.eta[l][m] = p

    def score(self, src_sentence, trg_sentence):
        """
        Compute probability of translating source sentence into target sentence, by marginalizing
        over all alignments.
        """
        l, m = len(trg_sentence), len(src_sentence)
        src_sentence, trg_sentence = [None] + src_sentence, ['UNUSED'] + trg_sentence

        eta = self.eta[l][m]
        if eta > 0.0:
            num_samples = 1000
            align_prob = self.sample_alignments(trg_sentence, src_sentence, num_samples)
        else:
            align_prob = self.max_alignment(trg_sentence, src_sentence)
            
        return eta * align_prob

    def sample_alignment(self, l, m):
        alignment = []
        for i in xrange(m):
            rand = random.random()
            sum_p = 0.0
            for j in xrange(l):
                p = self.delta[j][i][l][m]
                sum_p += p
                if rand < sum_p:
                    alignment.append(j)
                    break

        assert len(alignment) == m
        return alignment

    def max_alignment(self, machine, natural):
        l = len(machine)
        m = len(natural)

        prod = 1.0
        for k in xrange(1,m):
            pword = natural[k]

            best_match = 0.0
            for j in xrange(l):
                word = self.tau[pword][machine[j]]
                if word > best_match:
                    best_match = word
            prod *= best_match

        prob = prod / math.pow(l, m)
        return prob

    def sample_alignments(self, machine, natural, samples):
        l, m = len(machine), len(natural)

        ret = 0.0
        for _ in xrange(samples):
            sampled = self.sample_alignment(l, m)
            prod = 1.0
            for k, ak in enumerate(sampled):
                pword = natural[k]
                gword = machine[ak]

                prod *= self.tau[pword][gword]
            ret += prod
        ret = ret / samples
        return ret


class Model2Counts(Counts):
    """
    Auxiliary object to store counts of various parameters during training. Specifically includes
    counts for alignment.
    """
    def __init__(self):
        super(Model2Counts, self).__init__()
        self.alignment = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:
                                                                             defaultdict(lambda:
                                                                                         0.0))))
        self.alignment_for_any_i = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:
                                                                                       0.0)))

    def update_lexical_translation(self, count, s, t):
        self.t_given_s[t][s] += count
        self.any_t_given_s[s] += count

    def update_alignment(self, count, i, j, l, m):
        self.alignment[i][j][l][m] += count
        self.alignment_for_any_i[j][l][m] += count
