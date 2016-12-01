"""
ibm2.py

Core implementation of IBM Model 2 Machine Translation code. First instantiates an IBM 2 Model
with the necessary Tau and Delta parameters, then instantiates an IBM 1 Model. Performs Burn-in
EM Inference via IBM 1, then performs IBM 2 EM to learn alignments.
"""


class IBM2():
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