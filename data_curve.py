"""
data_curve.py

Given the training configuration (i.e. L0 L1), randomly samples sets of 20s to compute LOO
accuracy, and generates a plot of accuracy as more data is added.
"""
from models.ibm2 import IBM2
from preprocessor.reader import *
import matplotlib.pyplot as plt
import numpy as np
import sys

nl_format, ml_format, commands_format = "data/%s.en", "data/%s.ml", "data/%s.commands"


def loo_data_curve(nl_level, ml_level, step=20):
    """
    Performs LOO Cross-Validation, generates accuracy for the given Natural Language - Machine
    Language Pair.

    :param nl_level: Natural Language Level => One of 'L0', 'L1', or 'L2'
    :param ml_level: Machine Language Level => One of 'L0', 'L1', or 'L2'
    """
    nl_tokens, ml_tokens = get_tokens(nl_format % nl_level), get_tokens(ml_format % ml_level)
    ml_commands = get_tokens(commands_format % ml_level)
    pc = np.array(zip(*(nl_tokens, ml_tokens)))

    chunk_sizes, loo_accuracies = [], []
    for chunk_size in range(step, len(pc), step):
        idx = np.random.choice(len(pc), chunk_size)
        parallel_corpus = list(pc[idx])
        correct, total = 0, 0
        for i in range(len(parallel_corpus) - 1):
            # Split into LOO dataset and example
            dataset = parallel_corpus[:i] + parallel_corpus[i + 1:]
            example_en, example_ml = parallel_corpus[i]

            # Train IBM Model 2 on Dataset
            print 'Training IBM Model 2 on LOO Index:', i
            ibm2 = IBM2(dataset, 15)

            # Score Translations
            best_trans, best_score = None, 0.0
            for t in ml_commands:
                score = ibm2.score(example_en, t)
                if score > best_score:
                    best_trans, best_score = t, score
            print best_trans, best_score

            # Update Counters
            total += 1
            if best_trans == example_ml:
                correct += 1

            print 'Chunk %s LOO Index %s Current Accuracy:' % (str(chunk_size), str(i)), \
                float(correct) / float(total)

        chunk_sizes.append(chunk_size)
        loo_accuracies.append(float(correct) / float(total))

    # Print Chunk Sizes, Accuracies
    print 'Chunk Sizes:', chunk_sizes
    print 'Accuracies:', loo_accuracies

    # Plot Data Curve
    plt.plot(chunk_sizes, loo_accuracies)
    plt.title('%s - %s Data Curve' % (nl_level, ml_level))
    plt.xlabel('Number of Examples')
    plt.ylabel('LOO Accuracy')
    plt.show()


if __name__ == "__main__":
    # Read Level-Combination from Command Line
    args = sys.argv
    en_lvl, ml_lvl = args[1], args[2]

    # Run LOO Cross-Validation => Get Accuracy
    loo_data_curve(en_lvl, ml_lvl)