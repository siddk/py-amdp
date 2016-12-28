"""
data_curve.py

Given the training configuration (i.e. L0 L1), partitions data into a random 90 - 10 train/test
split, then randomly samples sets of 20s to compute accuracy, and generates a plot of accuracy as
more data is added.
"""
from collections import defaultdict
from models.ibm2 import IBM2
from preprocessor.reader import *
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import sys

nl_format, ml_format, commands_format = "clean_data/test/%s.en", "clean_data/test/%s.ml", "clean_data/test/%s.commands"


def loo_data_curve(nl_level, ml_level, save_id, step=20, save_fig=True):
    """
    Performs LOO Cross-Validation, generates accuracy for the given Natural Language - Machine
    Language Pair.

    :param nl_level: Natural Language Level => One of 'L0', 'L1', or 'L2'
    :param ml_level: Machine Language Level => One of 'L0', 'L1', or 'L2'
    """
    nl_tokens, ml_tokens = get_tokens(nl_format % nl_level), get_tokens(ml_format % ml_level)
    ml_commands = get_tokens(commands_format % ml_level)
    pc = zip(*(nl_tokens, ml_tokens))
    shuffle(pc)
    shuffle(pc)
    shuffle(pc)
    pc_train, pc_test = pc[:int(0.9 * len(pc))], pc[int(0.9 * len(pc)):]

    chunk_sizes, accuracies = [], []
    for chunk_size in range(step, len(pc_train), step):
        dataset = list(pc_train[:chunk_size])
        print 'Training IBM Model 2 on Chunk:', chunk_size
        ibm2 = IBM2(dataset, 15)

        correct, total = 0, 0
        for i in range(len(pc_test) - 1):
            # Get test command
            example_en, example_ml = pc_test[i]

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

        print 'Chunk %s Test Accuracy:' % str(chunk_size), float(correct) / float(total)
        chunk_sizes.append(chunk_size)
        accuracies.append(float(correct) / float(total))

    # Print Chunk Sizes, Accuracies
    print 'Chunk Sizes:', chunk_sizes
    print 'Accuracies:', accuracies

    if save_fig:
        # Plot Data Curve
        plt.plot(chunk_sizes, accuracies)
        plt.title('%s - %s Data Curve' % (nl_level, ml_level))
        plt.xlabel('Number of Examples')
        plt.ylabel('Accuracy')
        plt.xlim(0, len(pc_train) + step)
        #plt.show()
        plt.savefig('./{0}_{1}_{2}.png'.format(nl_level, ml_level, save_id))
        plt.clf()

    return chunk_sizes, accuracies


if __name__ == "__main__":
    # Read Level-Combination from Command Line
    args = sys.argv
    en_lvl, ml_lvl = args[1], args[2]

    # Run LOO Cross-Validation => Get Accuracy
    error_bar = defaultdict(list)
    for i in xrange(int(args[3])):
        x, y = loo_data_curve(en_lvl, ml_lvl, i + 1)
        for j in range(len(x)):
            error_bar[x[j]] += [y[j]]

    # Create and Save Error Bar
    tuples = sorted(list(error_bar.iteritems()), key=lambda x: x[0])
    plt.plot([x[0] for x in tuples], [np.mean(x[1]) for x in tuples], 'g--')
    plt.errorbar([x[0] for x in tuples], [np.mean(x[1]) for x in tuples],
                 yerr=1.96 * np.array([np.std(x[1]) for x in tuples]) * (1 / np.sqrt(int(args[3]))),
                 color='g')
    plt.title('%s - %s Test Accuracy vs. Number Training Examples' % (en_lvl, ml_lvl))
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Test Accuracy')
    plt.savefig('./{0}_{1}_error_bar.png'.format(en_lvl, ml_lvl))
