"""
neural_dual.py

Core script for training/evaluating the core dual Model that has two separate sets of IBM 2 Language
Models - one for picking the correct level of abstraction, and another for finding the conditional
probability of each reward function, given the picked level.
"""
from collections import defaultdict
from models.dual_nn import NNDual
from preprocessor.reader import *
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas

CONSTRAIN = False

nl_format, ml_format = "clean_data/test/%s.en", "clean_data/test/%s.ml"
commands_format = "clean_data/test/%s.commands" if CONSTRAIN else "clean_data/%s.commands"
levels = ['L0', 'L1', 'L2']

def data_curve(save_id, step=20, save_fig=False):
    """
    Plots accuracy over number of examples, across all-levels.
    """
    data = {}
    for lvl in levels:
        nl_tokens, ml_tokens = get_tokens(nl_format % lvl), get_tokens(ml_format % lvl)
        ml_commands = get_tokens(commands_format % lvl)
        pc = zip(*(nl_tokens, ml_tokens))
        shuffle(pc)
        shuffle(pc)
        shuffle(pc)
        pc_train, pc_test = pc[:int(0.9 * len(pc))], pc[int(0.9 * len(pc)):]
        data[lvl] = (pc_train, pc_test, ml_commands)

    chunk_sizes, accuracies, level_accuracies, level_confusion = [], [], [], {}
    for lvl in levels:
        level_confusion[lvl] = {}
        for lvl2 in levels:
            level_confusion[lvl][lvl2] = 0

    model = NNDual(data['L0'][0], data['L1'][0], data['L2'][0], data['L0'][2], data['L1'][2], data['L2'][2])
            
    for chunk_size in range(step, min(map(lambda z: len(z[0]), data.values())), step):
        print 'Training Neural Dual Model on Chunk:', chunk_size
        model.fit(chunk_size)

        correct, total, lvl_correct = 0, 0, 0
        for lvl in data:
            pc_test = data[lvl][1]
            for i in range(len(pc_test) - 1):
                # Get test command
                example_en, example_ml = pc_test[i]

                # Pick Level, Translation
                best_trans, score, level, level_score = model.score(example_en)
                level_confusion[levels[level]][lvl] += 1
                if lvl == levels[level]:
                    lvl_correct += 1
                    if best_trans == example_ml:
                        correct += 1
                    print best_trans, score
                total += 1
                
        print 'Chunk %s Level Selection Accuracy:' % str(chunk_size), float(lvl_correct) / float(total)
        print 'Chunk %s Test Accuracy:' % str(chunk_size), float(correct) / float(total)
        chunk_sizes.append(chunk_size)
        accuracies.append(float(correct) / float(total))
        level_accuracies.append(float(lvl_correct) / float(total))

    # Print Chunk Sizes, Accuracies
    print 'Chunk Sizes:', chunk_sizes
    print 'Accuracies:', accuracies
    print 'Level Selection Accuracies:', level_accuracies

    if save_fig:
        # Plot Data Curve
        plt.plot(chunk_sizes, accuracies)
        plt.title('Dual Model Data Curve')
        plt.xlabel('Number of Examples')
        plt.ylabel('Reward Function Accuracy')
        #plt.show()
        plt.savefig('./neural_dual_data_{0}.png'.format(save_id))
        plt.clf()

        # Plot Level Selection Accuracy Curve
        plt.plot(chunk_sizes, level_accuracies)
        plt.title('Dual Model AMDP Level Selection Data Curve')
        plt.xlabel('Number of Examples')
        plt.ylabel('Level Selection Accuracy')
        plt.savefig('./neural_dual_level_{0}.png'.format(save_id))

    print 'lc', level_confusion
    return chunk_sizes, accuracies, level_accuracies, pandas.DataFrame(level_confusion)


if __name__ == "__main__":
    # Read Command-Line Arguments (regular, or data-curve evaluation)
    args = sys.argv
    _, num_trials = args[1], args[2]

    # Run LOO Cross-Validation => Get Accuracy
    error_bar_data, error_bar_level, data_frames = defaultdict(list), defaultdict(list), []
    for i in xrange(int(num_trials)):
        x, y, z, df = data_curve(i + 1)
        data_frames.append(df)
        for j in range(len(x)):
            error_bar_data[x[j]] += [y[j]]
            error_bar_level[x[j]] += [z[j]]

    # Create and Save Data Error Bar
    tuples = sorted(list(error_bar_data.iteritems()), key=lambda x: x[0])
    plt.plot([x[0] for x in tuples], [np.mean(x[1]) for x in tuples], 'g--')
    plt.errorbar([x[0] for x in tuples], [np.mean(x[1]) for x in tuples],
                    yerr=1.96 * np.array([np.std(x[1]) for x in tuples]) * (1 / np.sqrt(int(args[2]))),
                    color='g')
    plt.title('Dual Model Test Accuracy vs. Number Training Examples')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Test Accuracy')
    plt.ylim([0, 1])
    plt.savefig('./neural_dual_data_error_bar_%s.png' % ('constrained' if CONSTRAIN else 'unconstrained'))
    plt.clf()

    # Create and Save Level Error Bar
    tuples = sorted(list(error_bar_level.iteritems()), key=lambda x: x[0])
    plt.plot([x[0] for x in tuples], [np.mean(x[1]) for x in tuples], 'g--')
    plt.errorbar([x[0] for x in tuples], [np.mean(x[1]) for x in tuples],
                    yerr=1.96 * np.array([np.std(x[1]) for x in tuples]) * (1 / np.sqrt(int(args[2]))),
                    color='g')
    plt.title('Dual Model Level Selection Accuracy vs. Number Training Examples')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Level Selection Accuracy')
    plt.ylim([0, 1])
    plt.savefig('./neural_dual_level_error_bar_%s.png' % ('constrained' if CONSTRAIN else 'unconstrained'))
    plt.clf()

    # Create and Save Level Confusion Matrix
    for x in data_frames:
        print x
    avg_df = sum(data_frames)
    print avg_df
    for lvl in avg_df.index:
        avg_df.loc[lvl] /= 0.01 * sum(avg_df.loc[lvl])
    avg_df.to_csv('neural_dual_confusion_%s.csv' % ('constrained' if CONSTRAIN else 'unconstrained'), encoding='utf-8')
