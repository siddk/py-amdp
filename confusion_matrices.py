"""
confusion_matrices.py

Generates confusion matrices for the specified configuration, to aid in error analysis.

Usage: python confusion_matrices [all|single|dual] [Num-Trials] [Level]
"""
import tensorflow as tf
from models.ibm2 import IBM2
from models.nn_classifier import NNClassifier
from models.rnn_classifier import RNNClassifier
from preprocessor.reader import *
from random import shuffle
import pandas
import pickle
import sys

nl_format, ml_format, commands_format = "clean_data/test/%s.en", "clean_data/test/%s.ml", "clean_data/test/%s.commands"

prefix = {"agentInRegion": "aReg",
          "agentInRoom": "aRoom",
          "blockInRegion": "bReg",
          "blockInRoom": "bRoom",
          "agent0": "",
          "block0": "",
          "room0": "r0",
          "room1": "r1",
          "room2": "r2",
          "door0": "d0",
          "door1": "d1"}


def convert(command):
    return " ".join([prefix[x] for x in command])


def get_dataframe(level, model='ibm2'):
    """
    Given the specific level to train on, take an arbitrary 90-10 split of the level data, then
    build the confusion matrix (represented as a dataframe).

    :param level: Level to train on.
    :return DataFrame representing the Confusion Matrix.
    """
    tf.reset_default_graph()
    # Load Data
    nl_tokens, ml_tokens = get_tokens(nl_format % level), get_tokens(ml_format % level)
    ml_commands = get_tokens(commands_format % level)
    pc = zip(*(nl_tokens, ml_tokens))
    shuffle(pc)
    shuffle(pc)
    shuffle(pc)
    pc_train, pc_test = pc[:int(0.9 * len(pc))], pc[int(0.9 * len(pc)):]

    # Initialize Confusion Matrix
    confusion_matrix = {}
    for i in ml_commands:
        confusion_matrix[convert(i)] = {}
        for j in ml_commands:
            confusion_matrix[convert(i)][convert(j)] = 0

    # Train Model
    if model == 'rnn':
        print 'Training RNN Classifier'
        m = RNNClassifier(list(pc_train), ml_commands)
    elif model == 'nn':
        print 'Training NN Classifier'
        m = NNClassifier(list(pc_train), ml_commands)
    else:
        print 'Training IBM Model!'
        m = IBM2(pc_train, 15)

    # Evaluate on Test Data
    correct, total = 0, 0
    for i in range(len(pc_test) - 1):
        # Get test command
        example_en, example_ml = pc_test[i]

        # Score Translations
        if model == 'ibm2':
            best_trans, best_score = None, 0.0
            for t in ml_commands:
                score = m.score(example_en, t)
                if score > best_score:
                    best_trans, best_score = t, score

        elif model in ['rnn', 'nn']:
            best_trans, best_score = m.score(example_en)

        # Update Counters
        total += 1
        if best_trans == example_ml:
            correct += 1

        # Update Confusion Matrix
        confusion_matrix[convert(example_ml)][convert(best_trans)] += 1

    # Return Matrix, Accuracy
    return pandas.DataFrame(confusion_matrix), float(correct) / float(total)


if __name__ == "__main__":
    # Read Command Line Arguments
    args = sys.argv
    run_type, model, num_trials = args[1], args[2], args[3]

    if run_type == 'dual':
        pass

    else:
        if run_type == 'all':
            lvl = 'L_ALL'

        if run_type == 'single':
            # Get Level to Train On
            lvl = args[4]

        # Generate DataFrames for Each Trial
        df, acc = [], []
        for trial in range(int(num_trials)):
            d, a = get_dataframe(lvl, model)
            df.append(d)
            acc.append(a)
            print 'Trial %s Accuracy:' % str(trial + 1), a

        # Pickle Entire Lists
        with open('%s_confusion.pik' % lvl, 'w') as f:
            pickle.dump((df, acc), f)

        # Write Confusion Matrix, Average Accuracy to File
        avg_df = sum(df)
        for i in avg_df.index:
            avg_df.loc[i] /= 0.01 * sum(avg_df.loc[i])
        avg_df.to_csv('%s_confusion.csv' % lvl, encoding='utf-8')

        print 'Average Accuracy on Level %s over %s Runs:' % (lvl, num_trials), sum(acc) / len(acc)

