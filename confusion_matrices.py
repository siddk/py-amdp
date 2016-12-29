"""
confusion_matrices.py

Generates confusion matrices for the specified configuration, to aid in error analysis.

Usage: python confusion_matrices [all|single|dual] [Num-Trials] [Level]
"""
from models.ibm2 import IBM2
from preprocessor.reader import *
from random import shuffle
import pandas
import pickle
import sys

nl_format, ml_format, commands_format = "clean_data/test/%s.en", "clean_data/test/%s.ml", "clean_data/test/%s.commands"


def get_dataframe(level):
    """
    Given the specific level to train on, take an arbitrary 90-10 split of the level data, then
    build the confusion matrix (represented as a dataframe).

    :param level: Level to train on.
    :return DataFrame representing the Confusion Matrix.
    """
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
        confusion_matrix[" ".join(i)] = {}
        for j in ml_commands:
            confusion_matrix[" ".join(i)][" ".join(j)] = 0

    # Train IBM Model
    print 'Training IBM Model!'
    ibm2 = IBM2(pc_train, 15)

    # Evaluate on Test Data
    correct, total = 0, 0
    for i in range(len(pc_test) - 1):
        # Get test command
        example_en, example_ml = pc_test[i]

        # Score Translations
        best_trans, best_score = None, -1
        for t in ml_commands:
            score = ibm2.score(example_en, t)
            if score > best_score:
                best_trans, best_score = t, score

        # Update Counters
        total += 1
        if best_trans == example_ml:
            correct += 1

        # Update Confusion Matrix
        print example_ml
        print best_trans
        confusion_matrix[" ".join(example_ml)][" ".join(best_trans)] += 1

    # Return Matrix, Accuracy
    return pandas.DataFrame(confusion_matrix), float(correct) / float(total)


if __name__ == "__main__":
    # Read Command Line Arguments
    args = sys.argv
    run_type, num_trials = args[1], args[2]

    if run_type == 'all':
        pass

    elif run_type == 'single':
        # Get Level to Train On
        lvl = args[3]

        # Generate DataFrames for Each Trial
        df, acc = [], []
        for trial in range(int(num_trials)):
            d, a = get_dataframe(lvl)
            df.append(d)
            acc.append(a)
            print 'Confusion Matrix:', d
            print 'Accuracy:', a

        # Pickle Entire Lists
        with open('%s_confusion.pik', 'w') as f:
            pickle.dump((df, acc), f)

        # Write Average Confusion Matrix, Average Accuracy to File
        avg_df = sum(df) / float(len(df))
        avg_df.to_csv('%s_confusion.csv', encoding='utf-8')

        print 'Average Accuracy on Level %s over %s Runs:' % (lvl, num_trials), sum(acc) / len(acc)

    elif run_type == 'dual':
        pass