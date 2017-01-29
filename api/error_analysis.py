"""
error_analysis.py

Print out all data samples at a designated level where trained model and ground truth differ for convenient analysis
"""
from models.single_rnn import RNNClassifier
from random import shuffle
import pickle
import sys

CONSTRAIN = False

# CLEANED
nl_format, ml_format = "../clean_data/%s.en", "../clean_data/%s.ml"
commands_format = "../clean_data/intense_clean_no_punct/%s.commands" if CONSTRAIN else "../clean_data/%s.commands"

def get_tokens(file_name):
    """
    Returns list of sentences, where each sentence is represented as a list of tokens.
    """
    out_list = []
    with open(file_name, 'r') as f:
        for line in f:
            out_list.append(line.split())
    return out_list


def get_data(level):
    # Load Data
    nl_tokens, ml_tokens = get_tokens(nl_format % level), get_tokens(ml_format % level)
    pc = zip(*(nl_tokens, ml_tokens))
    shuffle(pc)
    shuffle(pc)
    shuffle(pc)
    return pc

def load_model():
    with open('single_rnn_1_28_17/vocab.pik', 'r') as f:
        pc_train, ml_commands = pickle.load(f)
    model = RNNClassifier(pc_train, ml_commands)
    model.saver.restore(model.session, 'single_rnn_1_28_17/rnn.ckpt')
    return model

if __name__ == "__main__":
    # Read Command Line Arguments
    args = sys.argv
    lvl = args[1]

    m = load_model()
    print 'Model Loaded!'
    pc = get_data(lvl)

    for i in range(len(pc) - 1):
        # Get test command
        example_en, example_ml = pc[i]

        # Pick Translation
        best_trans, _ = m.score(example_en)
        best_trans
        if best_trans[1:] != example_ml:
            print 'Command:{0} \n\t True:{1} \t\t\t\t Predicted:{2}'.format(' '.join(example_en), ' '.join(example_ml), ' '.join(best_trans))
