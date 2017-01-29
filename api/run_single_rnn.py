"""
run_single_rnn.py

Core file for training and checkpointing the Single RNN Model - Also has  command line 
functionality for loading and running inference on a given natural language command.
"""
from models.single_rnn import RNNClassifier
from random import shuffle
import os
import pandas
import pickle
import sys

CONSTRAIN = False
CONFUSION = True

# CLEANED
nl_format, ml_format = "../clean_data/intense_clean_no_punct/%s.en", "../clean_data/intense_clean_no_punct/%s.ml"
commands_format = "../clean_data/intense_clean_no_punct/%s.commands" if CONSTRAIN else "../clean_data/%s.commands"

# RAW
# nl_format, ml_format = "../clean_data/%s.en", "../clean_data/%s.ml"

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

def get_tokens(file_name):
    """
    Returns list of sentences, where each sentence is represented as a list of tokens.
    """
    out_list = []
    with open(file_name, 'r') as f:
        for line in f:
            out_list.append(line.split())
    return out_list

def convert(command):
    return " ".join([prefix[x] for x in command])

def train_model(level):
    # Load Data
    nl_tokens, ml_tokens = get_tokens(nl_format % level), get_tokens(ml_format % level)
    ml_commands = get_tokens(commands_format % level)
    pc = zip(*(nl_tokens, ml_tokens))
    shuffle(pc)
    shuffle(pc)
    shuffle(pc)
    pc_train, pc_test = pc[:int(0.9 * len(pc))], pc[int(0.9 * len(pc)):]

    # Initialize Confusion Matrix
    if CONFUSION:
        confusion_matrix = {}
        for i in ml_commands:
            confusion_matrix[convert(i)] = {}
            for j in ml_commands:
                confusion_matrix[convert(i)][convert(j)] = 0

    model = RNNClassifier(pc_train, ml_commands)
    for idx in range(5):
        model.fit(len(pc_train))
        correct, total = 0, 0
        for i in range(len(pc_test) - 1):
            # Get test command
            example_en, example_ml = pc_test[i]

            # Pick Translation
            best_trans, score = model.score(example_en)
            if best_trans == example_ml:
                correct += 1
            total += 1
            
            if CONFUSION and idx == 4:
                confusion_matrix[convert(example_ml)][convert(best_trans)] += 1

        print 'Test Accuracy:', float(correct) / float(total)
    
    model.saver.save(model.session, '%s_single_rnn_ckpt/rnn.ckpt' % level)
    with open('%s_single_rnn_ckpt/vocab.pik' % level, 'w') as f:
        pickle.dump(pc_train, f)
    
    if CONFUSION:
        avg_df = pandas.DataFrame(confusion_matrix)
        for i in avg_df.index:
            avg_df.loc[i] /= 0.01 * sum(avg_df.loc[i])
        avg_df.to_csv('%s_single_rnn_confusion.csv' % level, encoding='utf-8')

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

    if not os.path.exists("%s_single_rnn_ckpt/checkpoint" % lvl):
        train_model(lvl)
    m = load_model(lvl)
    print 'Model Loaded!'
    while True:
        nl_command = raw_input("Enter a Natural Language Command: ")
        rf, _ = m.score(nl_command.split())
        print 'Predicted RF: %s' % rf
        print ""