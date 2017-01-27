"""
run_single_rnn_separate.py

Same as run_single_rnn, but treats shared reward functions as separate, across 
different levels.
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
nl_format, ml_format = "../clean_data/%s.en", "../clean_data/%s.ml"
commands_format = "../clean_data/intense_clean_no_punct/%s.commands" if CONSTRAIN else "../clean_data/%s.commands"

# RAW
# nl_format, ml_format = "../clean_data/%s.en", "../clean_data/%s.ml"

levels = ['L0', 'L1', 'L2']

prefix = {"agentInRegion": "aReg",
          "agentInRoom": "aRoom",
          "blockInRegion": "bReg",
          "blockInRoom": "bRoom",
          "goNorth": "north",
          "goSouth": "south",
          "goEast": "east",
          "goWest": "west",
          "agent0": "",
          "block0": "",
          "room0": "r0",
          "room1": "r1",
          "room2": "r2",
          "door0": "d0",
          "door1": "d1",
          "L0" : "L0",
          "L1" : "L1",
          "L2" : "L2"}

def get_tokens(file_name, level=None):
    """
    Returns list of sentences, where each sentence is represented as a list of tokens.
    """
    out_list = []
    with open(file_name, 'r') as f:
        for line in f:
            if level is not None:
                out_list.append([level] + line.split())
            else:
                out_list.append(line.split())
    return out_list

def convert(command):
    return " ".join([prefix[x] for x in command])

def train_model():
    # Load Data
    pc, ml_commands = [], []
    for level in levels:
        nl_tokens, ml_tokens = get_tokens(nl_format % level), get_tokens(ml_format % level, level)
        lvl_commands = get_tokens(commands_format % level, level)
        pc.extend(zip(*(nl_tokens, ml_tokens)))
        ml_commands.extend(lvl_commands)
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
        correct, total, lvl_correct = 0, 0, 0
        for i in range(len(pc_test) - 1):
            # Get test command
            example_en, example_ml = pc_test[i]

            # Pick Translation
            best_trans, score = model.score(example_en)
            if best_trans == example_ml:
                correct += 1
            if best_trans[0] == example_ml[0]:
                lvl_correct += 1
            total += 1
            
            if CONFUSION and idx == 4:
                confusion_matrix[convert(example_ml)][convert(best_trans)] += 1

        print 'Test Accuracy:', float(correct) / float(total)
        print 'Level Selection Accuracy:', float(lvl_correct) / float(total)
    
    model.saver.save(model.session, 'l_all_rnn_ckpt/rnn.ckpt')
    with open('l_all_rnn_ckpt/vocab.pik', 'w') as f:
        pickle.dump((pc_train, ml_commands), f)
    
    if CONFUSION:
        avg_df = pandas.DataFrame(confusion_matrix)
        for i in avg_df.index:
            avg_df.loc[i] /= 0.01 * sum(avg_df.loc[i])
        avg_df.to_csv('l_all_rnn_confusion.csv', encoding='utf-8')

def load_model():
    with open('l_all_rnn_ckpt/vocab.pik', 'r') as f:
        pc_train, ml_commands = pickle.load(f)
    model = RNNClassifier(pc_train, ml_commands)
    model.saver.restore(model.session, 'l_all_rnn_ckpt/rnn.ckpt')
    return model

if __name__ == "__main__":
    if not os.path.exists("l_all_rnn_ckpt/checkpoint"):
        train_model()
    m = load_model()
    print 'Model Loaded!'
    while True:
        nl_command = raw_input("Enter a Natural Language Command: ")
        rf, _ = m.score(nl_command.split())
        print 'Predicted RF: %s' % rf
        print ""
