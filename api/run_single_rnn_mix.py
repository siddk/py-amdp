"""
run_single_rnn_mix.py

Core file for training and checkpointing the Single RNN Model with data from one abstraction level
 used for training and another distinct level used for testing - Also has  command line 
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
nl_format, ml_format = "../clean_data/%s.en", "../clean_data/%s.ml"
#nl_format, ml_format = "../clean_data/intense_clean_no_punct/%s.en", "../clean_data/intense_clean_no_punct/%s.ml"
commands_format = "../clean_data/intense_clean_no_punct/%s.commands" if CONSTRAIN else "../clean_data/%s.commands"

# RAW
# nl_format, ml_format = "../clean_data/%s.en", "../clean_data/%s.ml"

levels = ['L0', 'L1', 'L2']

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

rf_map = {"agentInRoom agent0 room0": "agentInRegion agent0 room0",
          "agentInRoom agent0 room1": "agentInRegion agent0 room1",
          "blockInRoom block0 room1": "blockInRegion block0 room1",
          "blockInRoom block0 room2": "blockInRegion block0 room2",
          "agentInRoom agent0 room1 blockInRoom block0 room2": "agentInRegion agent0 room1 blockInRegion block0 room2",
          "agentInRoom agent0 room2 blockInRoom block0 room1": "agentInRegion agent0 room2 blockInRegion block0 room1",
          "agentInRegion agent0 room0": "agentInRoom agent0 room0",
          "agentInRegion agent0 room1": "agentInRoom agent0 room1",
          "blockInRegion block0 room1": "blockInRoom block0 room1",
          "blockInRegion block0 room2": "blockInRoom block0 room2",
          "agentInRegion agent0 room1 blockInRegion block0 room2": "agentInRoom agent0 room1 blockInRoom block0 room2",
          "agentInRegion agent0 room2 blockInRegion block0 room1": "agentInRoom agent0 room2 blockInRoom block0 room1"}

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

def train_model(level, test_level):
    # Load Data
    if level != test_level:
        pc = []
        if level != 'L_ALL':
            while levels[0] != level:
                shuffle(levels)
        for lvl in levels:
            nl_tokens, ml_tokens = get_tokens(nl_format % lvl), get_tokens(ml_format % lvl)
            if lvl != level and not (level in ['L1', 'L2'] and lvl in ['L1', 'L2']) and level != 'L_ALL':
                ml_tokens = [rf_map[" ".join(x)].split(" ") for x in ml_tokens]
            if lvl != 'L0' and level == 'L_ALL':
                ml_tokens = [rf_map[" ".join(x)].split(" ") for x in ml_tokens]
            if lvl == test_level and level == 'L_ALL':
                nl_tokens = nl_tokens[:int(0.9 * len(nl_tokens))]
                ml_tokens = ml_tokens[:int(0.9 * len(ml_tokens))]
            pc.extend(zip(*(nl_tokens, ml_tokens)))

            if lvl == level and level != 'L_ALL':
                train_len = len(nl_tokens)
                train_commands = get_tokens(commands_format % lvl)
            if lvl == 'L0' and level == 'L_ALL':
                train_commands = get_tokens(commands_format % lvl)
        if level == 'L_ALL':
            shuffle(pc)
            shuffle(pc)
            shuffle(pc)
        pc_train = pc

        test_nl_tokens, test_ml_tokens = get_tokens(nl_format % test_level), get_tokens(ml_format % test_level)
        if test_level != level and not (level in ['L1', 'L2'] and test_level in ['L1', 'L2']) and level != 'L_ALL':
            test_ml_tokens = [rf_map[" ".join(x)].split(" ") for x in test_ml_tokens]
        if test_level != 'L0' and level == 'L_ALL':
            test_ml_tokens = [rf_map[" ".join(x)].split(" ") for x in test_ml_tokens]
        test_nl_tokens = test_nl_tokens[int(0.9 * len(test_nl_tokens)):]
        test_ml_tokens = test_ml_tokens[int(0.9 * len(test_ml_tokens)):]
        pc_test = zip(*(test_nl_tokens, test_ml_tokens))
        shuffle(pc_test)
        shuffle(pc_test)
        shuffle(pc_test)
    else:
        nl_tokens, ml_tokens = get_tokens(nl_format % level), get_tokens(ml_format % level)
        train_commands = get_tokens(commands_format % level)
        pc = zip(*(nl_tokens, ml_tokens))
        shuffle(pc)
        shuffle(pc)
        shuffle(pc)
        pc_train, pc_test = pc[:int(0.9 * len(pc))], pc[int(0.9 * len(pc)):]

    # Initialize Confusion Matrix
    if CONFUSION:
        confusion_matrix = {}
        for i in train_commands:
            confusion_matrix[convert(i)] = {}
            for j in train_commands:
                confusion_matrix[convert(i)][convert(j)] = 0

    model = RNNClassifier(pc_train, train_commands, verbose=0)
    if level != test_level and level != 'L_ALL':
        model.train_x = model.train_x[:train_len]
        model.train_y = model.train_y[:train_len]

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
    
    # model.saver.save(model.session, '%s_%s_single_rnn_mix_ckpt/rnn.ckpt' % (level, test_level))
    # with open('%s_%s_single_rnn_mix_ckpt/vocab.pik' % (level, test_level), 'w') as f:
    #    pickle.dump(pc_train, f)
    
    if CONFUSION:
        avg_df = pandas.DataFrame(confusion_matrix)
        for i in avg_df.index:
            avg_df.loc[i] /= 0.01 * sum(avg_df.loc[i])
        avg_df.to_csv('%s_%s_single_rnn_mix_confusion.csv' % (level, test_level), encoding='utf-8')

        
def load_model(level, test_level):
    with open('%s_%s_single_rnn_mix_ckpt/vocab.pik' % (level, test_level), 'r') as f:
        pc_train = pickle.load(f)
    ml_commands = get_tokens(commands_format % level)
    model = RNNClassifier(pc_train, ml_commands)
    model.saver.restore(model.session, '%s_%s_single_rnn_mix_ckpt/rnn.ckpt' % (level, test_level))
    return model


if __name__ == "__main__":
    # Read Command Line Arguments
    args = sys.argv
    lvl = args[1]
    test_lvl = args[2]

    if not os.path.exists("%s_%s_single_rnn_mix_ckpt/checkpoint" % (lvl, test_lvl)):
        train_model(lvl, test_lvl)
        sys.exit(0)
    m = load_model(lvl, test_lvl)
    print 'Model Loaded!'
    while True:
        nl_command = raw_input("Enter a Natural Language Command: ")
        rf, _ = m.score(nl_command.split())
        print 'Predicted RF: %s' % rf
        print ""
