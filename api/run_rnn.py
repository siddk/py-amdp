"""
run_rnn.py 

Core file for training and checkpointing the Dual RNN Model - Also has  command line 
functionality for loading and running inference on a given natural language command.
"""
from models.dual_rnn import RNNDual
from random import shuffle
import os
import pickle

CONSTRAIN = False

nl_format, ml_format = "../clean_data/test/%s.en", "../clean_data/test/%s.ml"
commands_format = "../clean_data/test/%s.commands" if CONSTRAIN else "../clean_data/%s.commands"
levels = ['L0', 'L1', 'L2']

def get_tokens(file_name):
    """
    Returns list of sentences, where each sentence is represented as a list of tokens.
    """
    out_list = []
    with open(file_name, 'r') as f:
        for line in f:
            out_list.append(line.split())
    return out_list

def train_model(step=20): 
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
    
    model = RNNDual(data['L0'][0], data['L1'][0], data['L2'][0], data['L0'][2], data['L1'][2], 
                    data['L2'][2])
    
    for c in range(step, min(map(lambda z: len(z[0]), data.values())), step):
        model.fit(c)
        correct, total, lvl_correct = 0, 0, 0
        for lvl in data:
            pc_test = data[lvl][1]
            for i in range(len(pc_test) - 1):
                # Get test command
                example_en, example_ml = pc_test[i]

                # Pick Level, Translation
                best_trans, score, level, level_score = model.score(example_en)
                if lvl == levels[level]:
                    lvl_correct += 1
                    if best_trans == example_ml:
                        correct += 1
                total += 1
                
        print 'Level Selection Accuracy:', float(lvl_correct) / float(total)
        print 'Test Accuracy:', float(correct) / float(total)

    model.saver.save(model.session, 'rnn_ckpt/rnn.ckpt')
    with open('rnn_ckpt/vocab.pik', 'w') as f:
        pickle.dump((model.l0_pc, model.l1_pc, model.l2_pc, model.l0_commands, model.l1_commands,
                     model.l2_commands), f)

def load_model():
    with open('rnn_ckpt/vocab.pik', 'r') as f:
        l0_pc, l1_pc, l2_pc, l0_commands, l1_commands, l2_commands = pickle.load(f)
    model = RNNDual(l0_pc, l1_pc, l2_pc, l0_commands, l1_commands, l2_commands)
    model.saver.restore(model.session, "rnn_ckpt/rnn.ckpt")
    return model

if __name__ == "__main__":
    if not os.path.exists("rnn_ckpt/checkpoint"):
        train_model()
    m = load_model()
    print 'Model Loaded!'
    while True:
        nl_command = raw_input("Enter a Natural Language Command: ")
        rf, _, lvl, _ = m.score(nl_command.split())
        print 'Predicted Level: %s, Predicted RF: %s' % (lvl, rf)
        print ""