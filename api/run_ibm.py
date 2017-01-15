"""
run_ibm.py 

Core file for training and checkpointing the Dual IBM Model 2 - Also has  command line 
functionality for loading and running inference on a given natural language command.
"""
from models.ibm2 import IBM2
from random import shuffle
import os
import dill as pickle

CONSTRAIN = False

# CLEANED
nl_format, ml_format = "../clean_data/intense_clean_no_punct/%s.en", "../clean_data/intense_clean_no_punct/%s.ml"
commands_format = "../clean_data/intense_clean_no_punct/%s.commands" if CONSTRAIN else "../clean_data/%s.commands"

# RAW
# nl_format, ml_format = "../clean_data/%s.en", "../clean_data/%s.ml"
# commands_format = "../clean_data/test/%s.commands" if CONSTRAIN else "../clean_data/%s.commands"
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

def train_model():
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
    
    l0_pc, l1_pc, l2_pc = list(data["L0"][0][:]), list(data["L1"][0][:]), list(data["L2"][0][:])
    joint_dataset = l0_pc + l1_pc + l2_pc
    shuffle(joint_dataset)

    models = {"L0": IBM2(l0_pc, 15), "L1": IBM2(l1_pc, 15), "L2": IBM2(l2_pc, 15)}
    joint_ibm2 = IBM2(joint_dataset, 15)

    correct, total, lvl_correct = 0, 0, 0
    for lvl in data:
        pc_test = data[lvl][1]
        for i in range(len(pc_test) - 1):
            # Get test command
            example_en, example_ml = pc_test[i]

            # Pick Level
            level, level_max = "", 0.0
            for k in data:
                commands, curr_sum = data[k][2], 0.0
                for c in commands:
                    curr_sum += joint_ibm2.score(example_en, c)
                lvl_signal = curr_sum / len(commands)
                if lvl_signal >= level_max:
                    level, level_max = k, lvl_signal
            
            ml_commands = data[level][2]
            # Score Translations
            best_trans, best_score = None, 0.0
            for t in ml_commands:
                score = models[level].score(example_en, t)
                if score > best_score:
                    best_trans, best_score = t, score
            print best_trans, best_score
            
            # Update Counters
            if level == lvl:
                lvl_correct += 1
            if best_trans == example_ml:
                correct += 1
            total += 1
    
    print 'Level Selection Accuracy:', float(lvl_correct) / float(total)
    print 'Test Accuracy:', float(correct) / float(total)

    with open('ibm_ckpt/models.pik', 'w') as f:
        pickle.dump((models, joint_ibm2), f)
    
def load_model():
    with open('ibm_ckpt/models.pik', 'r') as f:
        models, joint = pickle.load(f)
    return models, joint

def score(models, joint, nl_command):
    ml_commands = [get_tokens(commands_format % lvl) for lvl in levels]
    
    # Pick Level
    level, level_max = "", 0.0
    for k in range(len(ml_commands)):
        commands, curr_sum = ml_commands[k], 0.0
        for c in commands:
            curr_sum += joint.score(nl_command, c)
        lvl_signal = curr_sum / len(commands)
        if lvl_signal >= level_max:
            level, level_max = k, lvl_signal
    
    # Score Reward Functions
    commands = ml_commands[level]
    best_trans, best_score = None, 0.0
    for t in commands:
        score = models["L"+str(level)].score(nl_command, t)
        if score > best_score:
            best_trans, best_score = t, score
    
    return level, best_trans
    

if __name__ == "__main__":
    if not os.path.exists("ibm_ckpt/models.pik"):
        train_model()
    m, j = load_model()
    print 'Model Loaded!'
    while True:
        nl_command = raw_input("Enter a Natural Language Command: ")
        lvl, rf = score(m, j, nl_command.split())
        print 'Predicted Level: %s, Predicted RF: %s' % (lvl, rf)
        print ""