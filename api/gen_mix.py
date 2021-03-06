"""
gen_mix.py 

Script that generates pickled versions of the mix data.
"""
from random import shuffle
import pickle

CLEAN = False
CONSTRAIN = False

L0_LIM = 869
L0_COMMANDS_LIM = 15

# CLEANED
nl_format, ml_format = "../clean_data/%s.en", "../clean_data/%s.ml"
commands_format = "../clean_data/intense_clean_no_punct/%s.commands" if CONSTRAIN else "../clean_data/%s.commands"
levels = ['L0', 'L1', 'L2']

# RAW
# nl_format, ml_format = "../clean_data/%s.en", "../clean_data/%s.ml"

FILE_NAME = "%s_data.pik" % ("clean" if CLEAN else "raw")

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

def split_data():
    # Load Data
    pc_level, ml_commands = {l: [] for l in levels}, {l: [] for l in levels}
    for level in levels:
        nl_tokens, ml_tokens = get_tokens(nl_format % level), get_tokens(ml_format % level)
        lvl_commands = get_tokens(commands_format % level)
        if level == 'L0':
            nl_tokens, ml_tokens = nl_tokens[:L0_LIM], ml_tokens[:L0_LIM]
            lvl_commands = lvl_commands[:L0_COMMANDS_LIM]
        pc = zip(*(nl_tokens, ml_tokens))
        shuffle(pc)
        shuffle(pc)
        shuffle(pc)
        pc_level[level].extend(pc)
        ml_commands[level].extend(lvl_commands)
    
    with open(FILE_NAME, 'w') as f:
        pickle.dump((pc_level, ml_commands), f)

if __name__ == "__main__":
    split_data()
