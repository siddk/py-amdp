"""
run_ibm_mix.py 
"""
from models.ibm2 import IBM2
import random
from random import shuffle
import pickle

levels = ['L0', 'L1', 'L2']

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


def train_model(level, test_level, data):
    with open(data, 'r') as f:
        pc, commands = pickle.load(f)

    if level != test_level:
        pc_train, pc_test = pc[level], pc[test_level]
    else:
        pc_train, pc_test = pc[level][:int(0.9 * len(pc[level]))], pc[level][int(0.9 * len(pc[level])):]
    shuffle(pc_train)
    shuffle(pc_train)
    shuffle(pc_train)

    shuffle(pc_test)
    shuffle(pc_test)
    shuffle(pc_test)
    
    all_commands = commands[level]
        
    joint_ibm2 = IBM2(pc_train, 15)
    correct, total = 0, 0
    for (example_en, example_ml) in pc_test:
        if level != test_level and not (level in ['L1', 'L2'] and test_level in ['L1', 'L2']):
            example_ml = rf_map[" ".join(example_ml)].split(" ")

        # Score Translations
        best_trans, best_score = None, 0.0
        for t in all_commands:
            score = joint_ibm2.score(example_en, t)
            if score >= best_score:
                best_trans, best_score = t, score
                    
        if best_trans == example_ml:
            correct += 1
        total += 1
    
    print 'Test Accuracy:', float(correct) / float(total)

    
if __name__ == "__main__":
    import sys
    args = sys.argv
    lvl = args[1]
    test_lvl = args[2]
    clean = False
    
    data_path = "%s_data.pik" % ("clean" if clean else "raw")

    train_model(lvl, test_lvl, data_path)
