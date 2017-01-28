"""
map_planning_times.py

Map each command in dataset to its mean planning time and save to disk
"""
from models.single_rnn import RNNClassifier
import os
import pickle
import sys
# import csv

nl_format, ml_format = "../clean_data/%s.en", "../clean_data/%s.ml"

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

def map_machine_lang(pf, lvl):
    pf = pf.replace('(', ' ').replace(',','').replace(')','')
    if 'L{0}'.format(lvl) in ['L1', 'L2']:
        pf = pf.replace('Room','Region')
    if 'blockInRoomAgentInRoom' in pf:
        split = pf.split()
        i = split[0].index('AgentInRoom')
        brf = split[0][:i]
        arf = split[0][i:]
        arf = arf[:1].lower() + arf[1:]
        pf = ' '.join([arf, split[3], split[4], brf, split[1], split[2]])
    if 'blockInRegionAgentInRegion' in pf:
        split = pf.split()
        i = split[0].index('AgentInRegion')
        brf = split[0][:i]
        arf = split[0][i:]
        arf = arf[:1].lower() + arf[1:]
        pf = ' '.join([arf, split[3], split[4], brf, split[1], split[2]])
    return 'L{0} {1}'.format(lvl, pf)

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

def load_model():
    with open('single_rnn_1_28_17/vocab.pik', 'r') as f:
        pc_train, ml_commands = pickle.load(f)
    model = RNNClassifier(pc_train, ml_commands)
    model.saver.restore(model.session, 'single_rnn_1_28_17/rnn.ckpt')
    return model

if __name__ == "__main__":
    if not os.path.exists("../api/single_rnn_1_28_17/checkpoint"):
        print 'Error: Unable to find checkpoint for Single-RNN - please train the model!'
        sys.exit(0)
    m = load_model()
    print 'Model Loaded!'

    timer = {'L0 goNorth':[], 'L0 goSouth':[], 'L0 goEast':[], 'L0 goWest':[],}
    out_header = ['predicted RF',
                  'small AMDP planner time',
                  'small AMDP std dev',
                  'small No heuristic AMDP planner',
                  'small base std dev',
                  'large AMDP planner time',
                  'large AMDP std dev',
                  'large No heuristic AMDP planner',
                  'large base std dev']

    rfs = []
    pc = []
    for level in levels:
        nl_tokens = get_tokens(nl_format % level)
        pc.extend(nl_tokens)
    
    count, total = 0, 0
    for nl_command in pc:
        rf, _ = m.score(nl_command)
        if rf[0] != 'L0':
            count += 1
        total += 1
        rfs.append(rf)
    
    print 'NON-L0 COUNT:', count, 'TOTAL:', total

    # THIS IS SO FUCKING INSANE I'M GOING TO HAVE ANGRY WORDS WITH SOMEONE
    import csv  # WHY DOES THIS AFFECT ANYTHING? => UNLESS - IMPORTING CSV CHANGES STRING ENCODING?
    # UGHHHHHHHHHHH - FUCK

    with open('./planning_times.csv') as data:
        reader = csv.DictReader(data)
        for row in reader:
            key = map_machine_lang(row['proposition function'], row['level solved at'])
            timer[key] = [row['small AMDP planner time'],
                          row['small AMDP std dev'],
                          row['small No heuristic AMDP planner'],
                          row['small base std dev'],
                          row['large AMDP planner time'],
                          row['large AMDP std dev'],
                          row['large No heuristic AMDP planner'],
                          row['large base std dev']]

    with open('./new_data_planning_times.csv', 'wb') as out:
        writer = csv.writer(out)
        writer.writerow(out_header)
        for rf in rfs:
            output = [' '.join(rf)] + timer[' '.join(rf)]
            if rf[0] != 'L0':
                l0_prop = list(rf) # THIS ALSO - UGH
                l0_prop[0] = 'L0'
                l0_prop[1:] = rf_map[' '.join(l0_prop[1:])].split()
                output[3] = timer[' '.join(l0_prop)][2]
                output[4] = timer[' '.join(l0_prop)][3]
                output[7] = timer[' '.join(l0_prop)][6]
                output[8] = timer[' '.join(l0_prop)][7]
            writer.writerow(output)
