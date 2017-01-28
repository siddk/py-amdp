"""
single_rnn_cross_val.py 
"""
from models.single_rnn import RNNClassifier
import random
import pickle
import tensorflow as tf

NUM_TRIALS = 3
FOLDS = 10

levels = ['L0', 'L1', 'L2']

def run_cross_val(data, out, N_10_EPOCHS=5):
    with open(data, 'r') as f:
        pc, commands = pickle.load(f)
    
    pc['L0'] = map(lambda x: (x[0], ['L0'] + x[1]), pc['L0'])
    pc['L1'] = map(lambda x: (x[0], ['L1'] + x[1]), pc['L1'])
    pc['L2'] = map(lambda x: (x[0], ['L2'] + x[1]), pc['L2'])

    commands['L0'] = map(lambda x: ['L0'] + x, commands['L0'])
    commands['L1'] = map(lambda x: ['L1'] + x, commands['L1'])
    commands['L2'] = map(lambda x: ['L2'] + x, commands['L2'])
    all_commands = commands['L0'] + commands['L1'] + commands['L2']
    
    l0_len, l1_len, l2_len = len(pc['L0']), len(pc['L1']), len(pc['L2'])
    l0_range = range(0, l0_len, l0_len / FOLDS)
    l1_range = range(0, l1_len, l1_len / FOLDS)
    l2_range = range(0, l2_len, l2_len / FOLDS)
    # l_range = range(0, min(l0_len, l1_len, l2_len), min(l0_len, l1_len, l2_len) / FOLDS)
    assert(len(l0_range) == len(l1_range) == len(l2_range) == 11)
    
    lvl_selection, reward_selection = [], []
    for i in range(FOLDS):
        tf.reset_default_graph()
        random.seed(21)
        val = {'L0': pc['L0'][l0_range[i]:l0_range[i + 1]],
               'L1': pc['L1'][l1_range[i]:l1_range[i + 1]],
               'L2': pc['L2'][l2_range[i]:l2_range[i + 1]]}

        l0_train = pc['L0'][:l0_range[i]] + pc['L0'][l0_range[i + 1]:]
        l1_train = pc['L1'][:l1_range[i]] + pc['L1'][l1_range[i + 1]:]
        l2_train = pc['L2'][:l2_range[i]] + pc['L2'][l2_range[i + 1]:]

        joint_dataset = l0_train + l1_train + l2_train
        random.shuffle(joint_dataset)

        model = RNNClassifier(joint_dataset, all_commands)
        for idx in range(N_10_EPOCHS):
            model.fit(len(joint_dataset))
        
        correct, lvl_correct, total = 0, 0, 0
        for lvl in levels:
            for (example_en, example_ml) in val[lvl]:
                # Pick Level, Translation
                best_trans, score = model.score(example_en)
                if best_trans == example_ml:
                    correct += 1
                if best_trans[0] == example_ml[0]:
                    lvl_correct += 1
                total += 1
        
        print 'Level Selection:', float(lvl_correct) / float(total)
        print 'Reward Selection:', float(correct) / float(total)

        lvl_selection.append(float(lvl_correct) / float(total))
        reward_selection.append(float(correct) / float(total))
    
    with open(out_path, 'w') as f:
        f.write("Fold Level Selection Accuracies: %s\nFold Reward Function Accuracies: %s\n" % (str(lvl_selection), str(reward_selection)))
        f.write("Average Level Selection Accuracy: %s\n" % str(sum(lvl_selection) / len(lvl_selection)))
        f.write("Average Reward Function Accuracy: %s\n" % str(sum(reward_selection) / len(reward_selection)))

if __name__ == "__main__":
    import sys
    args = sys.argv
    if args[1] == 'True':
        clean = True
    elif args[1] == 'False':
        clean = False
    
    data_path = "%s_data.pik" % ("clean" if clean else "raw")

    for trial in range(NUM_TRIALS):
        out_path = "%s_single_rnn_results_%s.txt" % ("clean" if clean else "raw", str(trial))
        run_cross_val(data_path, out_path)