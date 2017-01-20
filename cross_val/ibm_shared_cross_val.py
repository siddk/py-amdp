"""
ibm_shared_cross_val.py 

Same as ibm_cross_val, except with shared output commands
"""
from models.ibm2 import IBM2
import random
import pickle

NUM_TRIALS = 1
FOLDS = 10

levels = ['L0', 'L1', 'L2']

def run_cross_val(data, out):
    with open(data, 'r') as f:
        pc, commands = pickle.load(f)
    
    all_commands = commands['L0'] + commands['L1']
    
    l0_len, l1_len, l2_len = len(pc['L0']), len(pc['L1']), len(pc['L2'])
    l0_range = range(0, l0_len, l0_len / FOLDS)
    l1_range = range(0, l1_len, l1_len / FOLDS)
    l2_range = range(0, l2_len, l2_len / FOLDS)
    assert(len(l0_range) == len(l1_range) == len(l2_range) == 11)
    
    reward_selection = []
    for i in range(FOLDS):
        random.seed(21)
        val = {'L0': pc['L0'][l0_range[i]:l0_range[i + 1]],
               'L1': pc['L1'][l1_range[i]:l1_range[i + 1]],
               'L2': pc['L2'][l2_range[i]:l2_range[i + 1]]}

        l0_train = pc['L0'][:l0_range[i]] + pc['L0'][l0_range[i + 1]:]
        l1_train = pc['L1'][:l1_range[i]] + pc['L1'][l1_range[i + 1]:]
        l2_train = pc['L2'][:l2_range[i]] + pc['L2'][l2_range[i + 1]:]

        joint_dataset = l0_train + l1_train + l2_train
        random.shuffle(joint_dataset)
        
        joint_ibm2 = IBM2(joint_dataset, 15)
        correct, total = 0, 0
        for lvl in levels:
            for (example_en, example_ml) in val[lvl]:
                # Score Translations
                best_trans, best_score = None, 0.0
                for t in all_commands:
                    score = joint_ibm2.score(example_en, t)
                    if score >= best_score:
                        best_trans, best_score = t, score
                
                print "Correct:", example_ml, "Predicted:", best_trans, "Score:", best_score
                if best_trans == example_ml:
                    correct += 1
                total += 1
        
        print 'Reward Selection:', float(correct) / float(total)
        reward_selection.append(float(correct) / float(total))
    
    with open(out_path, 'w') as f:
        f.write("Fold Reward Function Accuracies: %s\n" % str(reward_selection))
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
        out_path = "%s_ibm_shared_results_%s.txt" % ("clean" if clean else "raw", str(trial))
        run_cross_val(data_path, out_path)