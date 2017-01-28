"""
process_results.py

Parse output results files for cross validation models and produce mean level inference and grounding accuracies
"""
from __future__ import division
import sys

model_map = {'IBM2': 'raw_ibm_results_{0}.txt',
             'MNN': 'raw_dual_nn_results_{0}.txt',
             'MRNN': 'raw_dual_rnn_results_{0}.txt',
             'SRNN': 'raw_single_rnn_results_{0}.txt'}

def parse_results(path):
    trials = 3
    lvl_inf = 0.0
    rf_inf = 0.0
    for i in range(trials):
        lines = open(path.format(i), 'rb').readlines()
        lines = lines[-2:]
        lvl_inf += float(lines[0].split(": ")[-1])
        rf_inf += float(lines[1].split(": ")[-1])
    return lvl_inf / trials, rf_inf / trials


if __name__ == "__main__":
    # Read Command Line Arguments
    args = sys.argv
    model = args[1]

    main_path = model_map[model]

    print 'Mean Level Inference %f \t Mean Grounding Accuracy %f' % parse_results(main_path)
