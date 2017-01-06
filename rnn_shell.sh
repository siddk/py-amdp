#!/bin/sh

python data_curve.py L2 L2 rnn 5
python data_curve.py L1 L1 rnn 5
python data_curve.py L0 L0 rnn 5
python data_curve.py L_ALL L_ALL rnn 5