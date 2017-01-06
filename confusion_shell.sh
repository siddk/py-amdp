#!/bin/sh

python confusion_matrices.py single rnn 10 L0
python confusion_matrices.py single rnn 10 L1
python confusion_matrices.py single rnn 10 L2
python confusion_matrices.py all rnn 10
python confusion_matrices.py single nn 10 L0
python confusion_matrices.py single nn 10 L1
python confusion_matrices.py single nn 10 L2
python confusion_matrices.py all nn 10
python neural_dual.py False 10
