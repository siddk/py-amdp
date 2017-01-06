#!/bin/sh

python confusion_matrices.py single 10 L0
python confusion_matrices.py single 10 L1
python confusion_matrices.py single 10 L2
python confusion_matrices.py all 10
python neural_dual.py False 10
