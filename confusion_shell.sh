#!/bin/sh

python confusion_matrices.py single 10 L0
python confusion_matrices.py single 10 L1
python confusion_matrices.py single 10 L2
python confusion_matrices.py all 10
python dual_model.py False 10
