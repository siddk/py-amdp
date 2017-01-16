#!/bin/sh

python ibm_cross_val.py True
python ibm_cross_val.py False
python dual_nn_cross_val.py True
python dual_nn_cross_val.py False