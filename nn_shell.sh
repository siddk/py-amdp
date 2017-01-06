#!/bin/sh

python data_curve.py L2 L2 nn 10
python data_curve.py L1 L1 nn 10
python data_curve.py L0 L0 nn 10
python data_curve.py L_ALL L_ALL nn 10