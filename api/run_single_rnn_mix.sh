#!/bin/sh

echo 'L0 - L1'
time mpiexec -n 5 python run_single_rnn_mix.py L0 L1
echo ''
echo ''

echo 'L0 - L2'
time mpiexec -n 5 python run_single_rnn_mix.py L0 L2
echo ''
echo ''

echo 'L1 - L2'
time mpiexec -n 5 python run_single_rnn_mix.py L1 L2
echo ''
echo ''

echo 'L1 - L0'
time mpiexec -n 5 python run_single_rnn_mix.py L1 L0
echo ''
echo ''

echo 'L1 - L2'
time mpiexec -n 5 python run_single_rnn_mix.py L1 L2
echo ''
echo ''

echo 'L2 - L0'
time mpiexec -n 5 python run_single_rnn_mix.py L2 L0
echo ''
echo ''

echo 'L2 - L1'
time mpiexec -n 5 python run_single_rnn_mix.py L2 L1
echo ''
echo ''

echo 'L0 - L0'
time mpiexec -n 5 python run_single_rnn_mix.py L0 L0
echo ''
echo ''

echo 'L1 - L1'
time mpiexec -n 5 python run_single_rnn_mix.py L1 L1
echo ''
echo ''

echo 'L2 - L2'
time mpiexec -n 5 python run_single_rnn_mix.py L2 L2
echo ''
echo ''

