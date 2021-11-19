#!/bin/bash

for s in 1 2 3 4 5 6 7 8 9 10

do 

	python3 cvarRL/evaluate.py --model experiment2_1_0.05_0.7_$s --cost 0.7 --stochasticity 0 --type 5m --episodes 1 --argmax $True --procs 1 --budget 1

	python3 cvarRL/evaluate.py --model experiment3_25_0.05_0.7_$s --cost 0.7 --stochasticity 0 --type 5m --episodes 1 --argmax $True --procs 1 --budget 1

	python3 cvarRL/evaluate.py --model experiment3_100_0.05_0.7_$s --cost 0.7 --stochasticity 0 --type 5m --episodes 1 --argmax $True --procs 1 --budget 1

done

python3 cvarRL/figure3.py --model experiment2_1_0.05_0.7_1 --cost 0.7 --stochasticity 0.05 --type 5m --episodes 300000 --budget 1

python3 cvarRL/figure3.py --model experiment3_25_0.05_0.7_1 --cost 0.7 --stochasticity 0.05 --type 5m --episodes 300000 --budget 25

python3 cvarRL/figure3.py --model experiment3_100_0.05_0.7_1 --cost 0.7 --stochasticity 0.05 --type 5m --episodes 300000 --budget 100

python3 cvarRL/create_save_figs.py
