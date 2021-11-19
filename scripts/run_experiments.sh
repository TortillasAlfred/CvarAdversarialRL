#!/bin/bash

for s in 1 2 3 4 5 6 7 8 9 10

do

	python3 cvarRL/train.py --algo ppo --model experiment2_1_0.05_0.7_$s --budget 1 --stochasticity 0.05 --cost 0.7 --frames 5000000 --save-interval 10 --max_lr 0.005 --min_lr 0.0001 --seed $s

	python3 cvarRL/train.py --algo ppo --model experiment3_25_0.05_0.7_$s --budget 25 --stochasticity 0.05 --cost 0.7 --frames 5000000 --save-interval 10 --max_lr 0.005 --min_lr 0.0001 --seed $s

	python3 cvarRL/train.py --algo ppo --model experiment3_100_0.05_0.7_$s --budget 100 --stochasticity 0.05 --cost 0.7 --frames 5000000 --save-interval 10 --max_lr 0.005 --min_lr 0.0001 --seed $s

done
​