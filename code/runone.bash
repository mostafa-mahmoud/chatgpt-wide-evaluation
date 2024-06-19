#!/bin/bash

for action in hptune predict ;
do
    for features in RoBERTa BoW ;
    do
        ./run_experiment.py $action $features $1
    done
done
