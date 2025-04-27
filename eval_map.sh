#!/bin/bash

# uncomment the following line to disable comet ml logging
# export COMET_DISABLED=True

python eval_map.py --split trainval --fname outputs/results_07trainval.pkl
python eval_map.py --split test --fname outputs/results_07test.pkl
