#!/bin/bash

# uncomment the following line to disable comet ml logging
# export COMET_DISABLED=True

python eval_map.py --split trainval
python eval_map.py --split test
