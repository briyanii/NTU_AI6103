# Faster RCNN
This repository attempts to reimplement the Faster RCNN paper from 2015.

# Steps
0. install `conda`
1. `conda env create -f environment.yml -n MyEnv`
recreate conda environment
2. `conda activate MyEnv`
activate conda env
3. `python download.py`
download the VOC Pascal Dataset
4. `sh train_4step.sh`
run the 4-step training process as described in the paper
5. `sh eval_map.sh`
evaluate mAP for 'voc pascal 07 trainval' and 'voc pascal 07 test'

# Relevant Papers
- Faster RCNN: https://arxiv.org/pdf/1506.01497v1
- Fast RCNN: https://arxiv.org/pdf/1504.08083
- RCNN: https://arxiv.org/abs/1311.2524

# Experiment Tracking
The checkpoints every 10K steps during the 4-step training process can be found at https://www.comet.com/ai6103/faster-rcnn/view/new/experiments


