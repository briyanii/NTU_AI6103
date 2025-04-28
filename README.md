# Faster RCNN
This repository attempts to reimplement the Faster RCNN paper from 2015.

# Steps
### 1. install `conda`
### 2. `conda env create -f environment.yml -n MyEnv`
- recreate conda environment
### 3. `conda activate MyEnv`
- activate conda env
### 4. `python download.py`
- download the VOC Pascal Dataset
### 5. `sh train_4step.sh`
- run the 4-step training process as described in the paper
### 6. `python plot_loss_curve.py`
- plot loss curves for 4-step alternating training
### 7. `sh eval_map.sh`
- evaluate mAP for 'voc pascal 07 trainval' and 'voc pascal 07 test'

# Relevant Papers
- Faster RCNN: https://arxiv.org/abs/1506.01497v1
- Fast RCNN: https://arxiv.org/abs/1504.08083
- RCNN: https://arxiv.org/abs/1311.2524

# Experiment Tracking
- The checkpoints every 10K steps during the 4-step training process can be found at https://www.comet.com/ai6103/faster-rcnn/view/new/experiments
### Environment Variables
#### `COMET_DISABLED`
- set to `True` to disable comet ml logging
#### `COMET_API_KEY`
- must be set if `COMET_DISABLED` is not set to "True"

