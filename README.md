# Relevant Papers
- Faster RCNN: https://arxiv.org/pdf/1506.01497v1
- Fast RCNN: https://arxiv.org/pdf/1504.08083

# Steps (WIP)
1. `python download.py` to download the required dataset / model weights
2. `python step1a.py` run training script for RPN
3. `python step1b.py` run script to get RPN roi proposals for step2
4. `python step2.py` run training script for FastRCNN Detection Head
5. `python step3.py` run training script for FasterRCNN RPN Head
6. `python step4.py` run training script for FasterRCNN Detection Head

