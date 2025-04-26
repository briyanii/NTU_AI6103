# Relevant Papers
- Faster RCNN: https://arxiv.org/pdf/1506.01497v1
- Fast RCNN: https://arxiv.org/pdf/1504.08083

# Steps (WIP)
0. install `conda`
1. `conda env create -f environment.yml -n MyEnv` create conda environment
2. `conda activate MyEnv` activate conda env
3. `python download.py` to download the required dataset / model weights
4. `sh train_4step.sh` to run the 4 step training as described in the paper

# Evaluation and Visualization (WIP)
- mAP calculation and Precision-Recall curve plotting: 

```bash
python map.py --year \[year of the eval dataset\] --image_set \[train, val, trainval, test\] --model_path \[path to model weights\]
```

- Detection result visualization

```bash
python map.py --year \[year of the eval dataset\] --image_set \[train, val, trainval, test\] --model_path \[path to model weights\]
```