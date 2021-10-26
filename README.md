# Anomaly Detection on MVTec dataset
This repo. is the implementation of ICLR 2020 paper, "ITERATIVE ENERGY-BASED PROJECTION ON A NORMAL DATA MANIFOLD FOR ANOMALY LOCALIZATION", on Pytorch.

Paper link: https://arxiv.org/abs/2002.03734

### training phase
```
python3 train.py --data_root ./mvtec_anomaly_detection --category grid --gpu 0 --epochs 20000
```
- data_root: Please refer to the path up to the MVTec dataset directory.
- category: Please select the category that you want to train.

### evaluation phase (anomaly detection)
```
python3 evaluation.py --gpu 0 --model_path ./checkpoint/grid/model --data_root ./mvtec_anomaly_detection --category grid
```
- model_path: Please refer to the trained model path.
- data_root: Please refer to the path up to the MVTec dataset directory.
- category: Please select the category that you want to evaluate.

## Dataset
This implementation presupporses using **MVTec AD**, so if you'd like to excuse this, you should get MVTec AD bellow link.
MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad


## Requirements
Pytorch: over 1.7.x<br>
scikit-image<br>
sklearn<br>
tqdm<br>
PIL
