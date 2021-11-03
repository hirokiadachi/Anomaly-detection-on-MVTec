# Anomaly Detection on MVTec dataset
This repo. is the implementation of ICLR 2020 paper, "ITERATIVE ENERGY-BASED PROJECTION ON A NORMAL DATA MANIFOLD FOR ANOMALY LOCALIZATION", on Pytorch.

Paper link: https://arxiv.org/abs/2002.03734

The network architecture and training details are same as "[Improving Unsupervised Defect Segmentation
by Applying Structural Similarity To Autoencoders](https://arxiv.org/abs/1807.02011)".

### training phase
```
python3 train.py --data_root ./mvtec_anomaly_detection --category grid --gpu 0 --epochs 20000 --model_type 'ae'
```
- data_root: Please refer to the path up to the MVTec dataset directory.
- category: Please select the category that you want to train.
- model_type: Please choose the model type from ae or vae.

### evaluation phase (anomaly detection)
```
python3 evaluation.py --gpu 0 --model_path ./checkpoint/grid/model --data_root ./mvtec_anomaly_detection --category grid --model_type 'ae'
```
- model_path: Please refer to the trained model path.
- data_root: Please refer to the path up to the MVTec dataset directory.
- category: Please select the category that you want to evaluate.
- model_type: Please choose the model type from ae or vae.

## Dataset
This implementation presupporses to use **MVTec AD**, so if you'd like to excuse this, you should get MVTec AD bellow link.<br>
MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad


### Categories
<details>
  <summary>Textures</summary>
  <br>
  <ul>
    <li> carpet </li>
    <li> grid </li>
    <li> leather </li>
    <li> tile </li>
    <li> wood </li>
  </ul>
</details>

<details>
  <summary>Objects</summary>
  <br>
  <ul>
    <li> bottle </li>
    <li> cable </li>
    <li> capsule </li>
    <li> hazelnut </li>
    <li> metalnut </li>
    <li> pill </li>
    <li> screw </li>
    <li> toothbrash </li>
    <li> transistor </li>
    <li> zipper </li>
  </ul>
</details>

### data structures
```
mvtec_anomaly_detection
 |--bottle
 |  |--ground_truth
 |  |  |--broken_large
 |  |  |  |--000_mask.png
 |  |  |  |     :
 |  |  |--broken_small
 |  |  |  |--000_mask.png
 |  |  |  |     :
 |  |  |--contamination
 |  |  |  |--000_mask.png
 |  |  |  |     :
 |  |--test
 |  |  |--broken_large
 |  |  |  |--000.png
 |  |  |  |     :
 |  |  |--broken_small
 |  |  |  |--000.png
 |  |  |  |     :
 |  |  |--contamination
 |  |  |  |--000.png
 |  |  |  |     :
 |  |  |--good
 |  |  |  |--000.png
 |  |  |  |     :
 |  |--train
 |  |  |--good
 |  |  |  |--000.png
 |  |  |  |     :
 |  |--license.txt
 |  |--reaadme.txt
 |--cable
 |    :
 |--license.txt
 |--mvtec_anomaly_detection.tar
 |--readme.txt
```

## Docker file
This dockerfile contains the necessary modules to run this code.
Firstly, please build a docker image with Dockerfile, and then run a docker container.
```
cd docker_file
docker build -t [REPOSITORY]:[TAG] .   Build docker image
docker run --runtime=nvidia -it --rm -p [port number]:8888 -v [local directory path]:/root --name [container name] --ipc=host [REPOSITORY]:[TAG]
```
If you'd like to use Jupyter lab, you execute `set_jupyter_lab.sh` after ran a docker container, please.
```
chmod 777 set_jupyter_lab.sh
./set_jupyter_lab.sh
```

## Requirements
Pytorch: over 1.7.x<br>
scikit-image<br>
sklearn<br>
tqdm<br>
PIL
