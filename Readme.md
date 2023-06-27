# BLAD
The source code of paper on SC2023.

## Introduction
Dynamic graph networks are widely used for learning time-evolving graphs, but prior work on training these networks is inefficient due to communication overhead, long synchronization, and poor resource usage. Therefore, we propose a system called BLAD to consider the above factors, to improve the training efficiency and decrease the cross-GPU communication on the GPU datacenters. 

## Hardware and Software Depedencies

- Hardware&software requirements

  1. Hardware Requirements

     1. CPU: Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
     2. Memroy: 252G
     3. NVIDIA RTX 2080 Ti GPU

  2. Software Requirements

     1. Ubuntu 20.04
     2. Docker 24.02
     3. GPU Driver: 450.51
     4. CUDA 11.1
     5. CUDNN 8.0
     6. Miniconda3-py37_4.9.2
     7. Pytorch 1.9.0
     8. DGL 0.8.0

## Preparing environment
1. Download and run the provided runtime backend with docker.
```shell
$ docker pull midway2018/blad_runtime
$ docker run -it --gpus=all --ipc=host midway2018/blad_runtime /bin/bash 
$ git clone https://github.com/fkh12345/BLAD.git
```
2. Activate conda and create python environment with essential dependencies
```shell
$ conda activate base
$ cd BALD
$ pip install -r requirement.txt
```

## How to run
The workflow of running the dynamic GNN model training can be summarized as follow:
1. Download the public datasets, and you can use our scripts to easily obtain them. 
```shell
$ cd get-datasets
$ # For real dynamic datasets
$ ./get_public_dataset.sh
$ python create_dgl_graph.py
$ # For static datasets (arxiv, products, reddit)
$ python convert_ogb.py

$ # After download all the datasets
$ python convert_graph_samply.py --dataset YOUR_TEST_DATASET
```

All the training data are stored in the `data/` folder.

2. Activate cuda mps service.
```shell
$ nvidia-cuda-mps-control -d
$ echo start_server -uid $UID$| nvidia-cuda-mps-control
```
3. Execute the training scripts. All the test cases are stored in one separated folder..
```shell
$ # One case for example (Executing egcn using the arxiv dataset)
$ cd experiments/ogb_graph/arxiv/egcn
$ python multi_train_ogb_sample_group.py
```


