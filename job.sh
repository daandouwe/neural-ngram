#!/bin/bash

#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=1:00:00
#PBS -qgpu

module load eb
module load Python/3.6.3-foss-2017b
module load CUDA/9.0.176
module load cuDNN/7.0.5-CUDA-9.0.176

./train.py wiki --batch-size 64 --use-glove --glove-dir ~/glove --emb-dim 50 --hidden-dims 100 --lr 1e-4
