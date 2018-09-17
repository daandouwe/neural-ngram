#!/bin/bash

#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=03:00:00
#PBS -qgpu

module load eb
module load Python/3.6.3-foss-2017b
module load CUDA/9.0.176
module load cuDNN/7.0.5-CUDA-9.0.176

set -x # echo on
SRCDIR=$HOME/neural-ngram
DATADIR=$HOME/data/wikitext/wikitext-2
TMP=$TMPDIR/daansdir2

# Copy training data to scratch
cp -r $DATADIR $TMP

mkdir -p $TMP/log
mkdir -p $TMP/models

ls -l $TMP

python $SRCDIR/main.py train \
    --data-dir $TMP --save-dir $TMP/models --log-dir $TMP/log \
    --order 13 --emb-dim 100 --hidden-dims 800,100 --tied \
    --epochs 30 --batch-size 32 --lr 1e-3

# Copy output directory from scratch to home
# cp -r $TMPDIR/log $SRCDIR/log
# cp -r $TMPDIR/models $SRCDIR/models
cp -r $TMPDIR $SRCDIR
