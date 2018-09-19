#!/bin/bash

#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=00:30:00
#PBS -qgpu

module load eb
module load Python/3.6.3-foss-2017b
module load CUDA/9.0.176
module load cuDNN/7.0.5-CUDA-9.0.176

set -x # echo on

ID=4-jobs
SRCDIR=$HOME/neural-ngram
DATADIR=$HOME/data/wikitext/wikitext-2
TMP=$TMPDIR/daansdir/$ID
LOGDIR=$TMP/log
MODELSDIR=$TMP/models

# Copy training data to scratch
mkdir -p $TMP $MODELSDIR $LOGDIR
cp $DATADIR/* $TMP
for gpu in {0..3}; do
    mkdir $MODELSDIR/$gpu $LOGDIR/$gpu
done
ls -l $TMP
ls -l $MODELSDIR
ls -l $LOGDIR


# GPU 0
export CUDA_VISIBLE_DEVICES=0
python $SRCDIR/main.py train \
    --data-dir $TMP --save-dir $MODELSDIR/0 --log-dir $LOGDIR/0 --no-headers \
    --order 13 --emb-dim 100 --hidden-dims 500 \
    --epochs 20 --batch-size 64 --dropout 0.5 &

# GPU 1
export CUDA_VISIBLE_DEVICES=1
python $SRCDIR/main.py train \
    --data-dir $TMP --save-dir $MODELSDIR/1 --log-dir $LOGDIR/1 --no-headers \
    --order 13 --emb-dim 100 --hidden-dims 500 \
    --epochs 20 --batch-size 32 --dropout 0.5 &

# GPU 2
export CUDA_VISIBLE_DEVICES=2
python $SRCDIR/main.py train \
    --data-dir $TMP --save-dir $MODELSDIR/2 --log-dir $LOGDIR/2 --no-headers \
    --order 13 --emb-dim 100 --hidden-dims 500 \
    --epochs 20 --batch-size 64 --lr 0.1 --dropout 0.5 &

# GPU 3
export CUDA_VISIBLE_DEVICES=3
python $SRCDIR/main.py train \
    --data-dir $TMP --save-dir $MODELSDIR/3 --log-dir $LOGDIR/3 --no-headers \
    --order 13 --emb-dim 100 --hidden-dims 500 \
    --epochs 20 --batch-size 32 --lr 0.1 --dropout 0.5


# Copy output directory from scratch to home
mkdir $SRCDIR/log/$ID
mkdir $SRCDIR/models/$ID

cp -r $LOGDIR/* $SRCDIR/log/$ID
cp -r $MODELSDIR/* $SRCDIR/models/$ID
