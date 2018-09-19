#!/bin/bash

#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=02:30:00
#PBS -qgpu

module load eb
module load Python/3.6.3-foss-2017b
module load CUDA/9.0.176
module load cuDNN/7.0.5-CUDA-9.0.176

set -x # echo on

ID=$PBS_JOBID
# ID=123
SRCDIR=$HOME/neural-ngram
DATADIR=$HOME/data/wikitext/wikitext-2
TMP=$TMPDIR/daansdir/$ID
LOGDIR=$TMP/log
MODELSDIR=$TMP/models

# Copy training data to scratch
mkdir -p $TMP $MODELSDIR $LOGDIR
cp $DATADIR/* $TMP
ls -l $TMP

python $SRCDIR/main.py train \
    --data-dir $TMP --save-dir $MODELSDIR --log-dir $LOGDIR --no-headers \
    --order 13 --emb-dim 300 --hidden-dims 1400 \
    --epochs 40 --batch-size 256 --dropout 0.65

# Copy output directory from scratch to home
mkdir $SRCDIR/log/$ID
mkdir $SRCDIR/models/$ID

cp -r $LOGDIR/* $SRCDIR/log/$ID
cp -r $MODELSDIR/* $SRCDIR/models/$ID
