# Neural Ngram language model
A PyTorch implementation of [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). Code for training and data-loading based on the PyTorch example [Word level language model](https://github.com/pytorch/examples/tree/master/word_language_model).

## Setup
To get the wikitext-2 dataset, run:
```bash
./get-data.sh
```

## Usage
A word-level example:
```bash
./train.py wiki --order 5 --batch-size 32
```

A character-level example:
```bash
./train.py wiki-char --use-char --order 12 --batch-size 1024
```

If you have pretrained GloVe vectors, you can use those:
```bash
./train.py wiki --use-glove --glove-dir path/to/glove --emb-dim 50
```


## Speed and accuracy
With the following arguments one epoch takes around 45 minutes:
```bash
./train.py wiki --use-glove --emb-dim 50 --hidden-dims 100 --batch-size 128 --epochs 10
```
This reaches a test perplexity of 224.89.

## Requirements
```
python>=3.6
torch==0.3.0.post4
numpy
tqdm
```

## TODO
- [ ] Convert to torch4
- [ ] Text generation by sampling.
- [ ] Perplexity for user input.
- [ ] Softmax approximation.
