# Neural ngram language model
A PyTorch implementation of [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).

## Setup
Get the wikitext-2 dataset.
```
TODO
```

## Usage
A word-level example:
```bash
./train.py <name> --order 5 --batch-size 1024 --print-every 100
```

A character-level example:
```bash
./train.py <name> --use-char --order 12 --batch-size 1024 --print-every 1000
```

## Requirements
```
python>=3.6
pytorch==0.3.0.post4
numpy
tqdm
```
