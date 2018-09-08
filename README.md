# Neural ngram language model
A PyTorch implementation of [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). Code for training and data-loading based on

## Setup
Get the wikitext-2 dataset.

## Usage
A word-level example:
```bash
./train.py wiki --order 5 --batch-size 32
```

A character-level example:
```bash
./train.py wiki-char --use-char --order 12 --batch-size 1024
```

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
