# Neural ngram language model
A PyTorch implementation of [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). Code for training and data-loading based on the PyTorch example [Word level language model](https://github.com/pytorch/examples/tree/master/word_language_model).

## Setup
To get the wikitext-2 dataset, run:
```bash
./get-data.sh
```

## Usage
A word-level example:
```bash
./main.py train --name wiki --order 5 --batch-size 32
```

A character-level example:
```bash
./main.py train --name wiki-char --use-char --order 12 --batch-size 1024
```

If you have pretrained GloVe vectors, you can use those:
```bash
./main.py train --name wiki --use-glove --glove-dir path/to/glove --emb-dim 50
```

Some other data arguments are:
```bash
--lower       # Lowercase all words in training data.
--no-header   # Remove all headers such as `=== History ===`.
```

## Speed and accuracy
With the following arguments one epoch takes around 45 minutes:
```bash
./main.py train --name wiki --use-glove --emb-dim 50 --hidden-dims 100 --batch-size 128 --epochs 10
```
This reaches a test perplexity of 224.89.

The loss is acting kinda funny though...

![loss](https://github.com/daandouwe/neural-ngram/blob/master/log/losses.png)

O loss surfaces...

## Generate text
To generate text, use:
```bash
./main.py generate --checkpoint path/to/saved/model
```
The `<eos>` token is replaced with a newline, and the rest is printed as is.

Other generation arguments are:
```bash
--temperature 0.9   # Temperature to manipulate distribution.
--start             # Provide an optional start of the generated text (can be longer than order)
--no-unk            # Do not generate unks, especially useful for low --temperature.
--no-sos            # Do not print <sos> tokens
```

See some generated text in [generate.txt](https://github.com/daandouwe/neural-ngram/blob/master/generated.txt).

## Plot embeddings
To visualize the trained embeddings of the model, use:
```bash
./main.py plot --checkpoint path/to/saved/model
```
This fits a 2D t-SNE plot with K-means cluster coloring of the 1000 most common words in the dataset. The requires [Bokeh](https://bokeh.pydata.org/en/latest/) for plotting and [scikit-learn](http://scikit-learn.org/stable/index.html) for t-SNE and K-means.

See an example html [here](https://github.com/daandouwe/neural-ngram/blob/master/plots/wiki.tsne.html). (Github does not render html files. To render, download and open, or use the link [http://htmlpreview.github.com/?https://github.com/daandouwe/neural-ngram/blob/master/plots/wiki.tsne.html](http://htmlpreview.github.com/?https://github.com/daandouwe/neural-ngram/blob/master/plots/wiki.tsne.html).)

## Requirements
```
python>=3.6
torch==0.3.0.post4
numpy
tqdm
```

## TODO
- [ ] Convert to torch4
- [X] Text generation by sampling.
- [ ] Plot embeddings with t-SNE
- [ ] Perplexity for user input.
- [ ] Softmax approximation.
