import os

import torch
from torch.autograd import Variable
import torch.nn as nn

from glove import load_glove


class NeuralNgram(nn.Module):
    """Neural ngram language model."""
    def __init__(self, order, emb_dim, vocab_size, hidden_dims=(200,), use_glove=False, dropout=0.0):
        """
        Args:
            order (int): the order of the language model, i.e. length of the history
            emb_dim (int): dimension of the word embeddings
            vocab_size (int): size of vocabulary
            hidden_dims (list): a list of integers that specify of the hidden layers
        """
        super(NeuralNgram, self).__init__()
        self.order = order
        self.emb_dim = emb_dim
        self.hidden_dims = hidden_dims
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.mlp = MLP(
            in_dim=order*emb_dim,
            hidden_dims=hidden_dims,
            activation='Tanh',
            dropout=dropout)
        self.decoder = nn.Linear(hidden_dims[-1], vocab_size)
        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            if len(param.shape) >= 2 and param.requires_grad:
                nn.init.xavier_uniform_(param)

    def load_glove(self, glove_dir, i2w):
        words = [i2w[i] for i in range(len(i2w))]
        logpath = os.path.join('log', 'glove.error.txt')
        # Log all words that had no glove vector (for later reference).
        with open(logpath, 'w') as f:
            embeddings = load_glove(words, self.emb_dim, glove_dir, logfile=f)
            self.embedding.weight.data = embeddings
            self.embedding.weight.requires_grad = False

    def tie_weights(self):
        if self.hidden_dims[-1] != self.emb_dim:
            raise ValueError('When using the tied flag, nhid must be equal to emsize')
        self.decoder.weight = self.embedding.weight

    def forward(self, indices):
        x = self.embedding(indices)
        x = x.view(indices.size(0), -1)  # Concatenate along embedding dimension.
        x = self.dropout(x)
        hidden = self.mlp(x)
        logits = self.decoder(hidden)
        return logits


class MLP(nn.Module):
    """Module for an MLP with dropout."""
    def __init__(self, in_dim, hidden_dims, activation, dropout):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        act_fn = getattr(nn, activation)
        layers = zip([in_dim] + hidden_dims[:-1], hidden_dims)
        for i, (in_dim, out_dim) in enumerate(layers):
            layer = nn.Linear(in_dim, out_dim)
            self.layers.add_module('fc_{}'.format(i), layer)

            if activation:
                self.layers.add_module('{}_{}'.format(activation, i),
                                       act_fn())
            if dropout:
                self.layers.add_module('dropout_{}'.format(i),
                                       nn.Dropout(dropout))

    def forward(self, x):
        return self.layers(x)
