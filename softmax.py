"""
Softmax approximation with Complementary Sum Sampling.
Source: http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/AISTATS2017.pdf
"""
import argparse
import os

import torch
import torch.nn as nn
import numpy as np

class ApproximateLoss(nn.Module):
    """Approximate a softmax classification loss"""
    def __init__(self, vocab_size, method='importance', num_samples=250, alpha=0.75, unigram=None):
        super(ApproximateLoss, self).__init__()
        assert method in ('importance', 'bernoulli'), method
        if method == 'importance':
            assert unigram is not None
        self.vocab_size = vocab_size
        self.method = method
        self.num_samples = num_samples
        self.alpha = alpha
        self.unigram = self.scale(unigram, alpha)
        self.temp = alpha

    def __call__(self, logits, y):
        samples = self.sample()
        print(samples)
        quit()
        loss = None
        return loss

    def scale(self, unigram, temp):
        assert 0 <= temp <= 1, temp
        unigram = unigram**temp
        unigram /= unigram.sum()
        return unigram

    def sample(self):
        return np.random.choice(
            np.arange(self.vocab_size), size=self.num_samples, p=self.unigram)


if __name__ == '__main__':
    main()
