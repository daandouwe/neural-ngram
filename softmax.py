"""
Softmax approximation with Complementary Sum Sampling.
Follows http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/AISTATS2017.pdf.
"""
import argparse
import os

import torch
import torch.nn as nn
import numpy as np


class ApproximateLoss(nn.Module):
    """Approximate a softmax crosse entropy classification loss."""
    def __init__(
        self,
        vocab_size,
        method='importance',
        num_samples=250,
        alpha=0.75,
        unigram=None,
        device=None,
    ):
        super(ApproximateLoss, self).__init__()
        assert method in ('importance', 'bernoulli'), method
        if method == 'importance':
            assert unigram is not None, 'need unigrams for importance sampling'
        self.vocab_size = vocab_size
        self.method = method
        self.num_samples = num_samples
        self.alpha = alpha
        self.temp = alpha
        self.device = device
        self.unigram = self.scale(unigram, alpha)

    def __call__(self, logits, targets):
        target_logits = logits[torch.arange(logits.size(0)), targets]  # [batch]
        target_scores = torch.exp(target_logits)  # [batch]
        Z_tilde = self._approximate_partition(logits, targets, target_scores)  # [batch]
        target_probs = target_scores / Z_tilde  # [batch]
        return -1 * torch.log(target_probs).mean(dim=0)  # NLL loss [1]

    def _approximate_partition(self, logits, targets, target_scores):
        with torch.no_grad():
            idx = self.sample(targets)  # [batch, num_samples]
            sampled_scores = torch.exp(torch.gather(logits.data, 1, idx))  # [batch, num_samples]
            sampled_probs = self.unigram[idx.numpy()]  # [batch, num_samples]
            weights = 1.0 / (
                self.num_samples * torch.from_numpy(sampled_probs)
            ).float()
            Z_tilde = target_scores.data + (weights * sampled_scores).sum(dim=1)
        return Z_tilde

    def scale(self, unigram, temp):
        assert 0 <= temp <= 1, temp
        unigram = unigram**temp
        unigram /= unigram.sum()
        return unigram

    def sample(self, targets):
        """
        Currently the method is incorrect. Use this:
            `For all of the methods we used S = 250 samples which
            are shared across every word in every sentence in the
            minibatch.`
        """
        def mask(probs, id):
            probs = np.copy(probs)
            probs[id] = 0
            return probs / probs.sum()

        samples = np.zeros((targets.size(0), self.num_samples), dtype=np.int64)
        for i, id in enumerate(targets.data.numpy()):
            samples[i] = np.random.choice(
                np.arange(self.vocab_size),
                size=self.num_samples,
                p=mask(self.unigram, id)
            )
        return torch.from_numpy(samples).to(self.device)
