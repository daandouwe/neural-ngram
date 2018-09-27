"""
Softmax approximation with Complementary Sum Sampling.
Follows Botev et al. 2017 (http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/AISTATS2017.pdf).
"""
import argparse
import os

import torch
import torch.nn as nn
import numpy as np

from sample import _sample


class ApproximateLoss:
    """Approximate a softmax for a cross entropy classification loss."""
    def __init__(
        self,
        vocab_size,
        method='importance',
        num_samples=250,
        alpha=0.75,
        unigram=None,
        device=None,
    ):
        assert method in ('importance', 'bernoulli'), method
        self.vocab_size = vocab_size
        self.method = method
        self.num_samples = num_samples
        self.alpha = alpha
        self.temp = alpha
        self.device = device
        if unigram is not None:
            # Use scaled unigram distribution as importance distribution.
            self.importance = self.scale(unigram, alpha)
        else:
            # Use uniform distribution as importance distribution.
            self.importance = np.ones(vocab_size) / vocab_size

    def __call__(self, logits, targets):
        """Compute negative log likelihood loss with approximated log-normalizer."""
        target_logits = logits[torch.arange(logits.size(0)), targets]  # [batch]
        log_partition = self._approximate_log_partition(logits, targets, target_logits)  # [batch]
        target_log_probs = target_logits - log_partition  # [batch]
        return -1 * target_log_probs.mean(dim=0)  # NLL loss [1]

    def _approximate_log_partition(self, logits, targets, target_logits):
        with torch.no_grad():
            idx = self.sample(targets)  # [batch, num_samples]
            sampled_logits = torch.gather(logits.data, 1, idx)  # [batch, num_samples]
            importance_probs = torch.from_numpy(self.importance[idx.numpy()]).float()  # [batch, num_samples]
            log_importance_weights = -1 * (
                torch.log(importance_probs) + torch.log(torch.tensor(float(self.num_samples))))  # [batch, num_samples]
            negative_approx = torch.logsumexp(
                log_importance_weights + sampled_logits, dim=1)
            positive_approx = torch.logsumexp(target_logits, dim=0) * torch.ones(target_logits.shape)  # [batch]
            log_partition = torch.logsumexp(
                torch.cat((positive_approx.unsqueeze(1), negative_approx.unsqueeze(1)), dim=1), dim=1)  # [batch, num_samples + 1]
        return log_partition

    def scale(self, probs, temp):
        assert 0 <= temp <= 1, temp
        probs = probs**temp  # scale
        probs /= probs.sum()  # renormalize
        return probs

    def sample(self, targets, correct=False, cython=False):
        def mask(probs, id):
            """Zero out probs that should not be sampled."""
            probs = np.copy(probs)
            probs[id] = 0  # zero out
            probs /= probs.sum()  # renormalize
            return probs

        if correct:
            # `Clean` method as described in Botev et al. 2017.
            samples = self._correct_samples(
                probs=self.importance,
                targets=targets.data.numpy())
        elif cython:
            # Fast sampling using a custom cython implementation.
            samples = _sample.sample(
                probs=mask(self.importance, targets.data.numpy()),  # `dirty` method by masking
                num_samples=self.num_samples)
            samples = np.tile(samples, (targets.size(0), 1))
        else:
            # Standard numpy random.
            samples = np.random.choice(
                np.arange(self.vocab_size),
                size=self.num_samples,
                p=mask(self.importance, targets.data.numpy()))  # `dirty` method by masking
            samples = np.tile(samples, (targets.size(0), 1))
        return torch.from_numpy(samples).to(self.device)

    def _correct_samples(self, probs, targets, num_extra=100):
        """The sampling method exactly as described in Botev et al. 2017."""
        def filter_correct_class(target, samples, extra):
            """Collect first `num_samples` that are not equal to target."""
            all_samples = np.concatenate((samples, extra))
            filtered = np.array([sample for sample in all_samples if sample != target])[:self.num_samples]
            # Can end up with too few samples, a very annoying downside of this method.
            assert len(filtered) == self.num_samples, len(filtered)
            return filtered

        samples = np.random.choice(
            np.arange(self.vocab_size),
            size=self.num_samples,
            p=self.importance)
        extra = np.random.choice(
            np.arange(self.vocab_size),
            size=num_extra,
            p=self.importance)
        out_samples = np.zeros((targets.shape[0], self.num_samples), np.int64)
        for i in range(targets.shape[0]):
            out_samples[i] = filter_correct_class(targets[i], samples, extra)
        return out_samples
