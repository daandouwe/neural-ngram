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
        if False:
            # This one does not work.
            target_logits = logits[torch.arange(logits.size(0)), targets]  # [batch]
            log_partition = self._approximate_log_partition(logits, targets, target_logits)  # [batch]
            target_log_probs = target_logits - log_partition  # [batch]
            return -1 * target_log_probs.mean(dim=0)  # NLL loss [1]
        else:
            # But this one does.
            return self._css(logits, targets)

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
        if correct:
            # `Clean` method as described in Botev et al. 2017.
            samples = self._correct_samples(
                probs=self.importance,
                targets=targets.data.numpy())
        else:
            samples = self._masked_samples(
                probs=self.importance,
                targets=targets.data.numpy(),
                cython=cython)
        return torch.from_numpy(samples).to(self.device)

    def _masked_samples(self, probs, targets, cython=False):
        def mask(probs, id):
            """Zero out probs that should not be sampled."""
            probs = np.copy(probs)
            probs[id] = 0  # zero out
            probs /= probs.sum()  # renormalize
            return probs

        """Sampling by masking undesired samples."""
        batch_size = targets.shape[0]
        samples = np.zeros((batch_size, self.num_samples), np.int64)
        for i in range(batch_size):
            if cython:
                # Faster(?) sampling using a custom cython implementation.
                samples[i] = _sample.sample(
                    probs=mask(self.importance, targets[i]),
                    num_samples=self.num_samples)
            else:
                # Regular numpy random sampling.
                samples[i] = np.random.choice(
                    np.arange(self.vocab_size),
                    size=self.num_samples,
                    p=mask(self.importance, targets[i]))  # `dirty` method by masking
        return samples

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
        batch_size = targets.shape[0]
        out_samples = np.zeros((batch_size, self.num_samples), np.int64)
        for i in range(batch_size):
            out_samples[i] = filter_correct_class(targets[i], samples, extra)
        return out_samples

    def _css(self, logits, targets):
        """
        Computes the negative log-likelihood with a CSS approximate of the Softmax.
        Args:
            logits(torch.FloatTensor): [B x V]
            targets(torch.LongTensor): [B]
        Returns:
            torch.FloatTensor: negative log-likelihood. [B]
        Author:
            Tom Pelsmaeker https://github.com/0Gemini0 (edited by Daan van Stigt)
        """
        # Obtain the positive and negative set, both parts of the normalizer
        positive_set = targets.unique()
        neg_dim = self.vocab_size - positive_set.shape[0]
        weights = np.ones(self.vocab_size) / neg_dim
        weights[positive_set] = 0
        negative_set = torch.tensor(np.random.choice(self.vocab_size, self.num_samples,
                                                     replace=False, p=weights)).to(self.device)

        # Extract the logits of the normalizer, normalizing the negative set in the process
        log_kappa = torch.log(torch.tensor(neg_dim / self.num_samples, device=self.device))
        logits[:, negative_set] += log_kappa
        normalizer = logits[:, torch.cat((positive_set, negative_set))]

        # The softmax stabilizer
        u = torch.max(normalizer, dim=1)[0]

        # Compute the log of stable exponentials. We also need to shift the logits.
        log_normalizer = torch.log(torch.exp(normalizer - u.unsqueeze(1)).sum(dim=1))
        log_logits = torch.log(torch.exp(torch.gather(logits, 1, targets.unsqueeze(1)) - u.unsqueeze(1))).squeeze(1)

        # We return the negative log likelihood
        logprobs = log_logits - log_normalizer
        return -1 * logprobs.mean(dim=0)
