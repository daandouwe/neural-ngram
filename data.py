import os
import torch

from utils import UNK, SOS, EOS, SOS_CHAR, EOS_CHAR


class Dictionary(object):
    def __init__(self):
        self.w2i = {}
        self.i2w = []

    def add_word(self, word):
        if word not in self.w2i:
            self.i2w.append(word)
            self.w2i[word] = len(self.i2w) - 1
        return self.w2i[word]

    def __len__(self):
        return len(self.i2w)


class Corpus(object):
    def __init__(self, path, headers=True, lower=False, chars=False):
        ext = 'raw' if chars else 'tokens'
        path = path + '-raw' if chars else path
        self.headers = headers
        self.lower = lower
        self.chars = chars
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, f'wiki.train.{ext}'))
        self.valid = self.tokenize(os.path.join(path, f'wiki.valid.{ext}'))
        self.test = self.tokenize(os.path.join(path, f'wiki.test.{ext}'))

    def get_data(self, path):
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                line = line.strip()
                if line.startswith('=') and not self.headers:
                    words =  []
                elif self.chars:
                    words = list(SOS_CHAR + line + EOS_CHAR)
                else:
                    words = [word.lower() for word in line.split()] if self.lower else line.split()
                    words = [SOS] + words + [EOS]
                yield words

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        tokens = 0
        for words in self.get_data(path):
            tokens += len(words)
            for word in words:
                self.dictionary.add_word(word)

        # Tokenize file content
        ids = torch.LongTensor(tokens)
        token = 0
        for words in self.get_data(path):
            for word in words:
                ids[token] = self.dictionary.w2i[word]
                token += 1

        return ids

    @property
    def vocab_size(self):
        return len(self.dictionary.i2w)
