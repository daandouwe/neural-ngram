#!/usr/bin/env python

import os
import argparse
import csv
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm

from data import Corpus
from model import NeuralNgram
from utils import UNK, SOS, SOS_CHAR, print_args, write_losses, list_hidden_dims


def batchify(data, batch_size):
	# Work out how cleanly we can divide the dataset into args.batch_size parts.
	nbatch = data.size(0) // batch_size
	# Trim off any extra elements that wouldn't cleanly fit (remainders).
	data = data.narrow(0, 0, nbatch * batch_size)
	# Evenly divide the data across the batch_size batches.
	data = data.view(batch_size, -1).t().contiguous()
	return data


def get_batch(data, i, order):
	x = Variable(torch.t(data[i:i+order]))
	y = Variable(data[i+order].view(-1))
	return x, y


def evaluate(data, model, criterion):
	total_loss = 0
	n_steps = data.size(0) - model.order - 1
	for i in tqdm(range(n_steps)):
		x, y = get_batch(data, i, model.order)
		out = model.forward(x)
		loss = criterion(out, y)
		total_loss += loss.data.numpy()[0]
	return total_loss / n_steps


def main(args):
	cuda = torch.cuda.is_available()

	# Set seed for reproducibility
	torch.manual_seed(42)
	np.random.seed(42)

	data_dir = os.path.expanduser(args.data_dir)
	corpus = Corpus(data_dir, chars=args.use_chars)
	train_data = batchify(corpus.train, args.batch_size)
	val_data = batchify(corpus.valid, args.batch_size)
	test_data = batchify(corpus.test, args.batch_size)


	# Logging
	print_args(args)
	print('Using cuda: {}'.format(cuda))
	print('Size of training set: {:,} tokens'.format(np.prod(train_data.size())))
	print('Size of validation set: {:,} tokens'.format(np.prod(val_data.size())))
	print('Size of test set: {:,} tokens'.format(np.prod(test_data.size())))
	print('Vocabulary size: {:,}'.format(corpus.vocab_size))
	print('Example data:')
	i2w = corpus.dictionary.i2w
	for k in range(5):
		x = map(lambda i: i2w[i], train_data[:args.order, k].numpy())
		y = [i2w[train_data[args.order, k]]]
		print(list(x), y)


	# Initialize model
	hidden_dims = list_hidden_dims(args.hidden_dims)
	model = NeuralNgram(
		order=args.order, emb_dim=args.emb_dim, vocab_size=corpus.vocab_size, hidden_dims=hidden_dims)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	scheduler = ReduceLROnPlateau(optimizer, threshold=1e-4, patience=1, factor=.5, verbose=True)
	criterion = nn.CrossEntropyLoss()
	if args.use_glove:
		model.load_glove(args.glove_dir, i2w=corpus.dictionary.i2w)
	if args.tied:
		model.tie_weights()
	if cuda:
		model.cuda()


	# Training
	print('Training...')
	losses = []
	num_steps = train_data.size(0) - args.order - 1
	best_val_loss = None
	t0 = time.time()
	try:
		for epoch in range(1, args.epochs+1):
			epoch_start_time = time.time()
			for step in range(1, num_steps+1):
				x, y = get_batch(train_data, step-1, args.order)

				# Forward pass
				logits = model.forward(x)
				loss = criterion(logits, y)

				# Update parameters
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# Bookkeeping
				np_loss = loss.data.numpy()[0]
				losses.append(np_loss)

				if step % args.print_every == 0:
					ls = losses[-args.print_every:]
					avg_loss = sum(ls) / args.print_every
					t1 = time.time()
					print('| epoch {} | step {}/{} | loss {:.3f} | '
							'ngram/s {:.1f}'.format(
						epoch, step, num_steps, avg_loss,
						args.print_every * args.batch_size / (t1-t0)))
					t0 = time.time()

				if step % args.save_every == 0:
					with open(args.save_dir + '.latest.model', 'wb') as f:
						torch.save(model, f)
					# torch.save(model.state_dict(), args.save_dir + '.latest.dict')

			print('Evaluating on validation set...')
			val_loss = evaluate(val_data, model, criterion)
			print('-' * 89)
			print('| end of epoch {:3d} | time {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
				epoch,
				(time.time() - epoch_start_time),
				val_loss, np.exp(val_loss))
			)
			print('-' * 89)

			if not best_val_loss or val_loss < best_val_loss:
				with open(args.save_dir + '.best.model', 'wb') as f:
					torch.save(model, f)
				best_val_loss = val_loss

			scheduler.step(val_loss)

	except KeyboardInterrupt:
		print('-' * 89)
		print('Exiting from training early')

	write_losses(losses, args.log_dir)
	print('Evaluating on test set...')
	test_loss = evaluate(test_data, model, criterion)
	print('=' * 89)
	print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
		test_loss, np.exp(test_loss)))
	print('=' * 89)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Positional args
	parser.add_argument('name', type=str,
						help="Name of the model, e.g. 'wiki-char'")

	# Directory args
	parser.add_argument('--data-dir', type=str, default='~/data/wikitext/wikitext-2',
						help='directory for training data')
	parser.add_argument('--log-dir', type=str, default='log',
						help='directory to write out logs')
	parser.add_argument('--save-dir', type=str, default='models/model',
						help='save directory for model')
	parser.add_argument('--glove-dir', type=str, default='~/embeddings/glove',
						help='directory with glove embeddings if --use-glove')

	# Model args
	parser.add_argument('--use-chars', action='store_true',
						help='make a character language model')
	parser.add_argument('--order', type=int, default=5,
						help='order of the language model')
	parser.add_argument('--emb-dim', type=int, default=30,
						help='dimensionality of the word embeddings')
	parser.add_argument('--hidden-dims', type=str, default='100',
						help='dimension of hidden layers as comma separated string')
	parser.add_argument('--unk', type=str, default=UNK,
						help='UNK word if used')
	parser.add_argument('--start-word', type=str, default=SOS,
						help='start of sequence for word model')
	parser.add_argument('--start-char', type=str, default=SOS_CHAR,
						help='start of sequence for character model')
	parser.add_argument('--use-glove', action='store_true',
						help='use pretrained glove word embeddings')
	parser.add_argument('--tied', action='store_true',
						help='tie the word embedding and softmax weights')


	# Training args
	parser.add_argument('--batch-size', type=int, default=128,
						help='size of the minibatch')
	parser.add_argument('--lr', type=float, default=1e-3,
						help='learning rate for optimizer')
	parser.add_argument('--epochs', type=int, default=10,
						help='number of epochs')
	parser.add_argument('--print-every', type=int, default=10,
						help='how often to print during training progress')
	parser.add_argument('--save-every', type=int, default=100,
						help='how often to save during training progress')

	args = parser.parse_args()

	main(args)
