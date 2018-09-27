#!/usr/bin/env python
import os
import argparse
import time
from collections import Counter

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm

from data import Corpus
from model import NeuralNgram
from generate import generate
from plot import plot
from softmax import ApproximateLoss
from utils import print_args, write_losses, list_hidden_dims, model_data_checks, normalize, clock_time


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
	model.eval()
	total_loss = 0
	n_steps = data.size(0) - model.order - 1
	for i in tqdm(range(n_steps)):
		x, y = get_batch(data, i, model.order)
		out = model(x)
		loss = criterion(out, y)
		total_loss += loss.data.data[0]
	return total_loss / n_steps


def train(args):
	cuda = torch.cuda.is_available()

	# Set seed for reproducibility.
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	data_dir = os.path.expanduser(args.data_dir)
	corpus = Corpus(data_dir, headers=args.no_headers, lower=args.lower, chars=args.use_chars)
	train_data = batchify(corpus.train, args.batch_size)
	val_data = batchify(corpus.valid, args.batch_size)
	test_data = batchify(corpus.test, args.batch_size)
	if cuda:
		train_data, val_data, test_data = train_data.cuda(), val_data.cuda(), test_data.cuda()

	# Logging
	print_args(args)
	print('Using cuda: {}'.format(cuda))
	print('Size of training set: {:,} tokens'.format(np.prod(train_data.size())))
	print('Size of validation set: {:,} tokens'.format(np.prod(val_data.size())))
	print('Size of test set: {:,} tokens'.format(np.prod(test_data.size())))
	print('Vocabulary size: {:,}'.format(corpus.vocab_size))
	print('Example data:')
	for k in range(100, 107):
		x = [corpus.dictionary.i2w[i] for i in train_data[k:args.order+k, 0]]
		y = [corpus.dictionary.i2w[train_data[k+args.order, 0]]]
		print(x, y)

	# Initialize model
	if args.resume:
		print(f'Resume training with model {args.checkpoint}...')
		with open(args.checkpoint, 'rb') as f:
			model = torch.load(f)
		model_data_checks(model, corpus, args)
	else:
		hidden_dims = list_hidden_dims(args.hidden_dims)
		model = NeuralNgram(
			order=args.order,
			emb_dim=args.emb_dim,
			vocab_size=corpus.vocab_size,
			hidden_dims=hidden_dims,
			dropout=args.dropout)
		if args.use_glove:
			print('Loading GloVe vectors...')
			model.load_glove(args.glove_dir, i2w=corpus.dictionary.i2w)
		if args.tied:
			print('Tying weights...')
			model.tie_weights()
		if cuda:
			model.cuda()

	parameters = [param for param in model.parameters() if param.requires_grad]
	optimizer = torch.optim.SGD(parameters, lr=args.lr)
	if args.softmax_approx:
		if args.unigram_proposal:
			unigram = normalize(Counter(corpus.train.numpy()))
			unigram = np.array([unigram[i] for i in range(len(unigram))])
		else:
			unigram = None
		criterion = ApproximateLoss(
			vocab_size=len(corpus.dictionary), method='importance', unigram=unigram)
	else:
		criterion = nn.CrossEntropyLoss()

	# Training
	print('Training...')
	losses = dict(train=[], val=[])

	lr = args.lr
	best_val_loss = None

	num_steps = train_data.size(0) - args.order - 1
	batch_order = np.arange(num_steps)

	t0 = time.time()
	try:
		for epoch in range(1, args.epochs+1):
			model.train()
			epoch_start_time = time.time()
			np.random.shuffle(batch_order)
			for step in range(1, num_steps+1):
				idx = batch_order[step-1]
				x, y = get_batch(train_data, idx, args.order)

				# Forward pass
				logits = model(x)
				loss = criterion(logits, y)

				if args.debug:
					# Debugging softmax approximation.
					xe = nn.CrossEntropyLoss()
					true_loss = xe(logits, y)
					print('approx {:>3.2f}, true {:>3.2f}, diff {:>3.4f}'.format(
						loss.data[0], true_loss.data[0], true_loss.data[0] - loss.data[0]))

				# Update parameters
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# Save loss.
				losses['train'].append(loss.cpu().data[0])

				if step % args.print_every == 0:
					avg_loss = sum(losses['train'][-args.print_every:]) / args.print_every
					t1 = time.time()
					steps_per_second = args.print_every / (t1 - t0)
					print('| epoch {} | step {}/{} | loss {:.4f} | lr {:.3f} | '
							'ngrams/sec {:.1f} | eta {}h{}m{}s'.format(
								epoch, step, num_steps, avg_loss, lr,
								steps_per_second * args.batch_size,
								*clock_time((num_steps - step) / steps_per_second)))
					t0 = time.time()

				if step % args.save_every == 0:
					modelpath = os.path.join(args.save_dir, f'{args.name}.latest.pt')
					with open(modelpath, 'wb') as f:
						torch.save(model, f)

			print('Evaluating on validation set...')
			val_loss = evaluate(val_data, model, criterion)
			losses['val'].append(val_loss)
			print('-' * 89)
			print('| end of epoch {:3d} | time {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
				epoch, (time.time() - epoch_start_time), val_loss, np.exp(val_loss)))
			print('-' * 89)

			if not best_val_loss or val_loss < best_val_loss:
				modelpath = os.path.join(args.save_dir, f'{args.name}.best.pt')
				with open(modelpath, 'wb') as f:
					torch.save(model, f)
				best_val_loss = val_loss
			else:
				# Anneal the learning rate if no improvement has been seen in the validation dataset.
				lr /= 4.0
				optimizer = torch.optim.SGD(parameters, lr=lr)
	except KeyboardInterrupt:
		print('-' * 89)
		print('Exiting from training early')

	write_losses(losses['train'], args.log_dir, name='train-losses')
	write_losses(losses['val'], args.log_dir, name='val-losses')

	print('Evaluating on test set...')
	test_loss = evaluate(test_data, model, criterion)
	print('=' * 89)
	print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
		test_loss, np.exp(test_loss)))
	print('=' * 89)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('mode', choices=['train', 'generate', 'plot'])

    # Debug mode.
	parser.add_argument('-d', '--debug', action='store_true')

	# Dir args
	parser.add_argument('--data-dir', type=str, default='data/wikitext-2',
						help='directory for training data')
	parser.add_argument('--log-dir', type=str, default='log',
						help='directory to write out logs')
	parser.add_argument('--save-dir', type=str, default='models',
						help='save directory for model')
	parser.add_argument('--glove-dir', type=str, default='~/embeddings/glove',
						help='directory with glove embeddings if --use-glove')

	# Data args
	parser.add_argument('--use-chars', action='store_true',
						help='make a character-level language model')
	parser.add_argument('--lower', action='store_true',
						help='lowercase all training data')
	parser.add_argument('--no-headers', action='store_false',
						help='remove headers from wikitext data')

	# Model args
	parser.add_argument('--name', type=str, default='wiki',
						help='name of the model, e.g. `wiki-char`')
	parser.add_argument('--order', type=int, default=5,
						help='order of the language model')
	parser.add_argument('--emb-dim', type=int, default=50,
						help='dimensionality of the word embeddings')
	parser.add_argument('--hidden-dims', type=str, default='100',
						help='dimension of hidden layers as comma separated string')
	parser.add_argument('--use-glove', action='store_true',
						help='use pretrained glove word embeddings')
	parser.add_argument('--tied', action='store_true',
						help='tie the word embedding and softmax weights')

	# Training args
	parser.add_argument('--batch-size', type=int, default=32,
						help='size of the minibatch')
	parser.add_argument('--lr', type=float, default=1.0,
						help='initial learning rate for optimizer')
	parser.add_argument('--epochs', type=int, default=10,
						help='number of epochs')
	parser.add_argument('--dropout', type=float, default=0.0,
						help='dropout (embedding and mlp)')
	parser.add_argument('--softmax-approx', action='store_true',
						help='use softmax approximation (complementary sum sampling)')
	parser.add_argument('--unigram-proposal', action='store_true',
						help='use unigram proposal distribution in softmax approx')
	parser.add_argument('--seed', type=int, default=42,
						help='random seed')
	parser.add_argument('--print-every', type=int, default=1000,
						help='how often to print during training progress')
	parser.add_argument('--save-every', type=int, default=10000,
						help='how often to save during training progress')
	parser.add_argument('--resume', action='store_true',
						help='resume training model saved in --checkpoint')

	# Generation args
	parser.add_argument('--checkpoint', type=str, default='models/wiki.latest.pt',
						help='model checkpoint to use')
	parser.add_argument('--outf', type=str, default='generated.txt',
						help='output file for generated text')
	parser.add_argument('--start', default=None,
						help='start of generated text')
	parser.add_argument('--num-samples', type=int, default='1000',
						help='number of words to generate')
	parser.add_argument('--temperature', type=float, default=1.0,
						help='temperature - higher will increase sample diversity')
	parser.add_argument('--no-unk', action='store_true',
						help='avoid generating unk')
	parser.add_argument('--no-sos', action='store_true',
						help='print sos and eos tokens')

	args = parser.parse_args()

	if args.mode == 'train':
		train(args)
	if args.mode == 'generate':
		generate(args)
	if args.mode == 'plot':
		plot(args)
