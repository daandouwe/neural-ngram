import os


UNK = '<unk>'
SOS = '<sos>'
EOS = '<eos>'

UNK_CHAR = '_'
SOS_CHAR = '~'
EOS_CHAR = '~'


def clock_time(s):
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return int(h), int(m), int(s)


def write_losses(losses, logdir, name='losses'):
	name = name + '.csv' if not name.endswith('.csv') else name
	logpath = os.path.join(logdir, name)
	with open(logpath, 'w') as f:
		print('loss', file=f)
		for loss in losses:
			print(loss, file=f)


def print_args(args):
	"""Prints all entries in args variable."""
	for key, value in vars(args).items():
		print(f'{key} : {value}')
	print()


def list_hidden_dims(hidden_dims):
    """Turn hidden dims from string into list: from '100,100' to [100, 100]"""
    assert isinstance(hidden_dims, str)
    if hidden_dims:
    	return [int(d) for d in hidden_dims.split(",")]


def model_data_checks(model, corpus, args):
	"""Check if model and data are consistent given args.

	Useful when loading a model. E.g. when resuming training,
	for generation, and for plotting.
	"""
	embeddings = model.embedding.weight
	ntokens = len(corpus.dictionary.w2i)
	if args.use_glove:
		assert not embeddings.requires_grad, 'embeddings were trained, while using glove.'
	else:
		assert embeddings.requires_grad, 'embeddings were not trained, while not using glove.'
	message = 'inconsistent sizes ntokens {:,} and embeddings {:,}. Not using the same data arguments?'.format(
		ntokens, embeddings.size(0))
	assert ntokens == embeddings.size(0), message


def normalize(counter):
	total = sum(counter.values())
	return dict((key, count/total) for key, count in counter.items())
