import os


UNK = '<unk>'
SOS = '<sos>'
EOS = '<eos>'
SOS_CHAR = '~'
EOS_CHAR = '~'


def write_losses(losses, logdir, name='losses'):
	name = name + '.csv' if not name.endswith('.csv') else name
	logpath = os.path.join(logdir, name)
	with open(logpath, 'w') as f:
		print('step,loss', file=f)
		for i, loss in enumerate(losses):
			print(','.join((i, loss)), file=f)


def print_args(args):
	"""Prints all entries in args variable."""
	for key, value in vars(args).items():
		print('{} : {}'.format(key, value))
	print()


def list_hidden_dims(hidden_dims):
    """Turn hidden dims from string into list: from '100,100' to [100, 100]"""
    assert isinstance(hidden_dims, str)
    if hidden_dims:
    	return [int(d) for d in hidden_dims.split(",")]


def load_glove(glove_path, dictionary):
	return None
