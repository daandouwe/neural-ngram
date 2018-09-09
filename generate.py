import torch
from torch.autograd import Variable

from data import Corpus
from utils import SOS, EOS


def generate(args):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    if args.temperature < 1e-3:
        parser.error("--temperature has to be greater or equal 1e-3")

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f)
    model.eval()

    corpus = Corpus(args.data_dir, chars=args.use_chars)
    ntokens = len(corpus.dictionary)
    start = 'this story starts with the '
    ids = [corpus.dictionary.w2i[w] for w in start.split()]
    input = Variable(torch.LongTensor(ids).unsqueeze(0))

    with open(args.outf, 'w') as outf:
        outf.write(start)

        for i in range(args.num_samples):
            output = model(input)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_idx = word_idx.data[0]
            word = corpus.dictionary.i2w[word_idx]

            ids.append(word_idx)
            input = Variable(torch.LongTensor(ids[-args.order:]).unsqueeze(0))

            if word in (SOS, EOS):
                outf.write('\n')
            else:
                outf.write(word + ' ')

            if i % 100 == 0:
                print('| Generated {}/{} words'.format(i, args.num_samples))
