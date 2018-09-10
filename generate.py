import os

import torch
from torch.autograd import Variable

from data import Corpus
from utils import UNK, SOS, EOS, UNK_CHAR, SOS_CHAR, EOS_CHAR, model_data_checks


def generate(args):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    if args.temperature < 1e-3:
        parser.error("--temperature has to be greater or equal 1e-3")

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f)
    model.eval()

    sos = SOS_CHAR if args.use_chars else SOS
    eos = EOS_CHAR if args.use_chars else EOS
    unk = UNK_CHAR if args.use_chars else UNK

    data_dir = os.path.expanduser(args.data_dir)
    corpus = Corpus(data_dir, headers=args.no_headers, lower=args.lower, chars=args.use_chars)
    ntokens = len(corpus.dictionary)

    model_data_checks(model, corpus, args)

    if args.start:
        start = list(args.start) if args.use_chars else args.start.split()
        input = start = [word.lower() for word in start] if args.lower else start
        if len(input) < model.order:
            input = (model.order - len(input))*[sos] + input
        elif len(input) > model.order:
            input = input[-model.order:]
    else:
        start = input = [sos]*model.order
    input = [word if word in corpus.dictionary.w2i else unk for word in input]
    ids = [corpus.dictionary.w2i[word] for word in input]
    input = Variable(torch.LongTensor(ids).unsqueeze(0))


    glue = '' if args.use_chars else ' '
    with open(args.outf, 'w') as outf:
        if args.start:
            outf.write(glue.join(start) + glue)
        for i in range(args.num_samples):
            output = model(input)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            if args.no_unk:
                word_weights[corpus.dictionary.w2i[unk]] = 0
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_idx = word_idx.data[0]
            word = corpus.dictionary.i2w[word_idx]

            ids.append(word_idx)
            input = Variable(torch.LongTensor(ids[-model.order:]).unsqueeze(0))

            if word in (sos, eos):
                outf.write('\n')
            else:
                outf.write(word + glue)

            if i % 100 == 0:
                print('| Generated {}/{} words'.format(i, args.num_samples))
