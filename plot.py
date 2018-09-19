import os
from collections import Counter

import torch
import numpy as np

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, save, output_file
from bokeh.palettes import d3

from data import Corpus
from utils import model_data_checks


def plot(args):
    num_words = 1000

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    # Load model.
    with open(args.checkpoint, 'rb') as f:
        try:
            model = torch.load(f)
        except:
            # Convert the model to CPU if the model is serialized on GPU.
            model = torch.load(f, map_location='cpu')
    model.eval()
    embeddings = model.embedding.weight.data

    # Load data.
    data_dir = os.path.expanduser(args.data_dir)
    corpus = Corpus(data_dir, headers=args.no_headers, lower=args.lower, chars=args.use_chars)
    ntokens = len(corpus.dictionary.w2i)

    # Some checks to see if data and model are consistent.
    model_data_checks(model, corpus, args)

    # Prepare embeddings from num_words most common words.
    most_common_idxs = Counter(corpus.train).most_common(num_words)
    most_common_idxs, _ = zip(*most_common_idxs)  # Discard counts
    most_common_words = [corpus.dictionary.i2w[i] for i in most_common_idxs]
    idxs = torch.LongTensor(most_common_idxs)
    embeddings = embeddings[idxs, :].numpy()

    # Make bokeh plot.
    emb_scatter(embeddings, most_common_words, model_name=args.name)


def emb_scatter(data, names, model_name, perplexity=30.0, k=20):
    """t-SNE plot of embeddings and coloring with K-means clustering.

    Uses t-SNE with given perplexity to reduce the dimension of the
    vectors in data to 2, plots these in a bokeh 2d scatter plot,
    and colors them with k colors using K-means clustering of the
    originial vectors. The colored dots are tagged with labels from
    the list names.

    Args:
        data (np.Array): the word embeddings shape [num_vectors, embedding_dim]
        names (list): num_vectors words same order as data
        perplexity (float): perplexity for t-SNE
        N (int): number of clusters to find by K-means
    """
    # Find clusters with kmeans.
    print('Finding clusters...')
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    klabels = kmeans.labels_

    # Get a tsne fit.
    print('Fitting t-SNE...')
    tsne = TSNE(n_components=2, perplexity=perplexity)
    emb_tsne = tsne.fit_transform(data)

    # Plot the t-SNE of the embeddings with bokeh,
    # source: https://github.com/oxford-cs-deepnlp-2017/practical-1
    fig = figure(tools='pan,wheel_zoom,reset,save',
               toolbar_location='above',
               title='T-SNE for most common words')

    # Set colormap as a list.
    colormap = d3['Category20'][k]
    colors = [colormap[i] for i in klabels]

    source = ColumnDataSource(
        data=dict(
            x1=emb_tsne[:,0],
            x2=emb_tsne[:,1],
            names=names,
            colors=colors))

    fig.scatter(x='x1', y='x2', size=8, source=source, color='colors')

    labels = LabelSet(x='x1', y='x2', text='names', y_offset=6,
                      text_font_size='8pt', text_color='#555555',
                      source=source, text_align='center')
    fig.add_layout(labels)

    output_file(os.path.join('plots', f'{model_name}.tsne.html'))
    save(fig)
