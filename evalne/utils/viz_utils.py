#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 16/04/2019

# This file provides simple methods for embedding and graph visualization.
# T-SNE is applied to embeddings with more than two dimensions in order to plot them in a 2d space.

import os
import pandas as pd
import networkx as nx
import matplotlib as mpl
from sklearn.manifold import TSNE

if os.environ.get('DISPLAY', '') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt


def plot_emb2d(emb, colors=None, filename=None):
    """
    Generates a scatter plot of the given embeddings. Optional colors for the nodes can be provided as well as
    a filename to store the results. If dim of embeddings > 2, uses t-SNE to reduce it to 2.

    Parameters
    ----------
    emb : matrix
        A Numpy matrix containing the node or edge embeddings.
    colors : array, optional
        A Numpy array containing the colors of each node. Default is None.
    filename : string, optional
        A string indicating the path and name of the file where to store the scatter plot.
        If not provided the plot is shown on screen. Default is None.
    """
    print('Generating embedding scatter plot...')

    # Get the size of the embedding
    n, dim = emb.shape

    # If needed, reduce dimensionality to 2 using t-SNE
    if dim > 2:
        print("Embedding dimension is {}, using t-SNE to reduce it to 2.".format(dim))
        emb = TSNE(n_components=2).fit_transform(emb)

    # Plot embeddings
    if colors is None:
        plt.scatter(emb[:, 0], emb[:, 1], alpha=0.6)
    else:
        plt.scatter(emb[:, 0], emb[:, 1], alpha=0.6, c=colors)

    # Store or show the scatter plot
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300, format='pdf', bbox_inches='tight')


def plot_graph2d(G, emb=None, labels=None, colors=None, filename=None):
    """
    Plots the given graph in 2d. If the embeddings of nodes are provided, they are used to place the nodes on the
    2d plane. If dim of embeddings > 2, then its reduced to 2 using t-SNE. Optional labels and colors for the nodes
    can be provided, as well as a filename to store the results.

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    emb : matrix, optional
        A Numpy matrix containing the node embeddings. Default is None.
    labels : dict, optional
        A dictionary containing nodeIDs as keys and node labels as values. Default is None.
    colors : array, optional
        A Numpy array containing the colors of each graph node. Default is None.
    filename : string, optional
        A string indicating the path and name of the file where to store the scatter plot.
        If not provided the plot is showed on screen. Default is None.
    """
    print('Generating embedding visualization...')

    if emb is not None:
        # Get the size of the embedding
        n, dim = emb.shape

        # If needed, reduce dimensionality to 2 using t-SNE
        if dim > 2:
            print("Embedding dimension is {}, using t-SNE to reduce it to 2.".format(dim))
            emb = TSNE(n_components=2).fit_transform(emb)
    else:
        # If no embeddings provided, use the spring layout to position nodes
        emb = nx.spring_layout(G)

    # Plot nodes and edges
    nx.draw_networkx_nodes(G, emb, node_size=100, alpha=0.6, node_color=colors)
    nx.draw_networkx_edges(G, emb, width=1.0, arrows=False, alpha=0.1)

    # Plot the labels if provided
    if labels is not None:
        nx.draw_networkx_labels(G, emb, labels=labels, font_size=6)

    # Store or show the scatter plot
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300, format='pdf', bbox_inches='tight')


def plot_curve(filename, x, y, x_label, y_label, title=None):
    """
    Plots y coordinates against x coordinates as a line.

    Parameters
    ----------
    filename : string
        A file or filename where to store the plot.
    x : array_like
        The x coordinates of the plot.
    y : array_like
        The y coordinates of the plot.
    x_label : string
        The label of the x axis.
    y_label : string
        The label of the y axis.
    title : string or None, optional
        The title of the plot. Default is None (no title).
    """
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def parallel_coord(scoresheet, features, class_col='methods'):
    """
    Generates a parallel coordinate plot from the given Scoresheet object and the set of features specified.

    Parameters
    ----------
    scoresheet : evalne.Scoresheet
        A Scoresheet object containing the results of an evaluation.
    features : list
        A list of strings indicating the features to show in the plot (in addition to methods and networks).
        Accepted features are: 'auroc', 'average_precision', 'precision', 'recall',
        'fallout', 'miss', 'accuracy', 'f_score', `eval_time` and `edge_embed_method`.
    class_col : string, optional
        Indicates the class to highlight. Options are `methods` and `networks`. Default is `methods`.
    """
    # Get dfs per feature and stack them
    f_dfs = []
    for f in features:
        f_dfs.append(scoresheet.get_pandas_df(metric=f).stack())

    # Concatenate dfs and reset indexing
    df = pd.concat(f_dfs, axis=1, join="inner")
    df.reset_index(inplace=True)

    # Set correct column names
    new_names = ['methods_str', 'networks_str']
    new_names.extend(features)
    df.set_axis(new_names, axis=1, inplace=True)

    # Make networks and methods numerical
    df['methods_str'] = pd.Categorical(df['methods_str'])
    df['methods'] = df['methods_str'].cat.codes
    df['methods'] = (df['methods'] - df['methods'].min()) / (df['methods'].max() - df['methods'].min())
    df['networks_str'] = pd.Categorical(df['networks_str'])
    df['networks'] = df['networks_str'].cat.codes
    df['networks'] = (df['networks'] - df['networks'].min()) / (df['networks'].max() - df['networks'].min())
    if 'edge_embed_method' in features:
        df['edge_embed_method'] = pd.Categorical(df['edge_embed_method'])
        df['edge_embed_method'] = df['edge_embed_method'].cat.codes     # TODO: fix this
        df['edge_embed_method'] = (df['edge_embed_method'] - df['edge_embed_method'].min()) / \
                                  (df['edge_embed_method'].max() - df['edge_embed_method'].min())
    if 'eval_time' in features:
        df['eval_time'] = (df['eval_time'] - df['eval_time'].min()) / \
                                  (df['eval_time'].max() - df['eval_time'].min())

    # Select all numerical cols
    num = ['methods', 'networks']
    num.extend(features)

    # Generate the plot
    pd.plotting.parallel_coordinates(df[num], class_col)
    ax = plt.gca()

    # Add labels
    if class_col == 'methods':
        for i, (label, val) in df.ix[:, ['networks_str', 'networks']].drop_duplicates().iterrows():
            ax.annotate(label, xy=(0, val), ha='left', va='center')
        aux = df.ix[:, ['methods_str', 'methods']].drop_duplicates()
        plt.legend(aux['methods_str'])
    elif class_col == 'networks':
        for i, (label, val) in df.ix[:, ['methods_str', 'methods']].drop_duplicates().iterrows():
            ax.annotate(label, xy=(0, val), ha='left', va='center')
        aux = df.ix[:, ['networks_str', 'networks']].drop_duplicates()
        plt.legend(aux['networks_str'])

    # Some changes to plot axis
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    plt.show()
