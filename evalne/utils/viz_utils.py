#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 16/04/2019

import os
import networkx as nx
from sklearn.manifold import TSNE
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt


def plot_emb2d(emb, colors=None, filename=None):
    r"""
    Generates scatter plot of the given embeddings. Optional colors for the nodes can be provided as well as
    a filename to store the results. If dim of embeddings > 2, uses t-SNE to reduce it to 2.

    Parameters
    ----------
    emb : matrix
        A Numpy matrix containing the node or edge embeddings.
    colors : array, optional
        A Numpy array containing the colors of each graph node. Default is None.
    filename : basestring, optional
        A string indicating the path and name of the file where to store the scatter plot.
        If not provided plot is showed on screen. Default is None.
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
    r"""
    Plots the given graph in 2d. If the embeddings of nodes are provided, they are used to place the nodes on the
    2d plane. If dim of embeddings > 2, then its reduced to 2 using t-SNE. Optional labels and colors for the nodes
    can be provided, as well as a filename to store the results.

    Parameters
    ----------
    G : graph
        A NetworkX graph
    emb : matrix, optional
        A Numpy matrix containing the node embeddings. Default is None.
    labels : dict, optional
        A dictionary containing as keys the nodeIDs and as values the node labels. Default is None.
    colors : array, optional
        A Numpy array containing the colors of each graph node. Default is None.
    filename : basestring, optional
        A string indicating the path and name of the file where to store the scatter plot.
        If not provided plot is showed on screen. Default is None.
    """

    print('Generating embedding vizualization...')

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
    nx.draw_networkx_nodes(G, emb, width=0.1, node_size=100, arrows=False, alpha=0.6, node_color=colors)
    nx.draw_networkx_edges(G, emb, width=1.0, alpha=0.1)

    # Plot the labels if provided
    if labels is not None:
        nx.draw_networkx_labels(G, emb, labels=labels, font_size=6)

    # Store or show the scatter plot
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300, format='pdf', bbox_inches='tight')

