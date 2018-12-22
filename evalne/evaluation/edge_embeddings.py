#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# This code provides implementations of several methods for computing edge embeddings from given node feature vectors.

from __future__ import division

import numpy as np


def average(X, ebunch):
    r"""
    Compute the edge embeddings all node pairs (u,v) in ebunch as the average of the embeddings of u and v.

    Parameters
    ----------
    X : dict
        A dictionary where keys are nodes in the graph and values are the node embeddings.
        The keys are of type str and the values of type array.
    ebunch : iterable of node pairs
        The edges, as pairs (u,v), for which the embedding will be computed.

    Returns
    -------
    edge_embeds : matrix
        A Numpy matrix containing the edge embeddings in the same order as ebunch.
    """
    # edge_embeds = np.zeros((len(ebunch), len(X.values()[0])))
    edge_embeds = np.zeros((len(ebunch), len(X[list(X.keys())[0]])))
    i = 0
    for edge in ebunch:
        edge_embeds[i] = (X[str(edge[0])] + X[str(edge[1])]) / 2.0
        i += 1
    return edge_embeds


def hadamard(X, ebunch):
    r"""
    Compute the edge embeddings all node pairs (u,v) in ebunch as the hadamard distance of the embeddings of u and v.

    Parameters
    ----------
    X : dict
        A dictionary where keys are nodes in the graph and values are the node embeddings.
        The keys are of type str and the values of type array.
    ebunch : iterable of node pairs
        The edges, as pairs (u,v), for which the embedding will be computed.

    Returns
    -------
    edge_embeds : matrix
        A Numpy matrix containing the edge embeddings in the same order as ebunch.
    """
    edge_embeds = np.zeros((len(ebunch), len(X[list(X.keys())[0]])))
    i = 0
    for edge in ebunch:
        edge_embeds[i] = X[str(edge[0])] * X[str(edge[1])]
        i += 1
    return edge_embeds


def weighted_l1(X, ebunch):
    r"""
    Compute the edge embeddings all node pairs (u,v) in ebunch as the weighted l1 distance of the embeddings of u and v.

    Parameters
    ----------
    X : dict
        A dictionary where keys are nodes in the graph and values are the node embeddings.
        The keys are of type str and the values of type array.
    ebunch : iterable of node pairs
        The edges, as pairs (u,v), for which the embedding will be computed.

    Returns
    -------
    edge_embeds : matrix
        A Numpy matrix containing the edge embeddings in the same order as ebunch.
    """
    edge_embeds = np.zeros((len(ebunch), len(X[list(X.keys())[0]])))
    i = 0
    for edge in ebunch:
        edge_embeds[i] = np.abs(X[str(edge[0])] - X[str(edge[1])])
        i += 1
    return edge_embeds


def weighted_l2(X, ebunch):
    r"""
    Compute the edge embeddings all node pairs (u,v) in ebunch as the weighted l2 distance of the embeddings of u and v.

    Parameters
    ----------
    X : dict
        A dictionary where keys are nodes in the graph and values are the node embeddings.
        The keys are of type str and the values of type array.
    ebunch : iterable of node pairs
        The edges, as pairs (u,v), for which the embedding will be computed.

    Returns
    -------
    edge_embeds : matrix
        A Numpy matrix containing the edge embeddings in the same order as ebunch.
    """
    edge_embeds = np.zeros((len(ebunch), len(X[list(X.keys())[0]])))
    i = 0
    for edge in ebunch:
        edge_embeds[i] = np.power(X[str(edge[0])] - X[str(edge[1])], 2)
        i += 1
    return edge_embeds


def compute_edge_embeddings(X, ebunch, method='hadamard'):
    r"""
    Helper method to call any of the edge embedding methods using a simple parameter.

    Parameters
    ----------
    X : dict
        A dictionary where keys are nodes in the graph and values are the node embeddings.
        The keys are of type str and the values of type array.
    ebunch : iterable of node pairs
        The edges, as pairs (u,v), for which the embedding will be computed.
    method : string, optional
        The method to be used for computing the embeddings. Options are: average, hadamard, l1 or l2.
        Default is hadamard.

    Returns
    -------
    edge_embeds : matrix
        A Numpy matrix containing the edge embeddings in the same order as ebunch.
    """
    if method == 'hadamard':
        return hadamard(X, ebunch)
    elif method == 'average':
        return average(X, ebunch)
    elif method == 'weighted_l1':
        return weighted_l1(X, ebunch)
    elif method == 'weighted_l2':
        return weighted_l2(X, ebunch)
    else:
        raise ValueError("Unknown method!")
