#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# This file provides implementations of several operators for computing node-pair embeddings from node feature vectors.

from __future__ import division

import numpy as np


def average(X, ebunch):
    """
    Computes the embedding of each node pair (u, v) in ebunch as the element-wise average of the embeddings of
    nodes u and v.

    Parameters
    ----------
    X : dict
        A dictionary of {`nodeID`: embed_vect, `nodeID`: embed_vect, ...}. Dictionary keys are expected to be of type
        string and values array_like.
    ebunch : iterable
        An iterable of node pairs (u,v) for which the embeddings must be computed.

    Returns
    -------
    emb : ndarray
        A column vector containing node-pair embeddings as rows. In the same order as ebunch.

    Notes
    -----
    Formally, if we use x(u) to denote the embedding corresponding to node u and x(v) to denote the embedding
    corresponding to node v, and if we use i to refer to the ith position in these vectors, then, the embedding of the
    pair (u, v) can be computed element-wise as: :math:`x(u, v)_i = \\frac{x(u)_i + x(v)_i}{2}`.
    Also note that all nodeID's in ebunch must exist in X, otherwise, the method will fail.

    Examples
    --------
    Simple example of function use and input parameters:

    >>> X = {'1': np.array([0, 0, 0, 0]), '2': np.array([2, 2, 2, 2]), '3': np.array([1, 1, -1, -1])}
    >>> ebunch = ((2, 1), (1, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2))
    >>> average(X, ebunch)
    array([[ 1. ,  1. ,  1. ,  1. ],
           [ 0. ,  0. ,  0. ,  0. ],
           [ 2. ,  2. ,  2. ,  2. ],
           [ 0.5,  0.5, -0.5, -0.5],
           [ 0.5,  0.5, -0.5, -0.5],
           [ 1.5,  1.5,  0.5,  0.5],
           [ 1.5,  1.5,  0.5,  0.5]])

    """
    edge_embeds = np.zeros((len(ebunch), len(X[list(X.keys())[0]])))
    i = 0
    for edge in ebunch:
        edge_embeds[i] = (X[str(edge[0])] + X[str(edge[1])]) / 2.0
        i += 1
    return edge_embeds


def hadamard(X, ebunch):
    """
    Computes the embedding of each node pair (u, v) in ebunch as the element-wise product between the
    embeddings of nodes u and v.

    Parameters
    ----------
    X : dict
        A dictionary of {`nodeID`: embed_vect, `nodeID`: embed_vect, ...}. Dictionary keys are expected to be of type
        string and values array_like.
    ebunch : iterable
        An iterable of node pairs (u,v) for which the embeddings must be computed.

    Returns
    -------
    emb : ndarray
        A column vector containing node-pair embeddings as rows. In the same order as ebunch.

    Notes
    -----
    Formally, if we use x(u) to denote the embedding corresponding to node u and x(v) to denote the embedding
    corresponding to node v, and if we use i to refer to the ith position in these vectors, then, the embedding of the
    pair (u, v) can be computed element-wise as: :math:`x(u, v)_i = x(u)_i * x(v)_i`.
    Also note that all nodeID's in ebunch must exist in X, otherwise, the method will fail.

    Examples
    --------
    Simple example of function use and input parameters:

    >>> X = {'1': np.array([0, 0, 0, 0]), '2': np.array([2, 2, 2, 2]), '3': np.array([1, 1, -1, -1])}
    >>> ebunch = ((2, 1), (1, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2))
    >>> hadamard(X, ebunch)
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 4.,  4.,  4.,  4.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 2.,  2., -2., -2.],
           [ 2.,  2., -2., -2.]])

    """
    edge_embeds = np.zeros((len(ebunch), len(X[list(X.keys())[0]])))
    i = 0
    for edge in ebunch:
        edge_embeds[i] = X[str(edge[0])] * X[str(edge[1])]
        i += 1
    return edge_embeds


def weighted_l1(X, ebunch):
    """
    Computes the embedding of each node pair (u, v) in ebunch as the element-wise weighted L1 distance between the
    embeddings of nodes u and v.

    Parameters
    ----------
    X : dict
        A dictionary of {`nodeID`: embed_vect, `nodeID`: embed_vect, ...}. Dictionary keys are expected to be of type
        string and values array_like.
    ebunch : iterable
        An iterable of node pairs (u,v) for which the embeddings must be computed.

    Returns
    -------
    emb : ndarray
        A column vector containing node-pair embeddings as rows. In the same order as ebunch.

    Notes
    -----
    Formally, if we use x(u) to denote the embedding corresponding to node u and x(v) to denote the embedding
    corresponding to node v, and if we use i to refer to the ith position in these vectors, then, the embedding of the
    pair (u, v) can be computed element-wise as: :math:`x(u, v)_i = |x(u)_i - x(v)_i|`.
    Also note that all nodeID's in ebunch must exist in X, otherwise, the method will fail.

    Examples
    --------
    Simple example of function use and input parameters:

    >>> X = {'1': np.array([0, 0, 0, 0]), '2': np.array([2, 2, 2, 2]), '3': np.array([1, 1, -1, -1])}
    >>> ebunch = ((2, 1), (1, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2))
    >>> weighted_l1(X, ebunch)
    array([[2., 2., 2., 2.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 3., 3.],
           [1., 1., 3., 3.]])

    """
    edge_embeds = np.zeros((len(ebunch), len(X[list(X.keys())[0]])))
    i = 0
    for edge in ebunch:
        edge_embeds[i] = np.abs(X[str(edge[0])] - X[str(edge[1])])
        i += 1
    return edge_embeds


def weighted_l2(X, ebunch):
    """
    Computes the embedding of each node pair (u, v) in ebunch as the element-wise weighted L2 distance between the
    embeddings of nodes u and v.

    Parameters
    ----------
    X : dict
        A dictionary of {`nodeID`: embed_vect, `nodeID`: embed_vect, ...}. Dictionary keys are expected to be of type
        string and values array_like.
    ebunch : iterable
        An iterable of node pairs (u,v) for which the embeddings must be computed.

    Returns
    -------
    emb : ndarray
        A column vector containing node-pair embeddings as rows. In the same order as ebunch.

    Notes
    -----
    Formally, if we use x(u) to denote the embedding corresponding to node u and x(v) to denote the embedding
    corresponding to node v, and if we use i to refer to the ith position in these vectors, then, the embedding of the
    pair (u, v) can be computed element-wise as: :math:`x(u, v)_i = (x(u)_i - x(v)_i)^2`.
    Also note that all nodeID's in ebunch must exist in X, otherwise, the method will fail.

    Examples
    --------
    Simple example of function use and input parameters:

    >>> X = {'1': np.array([0, 0, 0, 0]), '2': np.array([2, 2, 2, 2]), '3': np.array([1, 1, -1, -1])}
    >>> ebunch = ((2, 1), (1, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2))
    >>> weighted_l2(X, ebunch)
    array([[4., 4., 4., 4.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 9., 9.],
           [1., 1., 9., 9.]])

    """
    edge_embeds = np.zeros((len(ebunch), len(X[list(X.keys())[0]])))
    i = 0
    for edge in ebunch:
        edge_embeds[i] = np.power(X[str(edge[0])] - X[str(edge[1])], 2)
        i += 1
    return edge_embeds


def compute_edge_embeddings(X, ebunch, method='hadamard'):
    """
    Computes the embedding of each node pair (u, v) in ebunch as an element-wise operation on the embeddings of the end
    nodes u and v. The operator used is determined by the `method` parameter.

    Parameters
    ----------
    X : dict
        A dictionary of {`nodeID`: embed_vect, `nodeID`: embed_vect, ...}. Dictionary keys are expected to be of type
        string and values array_like.
    ebunch : iterable
        An iterable of node pairs (u,v) for which the embeddings must be computed.
    method : string, optional
        The operator to be used for computing the node-pair embeddings. Options are: `average`, `hadamard`,
        `weighted_l1` or `weighted_l2`. Default is `hadamard`.

    Returns
    -------
    emb : ndarray
        A column vector containing node-pair embeddings as rows. In the same order as ebunch.

    Examples
    --------
    Simple example of function use and input parameters:

    >>> X = {'1': np.array([0, 0, 0, 0]), '2': np.array([2, 2, 2, 2]), '3': np.array([1, 1, -1, -1])}
    >>> ebunch = ((2, 1), (1, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2))
    >>> compute_edge_embeddings(X, ebunch, 'average')
    array([[ 1. ,  1. ,  1. ,  1. ],
           [ 0. ,  0. ,  0. ,  0. ],
           [ 2. ,  2. ,  2. ,  2. ],
           [ 0.5,  0.5, -0.5, -0.5],
           [ 0.5,  0.5, -0.5, -0.5],
           [ 1.5,  1.5,  0.5,  0.5],
           [ 1.5,  1.5,  0.5,  0.5]])

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
