#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# This file provides implementations of a variety heuristics for computing node-pair similarities. These heuristics are
# commonly used as baselines for network embedding evaluation. All functions of the networkx.link_prediction module are
# reimplemented here and extended to directed graphs (following https://surface.syr.edu/etd/355/).
# MultiGraphs and weighted graphs are not supported.

# TODO: the apply_prediction method should probably return a numpy array (like edge_embeddings does) rather than a list.

from __future__ import division

import math
import random

import networkx as nx
import numpy as np

__all__ = ['common_neighbours',
           'jaccard_coefficient',
           'cosine_similarity',
           'lhn_index',
           'topological_overlap',
           'adamic_adar_index',
           'resource_allocation_index',
           'preferential_attachment',
           'random_prediction',
           'all_baselines']


def _apply_prediction(G, func, ebunch=None):
    r"""
    Applies the given function to each node-pair in ebunch, if this is provided, otherwise, it applies the function to
    all edges in G.

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    func : func
        A function on two inputs each being a node in the graph. Can return anything, but it should return a value
        representing the likelihood of a "link" between the two nodes.
    ebunch : iterable, optional
        An iterable of node pairs. If None, all edges in G will be used. Default is None.

    Returns
    -------
    sim : list
        A list of node-pair similarities in the same order as ebunch.

    """
    if ebunch is None:
        ebunch = list(G.edges)
    return list(map(lambda e: func(e[0], e[1]), ebunch))


def common_neighbours(G, ebunch=None, neighbourhood='in'):
    r"""
    Computes the common neighbours similarity between all node pairs in ebunch; or all nodes in G, if ebunch is None.
    Can be computed for directed and undirected graphs (see Notes for exact definitions).

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    ebunch : iterable, optional
        An iterable of node pairs. If None, all edges in G will be used. Default is None.
    neighbourhood : string, optional
        For directed graphs only. Determines if the in or the out-neighbourhood of nodes should be used.
        Default is 'in'.

    Returns
    -------
    sim : list
        A list of node-pair similarities in the same order as ebunch.

    Raises
    ------
    ValueError
        If G is directed and neighbourhood is not one of 'in' or 'out'.

    Notes
    -----
    For undirected graphs the common neighbours similarity of nodes 'u' and 'v' is defined as:

    .. math:: |\Gamma(u) \cap \Gamma(v)|

    For directed graphs we can consider either the in or the out-neighbourhoods, respectively:

    .. math::

        |\Gamma_i(u) \cap \Gamma_i(v)|

        |\Gamma_o(u) \cap \Gamma_o(v)|

    """
    def predict(u, v):
        return len(set(G[u]) & set(G[v]))

    def predict_in(u, v):
        su = set(map(lambda e: e[0], G.in_edges(u)))
        sv = set(map(lambda e: e[0], G.in_edges(v)))
        return len(su & sv)

    def predict_out(u, v):
        su = set(map(lambda e: e[1], G.out_edges(u)))
        sv = set(map(lambda e: e[1], G.out_edges(v)))
        return len(su & sv)

    # Select the appropriate function and return the results
    if G.is_directed():
        if neighbourhood == 'in':
            return _apply_prediction(G, predict_in, ebunch)
        elif neighbourhood == 'out':
            return _apply_prediction(G, predict_out, ebunch)
        else:
            raise ValueError("Unknown parameter value.")
    return _apply_prediction(G, predict, ebunch)


def jaccard_coefficient(G, ebunch=None, neighbourhood='in'):
    r"""
    Computes the Jaccard coefficient between all node pairs in ebunch; or all nodes in G, if ebunch is None.
    Can be computed for directed and undirected graphs (see Notes for exact definitions).

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    ebunch : iterable, optional
        An iterable of node pairs. If None, all edges in G will be used. Default is None.
    neighbourhood : string, optional
        For directed graphs only. Determines if the in or the out-neighbourhood of nodes should be used.
        Default is 'in'.

    Returns
    -------
    sim : list
        A list of node-pair similarities in the same order as ebunch.

    Raises
    ------
    ValueError
        If G is directed and neighbourhood is not one of 'in' or 'out'.

    Notes
    -----
    For undirected graphs the Jaccard coefficient of nodes 'u' and 'v' is defined as:

    .. math:: |\Gamma(u) \cap \Gamma(v)| / |\Gamma(u) \cup \Gamma(v)|

    For directed graphs we can consider either the in or the out-neighbourhoods, respectively:

    .. math::

        \frac{|\Gamma_i(u) \cap \Gamma_i(v)|}{|\Gamma_i(u) \cup \Gamma_i(v)|}

        \frac{|\Gamma_o(u) \cap \Gamma_o(v)|}{|\Gamma_o(u) \cup \Gamma_o(v)|}

    """
    def predict(u, v):
        union_size = len(set(G[u]) | set(G[v]))
        if union_size == 0:
            return 0
        return len(list(nx.common_neighbors(G, u, v))) / union_size

    def predict_in(u, v):
        su = set(map(lambda e: e[0], G.in_edges(u)))
        sv = set(map(lambda e: e[0], G.in_edges(v)))
        union_size = len(su | sv)
        if union_size == 0:
            return 0
        return len(su & sv) / union_size

    def predict_out(u, v):
        su = set(map(lambda e: e[1], G.out_edges(u)))
        sv = set(map(lambda e: e[1], G.out_edges(v)))
        union_size = len(su | sv)
        if union_size == 0:
            return 0
        return len(su & sv) / union_size

    # Select the appropriate function and return the results
    if G.is_directed():
        if neighbourhood == 'in':
            return _apply_prediction(G, predict_in, ebunch)
        elif neighbourhood == 'out':
            return _apply_prediction(G, predict_out, ebunch)
        else:
            raise ValueError("Unknown parameter value.")
    return _apply_prediction(G, predict, ebunch)


def cosine_similarity(G, ebunch=None, neighbourhood='in'):
    r"""
    Computes the cosine similarity between all node pairs in ebunch; or all nodes in G, if ebunch is None.
    Can be computed for directed and undirected graphs (see Notes for exact definitions).

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    ebunch : iterable, optional
        An iterable of node pairs. If None, all edges in G will be used. Default is None.
    neighbourhood : string, optional
        For directed graphs only. Determines if the in or the out-neighbourhood of nodes should be used.
        Default is 'in'.

    Returns
    -------
    sim : list
        A list of node-pair similarities in the same order as ebunch.

    Raises
    ------
    ValueError
        If G is directed and neighbourhood is not one of 'in' or 'out'.

    Notes
    -----
    For undirected graphs the cosine similarity of nodes 'u' and 'v' is defined as:

    .. math:: \frac{|\Gamma(u) \cap \Gamma(v)|}{\sqrt{|\Gamma(u)| |\Gamma(v)|}}

    For directed graphs we can consider either the in or the out-neighbourhoods, respectively:

    .. math::

        \frac{|\Gamma_i(u) \cap \Gamma_i(v)|}{\sqrt{|\Gamma_i(u)| |\Gamma_i(v)|}}

        \frac{|\Gamma_o(u) \cap \Gamma_o(v)|}{\sqrt{|\Gamma_o(u)| |\Gamma_o(v)|}}

    """
    def predict(u, v):
        den = math.sqrt(len(set(G[u])) * len(set(G[v])))
        if den == 0:
            return 0
        return len(list(nx.common_neighbors(G, u, v))) / den

    def predict_in(u, v):
        su = set(map(lambda e: e[0], G.in_edges(u)))
        sv = set(map(lambda e: e[0], G.in_edges(v)))
        den = math.sqrt(len(su) * len(sv))
        if den == 0:
            return 0
        return len(su & sv) / den

    def predict_out(u, v):
        su = set(map(lambda e: e[1], G.out_edges(u)))
        sv = set(map(lambda e: e[1], G.out_edges(v)))
        den = math.sqrt(len(su) * len(sv))
        if den == 0:
            return 0
        return len(su & sv) / den

    # Select the appropriate function and return the results
    if G.is_directed():
        if neighbourhood == 'in':
            return _apply_prediction(G, predict_in, ebunch)
        elif neighbourhood == 'out':
            return _apply_prediction(G, predict_out, ebunch)
        else:
            raise ValueError("Unknown parameter value.")
    return _apply_prediction(G, predict, ebunch)


def lhn_index(G, ebunch=None, neighbourhood='in'):
    r"""
    Computes the Leicht-Holme-Newman index [1]_ between all node pairs in ebunch; or all nodes in G, if ebunch is None.
    Can be computed for directed and undirected graphs (see Notes for exact definitions).

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    ebunch : iterable, optional
        An iterable of node pairs. If None, all edges in G will be used. Default is None.
    neighbourhood : string, optional
        For directed graphs only. Determines if the in or the out-neighbourhood of nodes should be used.
        Default is 'in'.

    Returns
    -------
    sim : list
        A list of node-pair similarities in the same order as ebunch.

    Raises
    ------
    ValueError
        If G is directed and neighbourhood is not one of 'in' or 'out'.

    Notes
    -----
    For undirected graphs the Leicht-Holme-Newman index of nodes 'u' and 'v' is defined as:

    .. math:: \frac{|\Gamma(u) \cap \Gamma(v)|}{|\Gamma(u)| |\Gamma(v)|}

    For directed graphs we can consider either the in or the out-neighbourhoods, respectively:

    .. math::

        \frac{|\Gamma_i(u) \cap \Gamma_i(v)|}{|\Gamma_i(u)| |\Gamma_i(v)|}

        \frac{|\Gamma_o(u) \cap \Gamma_o(v)|}{|\Gamma_o(u)| |\Gamma_o(v)|}

    References
    ----------
    .. [1] Leicht, E. A. and Holme, Petter and Newman, M. E. J. (2006).
           "Vertex similarity in networks.", Phys. Rev. E, 73, 10.1103/PhysRevE.73.026120.
    """
    def predict(u, v):
        den = G.degree(u) * G.degree(v)
        if den == 0:
            return 0
        return len(list(nx.common_neighbors(G, u, v))) / den

    def predict_in(u, v):
        su = set(map(lambda e: e[0], G.in_edges(u)))
        sv = set(map(lambda e: e[0], G.in_edges(v)))
        den = len(su) * len(sv)
        if den == 0:
            return 0
        return len(su & sv) / den

    def predict_out(u, v):
        su = set(map(lambda e: e[1], G.out_edges(u)))
        sv = set(map(lambda e: e[1], G.out_edges(v)))
        den = len(su) * len(sv)
        if den == 0:
            return 0
        return len(su & sv) / den

    # Select the appropriate function and return the results
    if G.is_directed():
        if neighbourhood == 'in':
            return _apply_prediction(G, predict_in, ebunch)
        elif neighbourhood == 'out':
            return _apply_prediction(G, predict_out, ebunch)
        else:
            raise ValueError("Unknown parameter value.")
    return _apply_prediction(G, predict, ebunch)


def topological_overlap(G, ebunch=None, neighbourhood='in'):
    r"""
    Computes the topological overlap [2]_ between all node pairs in ebunch; or all nodes in G, if ebunch is None.
    Can be computed for directed and undirected graphs (see Notes for exact definitions).

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    ebunch : iterable, optional
        An iterable of node pairs. If None, all edges in G will be used. Default is None.
    neighbourhood : string, optional
        For directed graphs only. Determines if the in or the out-neighbourhood of nodes should be used.
        Default is 'in'.

    Returns
    -------
    sim : list
        A list of node-pair similarities in the same order as ebunch.

    Raises
    ------
    ValueError
        If G is directed and neighbourhood is not one of 'in' or 'out'.

    Notes
    -----
    For undirected graphs the topological overlap of nodes 'u' and 'v' is defined as:

    .. math:: \frac{|\Gamma(u) \cap \Gamma(v)|}{min(|\Gamma(u)|,|\Gamma(v)|)}

    For directed graphs we can consider either the in or the out-neighbourhoods, respectively:

    .. math::

        \frac{|\Gamma_i(u) \cap \Gamma_i(v)|}{min(|\Gamma_i(u)|,|\Gamma_i(v)|)}

        \frac{|\Gamma_o(u) \cap \Gamma_o(v)|}{min(|\Gamma_o(u)|,|\Gamma_o(v)|)}

    References
    ----------
    .. [2] Ravasz, E., Somera, A. L., Mongru, D. A., Oltvai, Z. N., & Barabási, A. L. (2002).
           "Hierarchical organization of modularity in metabolic networks." Science, 297(5586), 1551-1555.
    """
    def predict(u, v):
        den = min(G.degree(u), G.degree(v))
        if den == 0:
            return 0
        return len(list(nx.common_neighbors(G, u, v))) / den

    def predict_in(u, v):
        su = set(map(lambda e: e[0], G.in_edges(u)))
        sv = set(map(lambda e: e[0], G.in_edges(v)))
        den = min(len(su), len(sv))
        if den == 0:
            return 0
        return len(su & sv) / den

    def predict_out(u, v):
        su = set(map(lambda e: e[1], G.out_edges(u)))
        sv = set(map(lambda e: e[1], G.out_edges(v)))
        den = min(len(su), len(sv))
        if den == 0:
            return 0
        return len(su & sv) / den

    # Select the appropriate function and return the results
    if G.is_directed():
        if neighbourhood == 'in':
            return _apply_prediction(G, predict_in, ebunch)
        elif neighbourhood == 'out':
            return _apply_prediction(G, predict_out, ebunch)
        else:
            raise ValueError("Unknown parameter value.")
    return _apply_prediction(G, predict, ebunch)


def adamic_adar_index(G, ebunch=None, neighbourhood='in'):
    r"""
    Computes the Adamic-Adar index between all node pairs in ebunch; or all nodes in G, if ebunch is None.
    Can be computed for directed and undirected graphs (see Notes for exact definitions).

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    ebunch : iterable, optional
        An iterable of node pairs. If None, all edges in G will be used. Default is None.
    neighbourhood : string, optional
        For directed graphs only. Determines if the in or the out-neighbourhood of nodes should be used.
        Default is 'in'.

    Returns
    -------
    sim : list
        A list of node-pair similarities in the same order as ebunch.

    Raises
    ------
    ValueError
        If G is directed and neighbourhood is not one of 'in' or 'out'.

    Notes
    -----
    For undirected graphs the Adamic-Adar index of nodes 'u' and 'v' is defined as:

    .. math:: \sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{1}{\log |\Gamma(w)|}

    For directed graphs we can consider either the in or the out-neighbourhoods, respectively:

    .. math::

        \sum_{w \in \Gamma_i(u) \cap \Gamma_i(v)} \frac{1}{\log |\Gamma_i(w)|}

        \sum_{w \in \Gamma_o(u) \cap \Gamma_o(v)} \frac{1}{\log |\Gamma_o(w)|}

    """
    def predict(u, v):
        return sum(1 / math.log(G.degree(w)) for w in nx.common_neighbors(G, u, v))

    def predict_in(u, v):
        su = set(map(lambda e: e[0], G.in_edges(u)))
        sv = set(map(lambda e: e[0], G.in_edges(v)))
        inters = su & sv
        res = 0
        for w in inters:
            l = len(G.in_edges(w))
            if l > 1:
                res += 1 / math.log(l)
        return res

    def predict_out(u, v):
        su = set(map(lambda e: e[1], G.out_edges(u)))
        sv = set(map(lambda e: e[1], G.out_edges(v)))
        inters = su & sv
        res = 0
        for w in inters:
            l = len(G.out_edges(w))
            if l > 1:
                res += 1 / math.log(l)
        return res

    # Select the appropriate function and return the results
    if G.is_directed():
        if neighbourhood == 'in':
            return _apply_prediction(G, predict_in, ebunch)
        elif neighbourhood == 'out':
            return _apply_prediction(G, predict_out, ebunch)
        else:
            raise ValueError("Unknown parameter value.")
    return _apply_prediction(G, predict, ebunch)


def resource_allocation_index(G, ebunch=None, neighbourhood='in'):
    r"""
    Computes the resource allocation index between all node pairs in ebunch; or all nodes in G, if ebunch is None.
    Can be computed for directed and undirected graphs (see Notes for exact definitions).

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    ebunch : iterable, optional
        An iterable of node pairs. If None, all edges in G will be used. Default is None.
    neighbourhood : string, optional
        For directed graphs only. Determines if the in or the out-neighbourhood of nodes should be used.
        Default is 'in'.

    Returns
    -------
    sim : list
        A list of node-pair similarities in the same order as ebunch.

    Raises
    ------
    ValueError
        If G is directed and neighbourhood is not one of 'in' or 'out'.

    Notes
    -----
    For undirected graphs the resource allocation index of nodes 'u' and 'v' is defined as:

    .. math:: \sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{1}{| \Gamma(w) |}

    For directed graphs we can consider either the in or the out-neighbourhoods, respectively:

    .. math::

        \sum_{w \in \Gamma_i(u) \cap \Gamma_i(v)} \frac{1}{|\Gamma_i(w)|}

        \sum_{w \in \Gamma_o(u) \cap \Gamma_o(v)} \frac{1}{|\Gamma_o(w)|}

    """
    def predict(u, v):
        return sum(1 / G.degree(w) for w in nx.common_neighbors(G, u, v))

    def predict_in(u, v):
        su = set(map(lambda e: e[0], G.in_edges(u)))
        sv = set(map(lambda e: e[0], G.in_edges(v)))
        inters = su & sv
        res = 0
        for w in inters:
            l = len(G.in_edges(w))
            if l > 1:
                res += 1 / l
        return res

    def predict_out(u, v):
        su = set(map(lambda e: e[1], G.out_edges(u)))
        sv = set(map(lambda e: e[1], G.out_edges(v)))
        inters = su & sv
        res = 0
        for w in inters:
            l = len(G.out_edges(w))
            if l > 1:
                res += 1 / l
        return res

    # Select the appropriate function and return the results
    if G.is_directed():
        if neighbourhood == 'in':
            return _apply_prediction(G, predict_in, ebunch)
        elif neighbourhood == 'out':
            return _apply_prediction(G, predict_out, ebunch)
        else:
            raise ValueError("Unknown parameter value.")
    return _apply_prediction(G, predict, ebunch)


def preferential_attachment(G, ebunch=None, neighbourhood='in'):
    r"""
    Computes the preferential attachment score between all node pairs in ebunch; or all nodes in G, if ebunch is None.
    Can be computed for directed and undirected graphs (see Notes for exact definitions).

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    ebunch : iterable, optional
        An iterable of node pairs. If None, all edges in G will be used. Default is None.
    neighbourhood : string, optional
        For directed graphs only. Determines if the in or the out-neighbourhood of nodes should be used.
        Default is 'in'.

    Returns
    -------
    sim : list
        A list of node-pair similarities in the same order as ebunch.

    Raises
    ------
    ValueError
        If G is directed and neighbourhood is not one of 'in' or 'out'.

    Notes
    -----
    For undirected graphs the preferential attachment score of nodes 'u' and 'v' is defined as:

    .. math:: |\Gamma(u)| |\Gamma(v)|

    For directed graphs we can consider either the in or the out-neighbourhoods, respectively:

    .. math::

        |\Gamma_i(u)| |\Gamma_i(v)|

        |\Gamma_o(u)| |\Gamma_o(v)|

    """
    def predict(u, v):
        return G.degree(u) * G.degree(v)

    def predict_in(u, v):
        return len(G.in_edges(u)) * len(G.in_edges(v))

    def predict_out(u, v):
        return len(G.out_edges(u)) * len(G.out_edges(v))

    # Select the appropriate function and return the results
    if G.is_directed():
        if neighbourhood == 'in':
            return _apply_prediction(G, predict_in, ebunch)
        elif neighbourhood == 'out':
            return _apply_prediction(G, predict_out, ebunch)
        else:
            raise ValueError("Unknown parameter value.")
    return _apply_prediction(G, predict, ebunch)


def random_prediction(G, ebunch=None, neighbourhood='in'):
    r"""
    Returns a float drawn uniformly at random from the interval (0.0, 1.0] for all node pairs in ebunch; or all nodes
    in G, if ebunch is None. Can be computed for directed and undirected graphs.

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    ebunch : iterable, optional
        An iterable of node pairs. If None, all edges in G will be used. Default is None.
    neighbourhood : string, optional
        Not used.

    Returns
    -------
    sim : list
        A list of node-pair similarities in the same order as ebunch.
    """
    def predict(u, v):
        return 1 if random.random() > 0.5 else 0

    return _apply_prediction(G, predict, ebunch)


def all_baselines(G, ebunch=None, neighbourhood='in'):
    r"""
    Computes a 5-dimensional embedding for all node pairs in ebunch; or all nodes in G, if ebunch is None.
    Each of the 5 dimensions correspond to the similarity between the nodes as computed by a different function
    (i.e. CN, JC, AA, RAI and PA). Can be computed for directed and undirected graphs.

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    ebunch : iterable, optional
        An iterable of node pairs. If None, all edges in G will be used. Default is None.
    neighbourhood : string, optional
        For directed graphs only. Determines if the in or the out-neighbourhood of nodes should be used.
        Default is 'in'.

    Returns
    -------
    emb : ndarray
        Column vector containing node-pair embeddings as rows.

    Raises
    ------
    ValueError
        If G is directed and neighbourhood is not one of 'in' or 'out'.
    """
    if ebunch is None:
        ebunch = list(G.edges)
    emb = np.zeros((len(ebunch), 5))
    for i in range(len(ebunch)):
        emb[i][0] = common_neighbours(G, [ebunch[i]], neighbourhood)[0]
        emb[i][1] = jaccard_coefficient(G, [ebunch[i]], neighbourhood)[0]
        emb[i][2] = adamic_adar_index(G, [ebunch[i]], neighbourhood)[0]
        emb[i][3] = resource_allocation_index(G, [ebunch[i]], neighbourhood)[0]
        emb[i][4] = preferential_attachment(G, [ebunch[i]], neighbourhood)[0]
    return emb
