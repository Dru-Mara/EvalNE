#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# This code is an extension of the networkx.link_prediction library where directed graphs are not considered.
# The extensions to directed networks were done according to the paper: https://surface.syr.edu/etd/355/
# This code provides implementations of several simple methods commonly used as evaluation in link prediction tasks.
# All these methods return a similarity score for each pair of input nodes.
# MultiGraphs and weighted graphs are not supported!

# TODO: the apply_prediction method should probably return a numpy array (like edge_embeddings does) rather than a list.

from __future__ import division

import networkx as nx
import numpy as np

__all__ = ['common_neighbours',
           'jaccard_coefficient',
           'adamic_adar_index',
           'resource_allocation_index',
           'preferential_attachment',
           'random_prediction',
           'all_baselines']


def _apply_prediction(G, func, ebunch=None):
    r"""
    Applies the given function to each edge in the specified iterable of edges.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    func : function
        A function on two inputs, each of which is a node in the graph. Can return anything,
        but it should return a value representing a prediction of the likelihood of a "link" joining the two nodes.
    ebunch : iterable, optional
        An iterable of pairs of nodes. If not specified, all edges in the graph G will be used.

    Returns
    -------
    sim : list
        A list of values in the same order as ebunch representing the similarity of each pair of nodes.
    """
    if ebunch is None:
        ebunch = list(G.edges)
    return list(map(lambda e: func(e[0], e[1]), ebunch))


def all_baselines(G, ebunch, neighbourhood='in'):
    """
    Computes a 5-dimensional embedding for each graph edge as an aggregation of the following 5 LP heuristics:
    [CN, JC, AA, RAI, PA.]

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    ebunch : iterable of node pairs, optional
        Common neighbours will be computed for each pair of nodes given in the iterable. The pairs must
        be given as 2-tuples (u, v) where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used. Default value is None.
    neighbourhood : string, optional
        For directed graphs determines if the in or the out-neighbourhoods of nodes should be used.
        Default value is 'in'.

    Returns
    -------
    emb : numpy array
        A numpy array representing the edge embeddings in the same order as ebunch.
    """
    emb = np.zeros((len(ebunch), 5))
    for i in range(len(ebunch)):
        emb[i][0] = common_neighbours(G, [ebunch[i]], neighbourhood)[0]
        emb[i][1] = jaccard_coefficient(G, [ebunch[i]], neighbourhood)[0]
        emb[i][2] = adamic_adar_index(G, [ebunch[i]], neighbourhood)[0]
        emb[i][3] = resource_allocation_index(G, [ebunch[i]], neighbourhood)[0]
        emb[i][4] = preferential_attachment(G, [ebunch[i]], neighbourhood)[0]
    return emb


def common_neighbours(G, ebunch=None, neighbourhood='in'):
    """
    Compute the common neighbours of all node pairs in ebunch.
    For undirected graphs common neighbours of nodes 'u' and 'v' is defined as:
    :math:`|\Gamma(u) \cap \Gamma(v)|`
    For directed graphs we can consider either the in or the out-neighbourhood:
    :math:`|\Gamma_i(u) \cap \Gamma_i(v)|`
    :math:`|\Gamma_o(u) \cap \Gamma_o(v)|`

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    ebunch : iterable of node pairs, optional
        Common neighbours will be computed for each pair of nodes given in the iterable. The pairs must
        be given as 2-tuples (u, v) where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used. Default value is None.
    neighbourhood : string, optional
        For directed graphs determines if the in or the out-neighbourhoods of nodes should be used.
        Default value is 'in'.

    Returns
    -------
    sim : list
        A list of values in the same order as ebunch representing the similarity of each pair of nodes.

    Raises
    ------
    ValueError
        If G is directed and neighbourhood is not one of 'in' or 'out'.
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
    Compute the Jaccard coefficient of all node pairs in ebunch.
    For undirected graphs Jaccard coefficient of nodes 'u' and 'v' is defined as:
    :math:`|\Gamma(u) \cap \Gamma(v)| / |\Gamma(u) \cup \Gamma(v)|`
    For directed graphs we can consider either the in or the out-neighbourhood:
    :math:`|\Gamma_i(u) \cap \Gamma_i(v)| / |\Gamma_i(u) \cup \Gamma_i(v)|`
    :math:`|\Gamma_o(u) \cap \Gamma_o(v)| / |\Gamma_o(u) \cup \Gamma_o(v)|`

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    ebunch : iterable of node pairs, optional
        Jaccard coefficient will be computed for each pair of nodes given in the iterable. The pairs must
        be given as 2-tuples (u, v) where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used. Default value is None.
    neighbourhood : string, optional
        For directed graphs determines if the in or the out-neighbourhoods of nodes should be used.
        Default value is 'in'.

    Returns
    -------
    sim : list
        A list of values in the same order as ebunch representing the similarity of each pair of nodes.

    Raises
    ------
    ValueError
        If G is directed and neighbourhood is not one of 'in' or 'out'.
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


def adamic_adar_index(G, ebunch=None, neighbourhood='in'):
    r"""
    Compute the Adamic-Adar index of all node pairs in ebunch.
    For undirected graphs the Adamic-Adar index of nodes 'u' and 'v' is defined as:
    :math:`\sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{1}{\log |\Gamma(w)|}`
    For directed graphs we can consider either the in or the out-neighbourhood:
    :math:`\sum_{w \in \Gamma_i(u) \cap \Gamma_i(v)} \frac{1}{\log |\Gamma_i(w)|}`
    :math:`\sum_{w \in \Gamma_o(u) \cap \Gamma_o(v)} \frac{1}{\log |\Gamma_o(w)|}`

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    ebunch : iterable of node pairs, optional
        Adamic-Adar index will be computed for each pair of nodes given in the iterable. The pairs must
        be given as 2-tuples (u, v) where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used. Default value is None.
    neighbourhood : string, optional
        For directed graphs determines if the in or the out-neighbourhoods of nodes should be used.
        Default value is 'in'.

    Returns
    -------
    sim : list
        A list of values in the same order as ebunch representing the similarity of each pair of nodes.

    Raises
    ------
    ValueError
        If G is directed and neighbourhood is not one of 'in' or 'out'.
    """
    def predict(u, v):
        return sum(1.0 / np.log(G.degree(w)) for w in nx.common_neighbors(G, u, v))

    def predict_in(u, v):
        su = set(map(lambda e: e[0], G.in_edges(u)))
        sv = set(map(lambda e: e[0], G.in_edges(v)))
        inters = su & sv
        res = 0
        for w in inters:
            l = len(G.in_edges(w))
            if l > 1:
                res += 1 / np.log(l)
        return res

    def predict_out(u, v):
        su = set(map(lambda e: e[1], G.out_edges(u)))
        sv = set(map(lambda e: e[1], G.out_edges(v)))
        inters = su & sv
        res = 0
        for w in inters:
            l = len(G.out_edges(w))
            if l > 1:
                res += 1 / np.log(l)
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
    Compute the resource allocation index of all node pairs in ebunch.
    For undirected graphs the resource allocation index of nodes 'u' and 'v' is defined as:
    :math:`\sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{1}{|\Gamma(w)|}`
    For directed graphs we can consider either the in or the out-neighbourhood:
    :math:`\sum_{w \in \Gamma_i(u) \cap \Gamma_i(v)} \frac{1}{|\Gamma_i(w)|}`
    :math:`\sum_{w \in \Gamma_o(u) \cap \Gamma_o(v)} \frac{1}{|\Gamma_o(w)|}`

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    ebunch : iterable of node pairs, optional
        Resource allocation index will be computed for each pair of nodes given in the iterable.
        The pairs must be given as 2-tuples (u, v) where u and v are nodes in the graph. If ebunch
        is None then all non-existent edges in the graph will be used. Default value is None.
    neighbourhood : string, optional
        For directed graphs determines if the in or the out-neighbourhoods of nodes should be used.
        Default value is 'in'.

    Returns
    -------
    sim : list
        A list of values in the same order as ebunch representing the similarity of each pair of nodes.

    Raises
    ------
    ValueError
        If G is directed and neighbourhood is not one of 'in' or 'out'.
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
    Compute the preferential attachment score of all node pairs in ebunch.
    For undirected graphs the preferential attachment score of nodes 'u' and 'v' is defined as:
    :math:`|\Gamma(u)| |\Gamma(v)|`
    For directed graphs we can consider either the in or the out-neighbourhood:
    :math:`|\Gamma_i(u)| |\Gamma_i(v)|`
    :math:`|\Gamma_o(u)| |\Gamma_o(v)|`

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    ebunch : iterable of node pairs, optional
        Preferential attachment score will be computed for each pair of nodes given in the iterable.
        The pairs must be given as 2-tuples (u, v) where u and v are nodes in the graph. If ebunch
        is None then all non-existent edges in the graph will be used. Default value is None.
    neighbourhood : string, optional
        For directed graphs determines if the in or the out-neighbourhoods of nodes should be used.
        Default value is 'in'.

    Returns
    -------
    sim : list
        A list of values in the same order as ebunch representing the similarity of each pair of nodes.

    Raises
    ------
    ValueError
        If G is directed and neighbourhood is not one of 'in' or 'out'.
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
    Returns a float draws uniformly at random from the interval (0.0, 1.0].

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    ebunch : iterable of node pairs, optional
        A random prediction will be returned for each pair of nodes given in the iterable.
        The pairs must be given as 2-tuples (u, v) where u and v are nodes in the graph. If ebunch
        is None then all non-existent edges in the graph will be used. Default value is None.
    neighbourhood : string, optional
        Not used.

    Returns
    -------
    sim : list
        A list of values in the same order as ebunch representing the similarity of each pair of nodes.
    """
    def predict(u, v):
        return 1 if np.random.random() > 0.5 else 0

    return _apply_prediction(G, predict, ebunch)

