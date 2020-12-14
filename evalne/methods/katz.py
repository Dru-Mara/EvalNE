#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# This file provides two different implementations of the katz score. The first, computes the exact score for the
# complete graph using the adjacency matrix. The second, computes the approximated Katz score for each input node-pair.
# Only undirected Graphs and Digraphs are supported.

from __future__ import division

import networkx as nx
import numpy as np

__all__ = ['Katz', 'KatzApprox']


class Katz(object):
    """
    Computes the exact katz similarity based on paths between nodes in the graph. Shorter paths will contribute more
    than longer ones. This contribution depends of the damping factor 'beta'. The exact katz score is computed using
    the adj matrix of the full graph.

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph with nodes being consecutive integers starting at 0.
    beta = float, optional
        The damping factor for the model. Default is 0.005.

    Notes
    -----
    The execution is based on dense matrices, so it may run out of memory.
    """

    def __init__(self, G, beta=0.005):
        self._G = G
        self.beta = beta
        self.sim = self._fit()

    def _fit(self):

        # Version using sparse matrices
        # adj = nx.adjacency_matrix(self._G)
        # ident = sparse.identity(len(self._G.nodes)).tocsc()
        # sim = inv(ident - adj.multiply(self.beta).T) - ident
        # adj = nx.adjacency_matrix(self._G)
        # aux = adj.multiply(-self.beta).T
        # aux.setdiag(1+aux.diagonal(), k=0)
        # sim = inv(aux)
        # sim.setdiag(sim.diagonal()-1)
        # print(sim.nnz)
        # print(adj.nnz)

        # Version using dense matrices
        adj = nx.adjacency_matrix(self._G)
        aux = adj.T.multiply(-self.beta).todense()
        np.fill_diagonal(aux, 1+aux.diagonal())
        sim = np.linalg.inv(aux)
        np.fill_diagonal(sim, sim.diagonal()-1)
        return sim

    def predict(self, ebunch):
        """
        Computes the katz score for all node-pairs in ebunch.

        Parameters
        ----------
        ebunch : iterable
            An  iterable of node-pairs for which to compute the katz score.

        Returns
        -------
        ndarray
            An array containing the similarity scores.
        """
        ebunch = np.array(ebunch)
        return np.array(self.sim[ebunch[:, 0], ebunch[:, 1]]).flatten()

    def save_sim_matrix(self, filename):
        """
        Stores the similarity matrix to a file with the given name.

        Parameters
        ----------
        filename : string
            The name and path of the file where the similarity matrix should be stored.
        """
        np.savetxt(filename, self.sim, delimiter=',', fmt='%d')

    def get_params(self):
        """
        Returns a dictionary of model parameters.

        Returns
        -------
        params : dict
            A dictionary of model parameters and their values.
        """
        params = {'beta': self.beta}
        return params


class KatzApprox(object):
    """
    Computes the approximated katz similarity based on paths between nodes in the graph. Shorter paths will contribute
    more than longer ones. This contribution depends of the damping factor 'beta'. The approximated score is computed
    using all paths between nodes of length at most 'path_len'.

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    beta : float, optional
        The damping factor for the model. Default is 0.005.
    path_len : int, optional
        The maximum path length to consider between each pair of nodes. Default is 3.

    Notes
    -----
    The implementation follows the indication in [1]. It becomes extremely slow for large dense graphs.

    References
    ----------
    .. [1] R. Laishram "Link Prediction in Dynamic Weighted and Directed Social Network using Supervised Learning"
           Dissertations - ALL. 355. 2015
    """

    def __init__(self, G, beta=0.005, path_len=3):
        self._G = G
        self.beta = beta
        self.path_len = path_len

    def fit_predict(self, ebunch):
        """
        Computes the katz score for all node-pairs in ebunch.

        Parameters
        ----------
        ebunch : iterable
            An  iterable of node-pairs for which to compute the katz score.

        Returns
        -------
        ndarray
            An array containing the similarity scores.
        """
        res = list()
        betas = np.zeros(self.path_len)
        for i in range(len(betas)):
            betas[i] = np.power(self.beta, i+1)
        for u, v in ebunch:
            paths = np.zeros(self.path_len)
            for path in nx.all_simple_paths(self._G, source=u, target=v, cutoff=self.path_len):
                paths[len(path)-2] += 1     # Simple paths output at most path_len+1, plus -1 because indexing at 0
            res.append(np.sum(betas * paths))
        return np.array(res).reshape(-1, 1)

    def get_params(self):
        """
        Returns a dictionary of model parameters.

        Returns
        -------
        params : dict
            A dictionary of model parameters and their values.
        """
        params = {'beta': self.beta, 'path_len': self.path_len}
        return params
