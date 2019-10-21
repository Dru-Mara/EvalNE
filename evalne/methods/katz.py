#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# This code provides two different implementations of the katz score. The first one computes the exact score for the
# complete graph using the adjacency matrix. The second one computes the approximated Katz score for each pair of input
# nodes.
# Only undirected Graphs and Digraphs are supported.

# TODO: the predict method will not work if the nodes are not consecutive integers
# TODO: Both the exact (with sparse matrices) and aprox versions are extremely slow
# TODO: the default for now is to take the adj mat as dense and do the computations. Can easily run out of memory...

from __future__ import division

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import inv


class Katz(object):
    r"""
    Computes the Katz similarity based on paths between nodes in the graph. Shorter paths will contribute more than
    longer ones. This contribution depends of the damping factor 'beta'. The exact score is computed using the
    adj matrix of the full graph. This class exposes fit, predict, score and save_sim_matrix functions.

    Parameters
    ----------
    G : graph
        A NetworkX graph
    beta = float, optional
        The damping factor for the model. Default is 0.005
    """

    def __init__(self, G, beta=0.005):
        self._G = G
        self.beta = beta
        self.sim = self._fit()

    def _fit(self):

        # Versions using sparse matrices
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
        ebunch = np.array(ebunch)
        return np.array(self.sim[ebunch[:, 0], ebunch[:, 1]]).flatten()

    def save_sim_matrix(self, filename):
        np.savetxt(filename, self.sim, delimiter=',', fmt='%d')

    def get_params(self):
        params = {'beta': self.beta}
        return params


class KatzApprox(object):
    r"""
    Computes the Katz similarity based on paths between nodes in the graph. Shorter paths will contribute more than
    longer ones. This contribution depends of the damping factor 'beta'. The approximated score is computed using only
    a subset of paths of length at most 'path_len' between every pair of nodes. This class exposes fit_predict
    and score functions.
    Reference: https://surface.syr.edu/etd/355/

    Parameters
    ----------
    G : graph
        A NetworkX graph
    beta : float, optional
        The damping factor for the model. Default is 0.005
    path_len : int, optional
        The maximum path length to consider between each pair of nodes. Default is 3.
    """

    def __init__(self, G, beta=0.005, path_len=3):
        self._G = G
        self.beta = beta
        self.path_len = path_len

    def fit_predict(self, ebunch):
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
        params = {'beta': self.beta, 'path_len': self.path_len}
        return params
