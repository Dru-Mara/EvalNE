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

from __future__ import division

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import inv

from evalne.evaluation import score


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
        self.sim = None
        self._fit()

    def _fit(self):
        adj = nx.adjacency_matrix(self._G)
        ident = sparse.identity(len(self._G.nodes))
        self.sim = inv(ident - self.beta * adj.transpose()) - ident     # Maybe adj should not be inverted¿?

    def predict(self, ebunch):
        ebunch = np.array(ebunch)
        return np.array(self.sim[ebunch[:, 0], ebunch[:, 1]]).flatten()

    def scoresheet(self, ebunch, y_true):
        # metrics.score(y_true=y_true, y_pred=self.predict(ebunch), method=self.scoring_method)
        y_pred = self.predict(ebunch)
        return score.ScoreSheet(method='Katz', params={'beta': self.beta},
                                test_results=True, y_true=y_true, y_pred=y_pred, y_bin=y_pred)

    def save_sim_matrix(self, filename):
        np.savetxt(filename, self.sim, delimiter=',', fmt='%d')


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

    def scoresheet(self, ebunch, y_true):
        # metrics.score(y_true=y_true, y_pred=self.fit_predict(ebunch), method=self.scoring_method)
        y_pred = self.fit_predict(ebunch)
        # return score.Scores(y_true=y_true, y_pred=y_pred, y_bin=y_pred)
        return score.ScoreSheet(method='Katz', params={'beta': self.beta, 'path_len': self.path_len},
                                test_results=True, y_true=y_true, y_pred=y_pred, y_bin=y_pred)
