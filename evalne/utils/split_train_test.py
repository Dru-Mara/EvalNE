#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# This file provides a set of functions for (1) edge sampling: sampling sets of train and test edges from given input
# graphs, (2) non-edge sampling: sampling sets of train and test non-edges i.e. pairs of unconnected nodes,
# (3) node-pair sampling: randomly sampling a subset of all possible node pairs including edges and non-edges.
# Additional functions for computing a spanning tree of a given graph are provided.

from __future__ import division
from __future__ import print_function

import os
import random
import warnings

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import tril
from scipy.sparse import triu
from scipy.sparse.csgraph import depth_first_tree
from sklearn.externals.joblib import Parallel, delayed

from evalne.utils import preprocess as pp


##################################################
#                 Edge sampling                  #
##################################################

def split_train_test(G, train_frac=0.51, st_alg='wilson'):
    """
    Splits the edges of the input graph in sets of train and test and returns the results. Split is performed using the
    spanning tree approach (see Notes). The resulting train edge set has the following properties: spans a graph
    (digraph) with a single connected (weakly connected) component and the same nodes as G.
    
    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph with a single connected (weakly connected) component.
    train_frac : float, optional
        The proportion of train edges w.r.t. the total number of edges in the input graph (range (0.0, 1.0]).
        Default is 0.51.
    st_alg : string, optional
        The algorithm to use for generating the spanning tree constituting the backbone of the train set.
        Options are: 'wilson' and 'broder'. The first option, 'wilson' is faster in most cases. Default is 'wilson'.

    Returns
    -------
    train_E : set
       The set of train edges.
    test_E : set
       The set of test edges.

    Raises
    ------
    ValueError
        If the train_frac parameter is not in range (0, 1].
        If the input graph G has more than one (weakly) connected component.

    See Also
    --------
    evalne.utils.split_train_test.broder_alg, evalne.utils.split_train_test.wilson_alg :
        The two options for the st_alg parameter.

    Notes
    -----
    The method proceeds as follows: (1) a spanning tree of the input graph is selected uniformly at random using broder
    or wilson's algorithm, (2) randomly selected edges are added to those of the spanning tree until train_frac is
    reached, (3) the remaining edges, not used in previous steps, form the test set.
    """
    # Sanity check to make sure the input is correct
    _sanity_check(G)
    if train_frac <= 0.0 or train_frac > 1.0:
        raise ValueError('The train_frac parameter needs to be in range: (0.0, 1.0]')
    if train_frac == 1.0:
        return set(G.edges()), set()

    # Create a set of all edges in G
    E = set(G.edges)

    if st_alg == 'broder':
        # Compute a random spanning tree using broder's algorithm
        train_E = broder_alg(G, E)
    else:
        # Compute a random spanning tree using wilson's algorithm
        train_E = wilson_alg(G, E)

    # Fill test edge set as all edges not in the spanning tree
    test_E = E - train_E

    # Compute num train edges
    num_E = len(E)
    num_train_E = np.ceil(train_frac * num_E)

    # Check if the num edges in the spanning tree is already greater than the num train edges
    num_toadd = int(num_train_E - len(train_E))
    if num_toadd <= 0:
        print("WARNING: In order to return a connected train set the train_frac parameter needs to be higher!")
        print("In this case, the provided train set constitutes a random spanning tree of the input graph.")
        print("The train_frac value used is: {}".format(len(train_E) / num_E))
        print("Edges requested: train = {}, test = {}".format(num_train_E, num_E - num_train_E))
        print("Edges returned: train = {}, test = {}".format(len(train_E), num_E - len(train_E)))
    else:
        # Add more edges to train set from test set until it has desired size
        edges = set(random.sample(test_E, num_toadd))
        test_E = test_E - edges
        train_E = train_E | edges

    # Perform some simple checks
    assert E == (test_E | train_E)
    assert len(E) == len(test_E) + len(train_E)
    if num_toadd > 0:
        assert num_train_E == len(train_E)

    # Return the sets of edges
    return train_E, test_E


def quick_split(G, train_frac=0.51):
    """
    Splits the edges of the input graph in sets of train and test and returns the results. Split is performed using the
    quick split approach (see Notes). The resulting train edge set has the following properties: spans a graph
    (digraph) with a single connected (weakly connected) component and the same nodes as G.

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph with a single connected (weakly connected) component.
    train_frac : float, optional
        The proportion of train edges w.r.t. the total number of edges in the input graph (range (0.0, 1.0]).
        Default is 0.51.

    Returns
    -------
    train_E : ndarray
       Column vector of train edges as pairs src, dst.
    test_E : ndarray
       Column vector of test edges as pairs src, dst.

    Raises
    ------
    ValueError
        If the train_frac parameter is not in range (0, 1].
        If the input graph G has more than one (weakly) connected component.

    Notes
    -----
    The method proceeds as follows: (1) a spanning tree of the input graph is generated using a depth first tree
    approach starting at a random node, (2) randomly selected edges are added to those of the spanning tree until
    train_frac is reached, (3) the remaining edges, not used in previous steps, form the test set.
    """
    # Sanity check to make sure the input is correct
    _sanity_check(G)
    if train_frac <= 0.0 or train_frac > 1.0:
        raise ValueError('The train_frac parameter needs to be in range: (0.0, 1.0]')
    if train_frac == 1.0:
        return set(G.edges()), set()

    # Get Adj matrix
    if nx.is_directed(G):
        a = nx.adj_matrix(G)
    else:
        a = triu(nx.adj_matrix(G), k=1)

    # Compute initial statistics and linear indx of nonzeros
    n = a.shape[0]
    num_tr_e = int(a.nnz * train_frac)
    nz_lin_ind = np.ravel_multi_index(a.nonzero(), (n, n))

    # Build a dft starting at a random node. If dir false returns only upper triangle
    dft = depth_first_tree(a, np.random.randint(0, a.shape[0]), directed=nx.is_directed(G))
    if nx.is_directed(G):
        dft_lin_ind = np.ravel_multi_index(dft.nonzero(), (n, n))
    else:
        dft_lin_ind = np.ravel_multi_index(triu(tril(dft).T + dft, k=1).nonzero(), (n, n))

    # From all nonzero indx remove those in dft. From the rest take enough to fill train quota. Rest are test
    rest_lin_ind = np.setdiff1d(nz_lin_ind, dft_lin_ind)
    aux = np.random.choice(rest_lin_ind, num_tr_e-len(dft_lin_ind), replace=False)
    lin_tr_e = np.union1d(dft_lin_ind, aux)
    lin_te_e = np.setdiff1d(rest_lin_ind, aux)

    # Unravel the linear indices to obtain src, dst pairs
    tr_e = np.array(np.unravel_index(np.array(lin_tr_e), (n, n))).T
    te_e = np.array(np.unravel_index(np.array(lin_te_e), (n, n))).T

    # Return the sets of edges
    return tr_e, te_e


def naive_split_train_test(G, train_frac=0.51):
    """
    Splits the edges of the input graph in sets of train and test and returns the results. Split is performed using the
    naive split approach (see Notes). The resulting train edge set has the following properties: spans a graph
    (digraph) with a single connected (weakly connected) component and the same nodes as G.

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph with a single connected (weakly connected) component.
    train_frac : float, optional
        The proportion of train edges w.r.t. the total number of edges in the input graph (range (0.0, 1.0]).
        Default is 0.51.

    Returns
    -------
    train_E : set
       The set of train edges.
    test_E : set
        The set of test edges.

    Raises
    ------
    ValueError
        If the train_frac parameter is not in range (0, 1].
        If the input graph G has more than one (weakly) connected component.

    Notes
    -----
    The method proceeds as follows: (1) compute the number of test edges needed from train_frac and size of G.
    (2) randomly remove edges from the input graph one at a time. If removing an edge causes the graph to become
    disconnected, add it back, otherwise, put it in the set of test edges. (3) repeat the previous step until all
    test edges are selected. (4) the remaining edges constitute the train set.
    """
    # Sanity check to make sure the input is correct
    _sanity_check(G)
    if train_frac <= 0.0 or train_frac > 1.0:
        raise ValueError('The train_frac parameter needs to be in range: (0.0, 1.0]')
    if train_frac == 1.0:
        return set(G.edges()), set()

    # Is directed
    directed = G.is_directed()
    H = G.copy()

    # Create a set of all edges in G
    aux = np.array(H.edges)
    np.random.shuffle(aux)
    E = set([tuple(edge) for edge in aux])

    # Compute num train edges
    num_E = len(E)
    num_train_E = np.ceil(train_frac * num_E)
    num_test_E = num_E - num_train_E

    # Initialize the sets of train and test edges
    train_E = set(H.edges())
    test_E = set()

    # Iterate over shuffled edges, add to train/val sets
    for i, edge in enumerate(E):
        node1 = edge[0]
        node2 = edge[1]

        # If removing edge would disconnect a connected component, backtrack and move on
        H.remove_edge(node1, node2)
        if directed:
            if nx.number_weakly_connected_components(H) > 1:
                H.add_edge(node1, node2)
                continue
        else:
            if nx.number_connected_components(H) > 1:
                H.add_edge(node1, node2)
                continue

        # Fill test_edges
        if len(test_E) < num_test_E:
            test_E.add(edge)
            train_E.remove(edge)
        else:
            break

    # Perform some simple checks
    assert E == (test_E | train_E)
    assert len(E) == len(train_E) + len(test_E)

    # Return the sets of edges
    return train_E, test_E


def rand_split_train_test(G, train_frac=0.51):
    """
    Splits the edges of the input graph in sets of train and test and returns the results. Split is performed using the
    random split approach (see Notes). The resulting train edge set has the following properties: spans a graph
    (digraph) with a single connected (weakly connected) component.

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    train_frac : float, optional
        The proportion of train edges w.r.t. the total number of edges in the input graph (range (0.0, 1.0]).
        Default is 0.51.

    Returns
    -------
    train_E : set
        The set of train edges.
    test_E : set
        The set of test edges.

    Raises
    ------
    ValueError
        If the train_frac parameter is not in range (0, 1].

    Notes
    -----
    The method proceeds as follows: (1) randomly remove 1-train_frac percent of edges from the input graph.
    (2) from the remaining edges compute the main connected component and these will be the train edges. (3) from the
    set of removed edges, those such that both end nodes exist in the train edge set computed in the previous step,
    are added to the final test set.
    """
    if train_frac <= 0.0 or train_frac > 1.0:
        raise ValueError('The train_frac parameter needs to be in range: (0.0, 1.0]')
    if train_frac == 1.0:
        return set(G.edges()), set()

    # Create a set of all edges in G
    E = set(G.edges)
    num_E = len(E)

    # Compute the potential number of train and test edges which corresponds to the fraction given
    num_train_E = int(np.ceil(train_frac * num_E))
    num_test_E = int(num_E - num_train_E)

    # Randomly remove 1-train_frac edges from the graph and store them as potential test edges
    pte_edges = set(random.sample(E, num_test_E))

    # The remaining edges are potential train edges
    ptr_edges = E - pte_edges

    # Create a graph containing all ptr_edges and compute the mainCC
    if G.is_directed():
        H = nx.DiGraph()
        H.add_edges_from(ptr_edges)
        maincc = max(nx.weakly_connected_component_subgraphs(H), key=len)
    else:
        H = nx.Graph()
        H.add_edges_from(ptr_edges)
        maincc = max(nx.connected_component_subgraphs(H), key=len)

    # The edges in the mainCC graph are the actual train edges
    train_E = set(maincc.edges)

    # Remove potential test edges for which the end nodes do not exist in the train_E
    test_E = set()
    for (src, dst) in pte_edges:
        if src in maincc.nodes and dst in maincc.nodes:
            test_E.add((src, dst))

    # Return the sets of edges
    return train_E, test_E


def timestamp_split(G, train_frac=0.51):
    """
    Splits the edges of the input graph in sets of train and test and returns the results. Split is performed using edge
    timestamps (see Notes). The resulting train edge set has the following properties: spans a graph (digraph) with
    a single connected (weakly connected) component.

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph where edge weights are timestamps.
    train_frac : float, optional
        The proportion of train edges w.r.t. the total number of edges in the input graph (range (0.0, 1.0]).
        Default is 0.51.

    Returns
    -------
    train_E : ndarray
        Column vector of train edges as pairs src, dst.
    test_E : ndarray
        Column vector of test edges as pairs src, dst.
    tg : graph
        A NetworkX graph containing only the edges in the train edge set.

    Raises
    ------
    ValueError
        If the train_frac parameter is not in range (0, 1].
        If the input graph G has more than one (weakly) connected component.

    Notes
    -----
    The method proceeds as follows: (1) sort all edges by timestamp. (2) randomly remove 1-train_frac percent of edges
    from the input graph. (3) from the remaining edges compute the main connected component and these will be the train
    edges. (4) from the set of removed edges, those such that both end nodes exist in the train edge set computed in
    the previous step, are added to the final test set.
    """
    # Sanity check to make sure the input is correct
    _sanity_check(G)
    if train_frac <= 0.0 or train_frac > 1.0:
        raise ValueError('The train_frac parameter needs to be in range: (0.0, 1.0]')
    if train_frac == 1.0:
        return set(G.edges()), set()

    # Get Adj matrix
    if nx.is_directed(G):
        a = nx.adj_matrix(G)
    else:
        a = triu(nx.adj_matrix(G), k=1)

    # Argsort data and compute the idx where we split train from test
    ordered = np.argsort(a.data)
    split_idx = int(len(ordered) * train_frac) - 1

    # Mask train edges and get all possible edges
    mask_tr = ordered > split_idx
    nz = a.nonzero()

    # Use the mask to select only train and test from nz
    # There will be no overlap between tr and te because nz contains only unique pairs
    tr_e = np.array((nz[0][~mask_tr], nz[1][~mask_tr])).T
    te_e = np.array((nz[0][mask_tr], nz[1][mask_tr])).T

    # Taking the most recent edges for testing can cause train to be disconnected so make sure it isn't
    tg = nx.Graph()
    tg.add_edges_from(tr_e)
    tg, ids = pp.prep_graph(tg, relabel=True, del_self_loops=True, maincc=True)
    tr_e = np.array(tg.edges)

    d = dict(ids)
    te_e = set(zip(te_e[:, 0], te_e[:, 1]))
    nte_e = map(lambda x: (d.get(x[0], -1), d.get(x[1], -1)), te_e)
    te_e = np.array(nte_e)

    # We now only keep the test edges between nodes in tg
    # Remove nodes that are in test but not train
    newn = np.setdiff1d(np.unique(te_e), np.unique(tr_e))
    mask = np.isin(te_e, newn).sum(axis=1).astype(bool)
    te_e = te_e[~mask, :]

    # Return the sets of edges
    return tr_e, te_e, tg


##################################################
#               Non-edge sampling                #
##################################################


def generate_false_edges_owa(G, train_E, test_E, num_fe_train=None, num_fe_test=None):
    """
    Randomly samples pairs of unconnected nodes or non-edges from the input graph according to the open world
    assumption (see Notes). Returns sets of train and test non-edges.

    Parameters
    ----------
    G : graph
       A NetworkX graph or digraph.
    train_E : set
       The set of train edges.
    test_E : set
       The set of test edges.
    num_fe_train : int, optional
       The number of train non-edges to generate. Default is None (same number as train edges).
    num_fe_test : int, optional
       The number of test non-edges to generate. Default is None (same number as test edges).

    Returns
    -------
    train_E_false : set
       The set of train non-edges.
    test_E_false : set
       The set of test non-edges.

    Raises
    ------
    ValueError
        If the input graph G has more than one (weakly) connected component.
        If more non-edges than existing in the graph are required.

    Notes
    -----
    The open world assumption considers that for generating train non-edges one has access to the train edges only.
    Therefore, train non-edges could overlap with test edges. This scenario is common for evolving graphs where edges
    can only appear. In this case, one has access to certain train edges present in the graph at time t. Non-edges
    are then sampled using this information. At time t+1, the newly arrived test edges may have been previously
    selected as train non-edges. For undirected graphs the output is sorted (smallNodeID, bigNodeID).
    """
    # Sanity check to make sure the input is correct
    _sanity_check(G)

    # Create a set of vertices
    V = set(G.nodes)

    # Initialize the sizes of the non-edges
    if num_fe_train is None:
        num_fe_train = len(train_E)

    if num_fe_test is None:
        num_fe_test = len(test_E)

    # Make sure the required amount of non-edges can be generated
    max_nonedges = len(V) * len(V) - len(train_E)
    if num_fe_train > max_nonedges:
        raise ValueError('Too many train non-edges required! Max available for train+test is {}'.format(max_nonedges))
    else:
        if num_fe_train + num_fe_test > max_nonedges:
            warnings.warn('Too many non-edges required in train+test! '
                          'Using maximum number of test non-edges available: {}'.format(max_nonedges - num_fe_train))
            num_fe_test = max_nonedges - num_fe_train

    # Create sets to store the non-edges
    train_E_false = set()
    test_E_false = set()

    # Generate train non-edges
    while len(train_E_false) < num_fe_train:
        edge = tuple(random.sample(V, 2))
        redge = tuple(reversed(edge))
        if edge not in train_E:
            if G.is_directed():
                train_E_false.add(edge)
            else:
                if redge not in train_E:
                    train_E_false.add(tuple(sorted(edge)))

    # Generate test non-edges
    while len(test_E_false) < num_fe_test:
        edge = tuple(random.sample(V, 2))
        redge = tuple(reversed(edge))
        if edge not in train_E and edge not in test_E and edge not in train_E_false:
            if G.is_directed():
                test_E_false.add(edge)
            else:
                if redge not in train_E and redge not in test_E and redge not in train_E_false:
                    test_E_false.add(tuple(sorted(edge)))

    # Perform some simple check before returning the result
    assert len(train_E_false) == num_fe_train
    assert len(test_E_false) == num_fe_test
    assert train_E_false.isdisjoint(test_E_false)
    assert train_E_false.isdisjoint(train_E)
    assert test_E_false.isdisjoint(train_E | test_E)

    # Return the sets of non-edges
    return train_E_false, test_E_false


def generate_false_edges_cwa(G, train_E, test_E, num_fe_train=None, num_fe_test=None):
    """
    Randomly samples pairs of unconnected nodes or non-edges from the input graph according to the closed world
    assumption (see Notes). Returns sets of train and test non-edges.

    Parameters
    ----------
    G : graph
       A NetworkX graph or digraph.
    train_E : set
       The set of train edges.
    test_E : set
       The set of test edges.
    num_fe_train : int, optional
       The number of train non-edges to generate. Default is None (same number as train edges).
    num_fe_test : int, optional
       The number of test non-edges to generate. Default is None (same number as test edges).

    Returns
    -------
    train_E_false : set
       The set of train non-edges.
    test_E_false : set
       The set of test non-edges.

    Raises
    ------
    ValueError
        If the input graph G has more than one (weakly) connected component.
        If more non-edges than existing in the graph are required.

    Notes
    -----
    The closed world assumption considers that one is certain about some node pairs being non-edges. Therefore,
    train non-edges cannot overlap with wither train not test edges. This scenario is common for static graphs
    where information about both the edges (positive interactions) and non-edges (negative interactions) is knows,
    e.g. protein protein interaction networks.
    """
    # Sanity check to make sure the input is correct
    _sanity_check(G)

    # Create a set of vertices
    V = set(G.nodes)

    # Initialize the sizes of the non-edges
    if num_fe_train is None:
        num_fe_train = len(train_E)

    if num_fe_test is None:
        num_fe_test = len(test_E)

    # Make sure the required amount of non-edges can be generated
    max_nonedges = len(V) * len(V) - len(G.edges)
    if num_fe_train > max_nonedges:
        raise ValueError(
            'Too many train non-edges required! Max available for train+test is {}'.format(max_nonedges))
    else:
        if num_fe_train + num_fe_test > max_nonedges:
            warnings.warn('Too many non-edges required in train+test! '
                          'Using maximum number of test non-edges available: {}'.format(max_nonedges - num_fe_train))
            # num_fe_test = max_nonedges - num_fe_train
            return _getall_false_edges(G, (1.0*num_fe_train)/max_nonedges)

    # Create sets to store the non-edges
    train_E_false = set()
    test_E_false = set()

    # Generate train non-edges
    while len(train_E_false) < num_fe_train:
        edge = tuple(random.sample(V, 2))
        redge = tuple(reversed(edge))
        if edge not in train_E and edge not in test_E:
            if G.is_directed():
                train_E_false.add(edge)
            else:
                if redge not in train_E and redge not in test_E:
                    train_E_false.add(tuple(sorted(edge)))

    # Generate test non-edges
    while len(test_E_false) < num_fe_test:
        edge = tuple(random.sample(V, 2))
        redge = tuple(reversed(edge))
        if edge not in train_E and edge not in test_E and edge not in train_E_false:
            if G.is_directed():
                test_E_false.add(edge)
            else:
                if redge not in train_E and redge not in test_E and redge not in train_E_false:
                    test_E_false.add(tuple(sorted(edge)))

    # Perform some simple check before returning the result
    assert len(train_E_false) == num_fe_train
    assert len(test_E_false) == num_fe_test
    assert train_E_false.isdisjoint(test_E_false)
    assert train_E_false.isdisjoint(train_E | test_E)
    assert test_E_false.isdisjoint(train_E | test_E)

    # Return the sets of non-edges
    return train_E_false, test_E_false


def _getall_false_edges(G, fe_train_frac):
    """
    Helper function for generating all non-edge pairs possible and splitting them into train and test sets.

    Parameters
    ----------
    G : graph
       A NetworkX graph or digraph.
    fe_train_frac : float
        The proportion of train non-edges w.r.t. the total number of non-edges required (range (0.0, 1.0]).

    Returns
    -------
    train_E_false : set
       The set of train non-edges.
    test_E_false : set
       The set of test non-edges.
    """
    print("Generating all non-edges and splitting them in train and test...")
    train_E_false = list()
    test_E_false = list()
    for e in nx.non_edges(G):
        r = random.uniform(0, 1)
        if r <= fe_train_frac:
            train_E_false.append(e)
        else:
            test_E_false.append(e)

    # Return the sets of non-edges
    return train_E_false, test_E_false


def quick_nonedges(G, train_frac=0.51, fe_ratio=1.0):
    """
    Randomly samples pairs of unconnected nodes or non-edges from the input graph according to the open world
    assumption. Is a more efficient implementation of `generate_false_edges_owa` which also does not require the
    train and test edge sets as input. Returns sets of train and test non-edges.

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    train_frac : float, optional
        The proportion of train edges w.r.t. the total number of edges in the input graph (range (0.0, 1.0]).
        Default is 0.51.
    fe_ratio : float, optional
        The ratio of non-edges to edges to sample. For fr_ratio > 0 and < 1 less non-edges than edges will be
        generated. For fe_edges > 1 more non-edges than edges will be generated. Default 1, same amounts.

    Returns
    -------
    train_E_false : ndarray
       Column vector of train non-edges as pairs src, dst.
    test_E_false : ndarray
       Column vector of test non-edges as pairs src, dst.

    Raises
    ------
    ValueError
        If more non-edges than existing in the graph are required.
    """
    # fe_ration can be any float or keyword 'prop'
    a = nx.adj_matrix(G)
    n = a.shape[0]
    density = a.nnz / n ** 2
    if fe_ratio == 'prop':
        fe_ratio = np.floor(1.0 / density)
    if not nx.is_directed(G):
        num_fe = int((a.nnz/2.0) * fe_ratio)
    else:
        num_fe = int(a.nnz * fe_ratio)
    num_fe_tr = int(train_frac * num_fe)

    # Make sure we have enough non-edges
    if num_fe > (n**2 - (a.nnz + n)):
        raise ValueError('Too many non-edges required!')

    # Get linear indexes of 1s in A
    lin_indexes = np.ravel_multi_index(a.nonzero(), (n, n))
    inv_indx = np.union1d(lin_indexes, np.ravel_multi_index(np.diag_indices(n), (n, n)))
    # we could generate more FE than we need to make sure we find enough 0s
    candidates = np.random.randint(0, n**2, size=int(num_fe/(1-density)))

    # make sure there is no overlap
    fe_lin_ind = np.setdiff1d(candidates, inv_indx)

    while len(fe_lin_ind) < num_fe:
        new_cands = np.random.randint(0, n ** 2, size=num_fe-len(fe_lin_ind))
        valid_cands = np.setdiff1d(new_cands, inv_indx)
        fe_lin_ind = np.union1d(fe_lin_ind, valid_cands)

    fe_lin_ind = fe_lin_ind[:num_fe]
    aux = np.array(np.unravel_index(fe_lin_ind, (n, n))).T
    fe_tr = aux[:num_fe_tr, :]
    fe_te = aux[num_fe_tr:, :]

    # Return the sets of non-edges
    return fe_tr, fe_te


##################################################
#              Node-pair Sampling                #
##################################################


def _compute_one_split(G, output_path, owa=True, train_frac=0.51, num_fe_train=None, num_fe_test=None, split_id=0):
    """
    Splits the edges of the input graph in sets of train and test using the spanning tree approach and generates sets
    of train and test non-edges. The four resulting sets of node pairs are written to different files.
    The resulting train edge set has the following properties: spans a graph (digraph) with a single connected
    (weakly connected) component and the same nodes as G.

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph with a single connected (weakly connected) component.
    output_path : string
        Indicates the path where the output will be stored.
    owa : bool, optional
        Encodes the belief that the network respects or not the open world assumption. Default is True.
        If owa=True, train non-edges are sampled from the train graph only and can overlap with test edges.
        If owa=False, train non-edges are sampled from the full graph and cannot overlap with test edges.
    train_frac : float, optional
        The proportion of train edges w.r.t. the total number of edges in the input graph (range (0.0, 1.0]).
        Default is 0.51.
    num_fe_train : int, optional
        The number of train non-edges to generate. Default is None (same number as train edges).
    num_fe_test : int, optional
        The number of test non-edges to generate. Default is None (same number as test edges).
    split_id : int, optional
        A unique ID for this train/test split. Default is 0.

    See also
    --------
    evalne.utils.split_train_test.split_train_test :
        Method used to split graph edges in train and test sets.
    evalne.utils.split_train_test.generate_false_edges_owa :
        Method used to generate non-edges if owa=True.
    evalne.utils.split_train_test.generate_false_edges_cwa :
        Method used to generate non-edges if owa=False.
    """
    # Generate train and test edge splits
    train_E, test_E = split_train_test(G, train_frac)

    # Generate the train/test false edges
    if owa:
        train_E_false, test_E_false = generate_false_edges_owa(G, train_E, test_E, num_fe_train, num_fe_test)
    else:
        train_E_false, test_E_false = generate_false_edges_cwa(G, train_E, test_E, num_fe_train, num_fe_test)

    #  Write the computed split to a file
    store_train_test_splits(output_path, train_E, train_E_false, test_E, test_E_false, split_id)


def compute_splits_parallel(G, output_path, owa=True, train_frac=0.51, num_fe_train=None, num_fe_test=None,
                            num_splits=10):
    """
    Computes in parallel the required number of train/test splits of input graph edges. Also generated the same number
    of train and test non-edge sets. The resulting sets of node pairs are written to different files. The resulting
    train edge set has the following properties: spans a graph (digraph) with a single connected (weakly connected)
    component and the same nodes as G.

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph with a single connected (weakly connected) component.
    output_path : string
        Indicates the path where data will be stored. Can include a name for all splits to share.
    owa : bool, optional
        Encodes the belief that the network respects or not the open world assumption. Default is True.
        If owa=True, train non-edges are sampled from the train graph only and can overlap with test edges.
        If owa=False, train non-edges are sampled from the full graph and cannot overlap with test edges.
    train_frac : float, optional
        The proportion of train edges w.r.t. the total number of edges in the input graph (range (0.0, 1.0]).
        Default is 0.51.
    num_fe_train : int, optional
        The number of train non-edges to generate. Default is None (same number as train edges).
    num_fe_test : int, optional
        The number of test non-edges to generate. Default is None (same number as test edges).
    num_splits : int, optional
        The number of train/test splits to generate. Default is 10.

    See also
    --------
    evalne.utils.split_train_test.split_train_test :
        This method used to split graph edges in train and test.
    evalne.utils.split_train_test.generate_false_edges_owa :
        Method used to generate non-edges if owa=True.
    evalne.utils.split_train_test.generate_false_edges_cwa :
        Method used to generate non-edges if owa=False.
    """
    # Compute the splits sequentially or in parallel
    backend = 'multiprocessing'
    path_func = delayed(_compute_one_split)
    Parallel(n_jobs=num_splits, verbose=True, backend=backend)(
        path_func(G, output_path, owa, train_frac, num_fe_train, num_fe_test, split) for split in range(num_splits))


def random_edge_sample(a, samp_frac=0.01, directed=False, sample_diag=False):
    """
    Samples uniformly at random positions from the input adjacency matrix. The samples are returned in two sets, one
    containing all edges sampled and one containing all non-edges.

    Parameters
    ----------
    a : sparse matrix
        A sparse adjacency matrix.
    samp_frac : float, optional
        A float in range [0,1] representing the fraction of all edge pairs to sample. Default is 0.01 (1%)
    directed : bool, optional
        A flag indicating if the adjacency matrix should be considered directed or undirected. If undirected,
        indices are obtained only from the lower triangle. Default is False.
    sample_diag : bool, optional
        If True elements from the main diagonal are also sampled (i.e. self loops). Else they are excluded from the
        sampling process. Default is False.

    Returns
    -------
    pos_e : ndarray
        Sampled node pairs that are edges.
    neg_e : ndarray
        Sampled node pairs that are non-edges.
    """
    if samp_frac > 1.0 or samp_frac < 0.0:
        raise ValueError("The samp_frac parameter is expected to be in range [0,1].")

    n = a.shape[0]
    num_samp = int(n ** 2 * samp_frac)

    # Generate a mask of sampled elements
    lin_ind = np.random.choice(n ** 2, size=num_samp, replace=False)
    j = np.floor(lin_ind * 1. / n).astype(int, copy=False)
    i = (lin_ind - j * n).astype(int, copy=False)
    v = np.ones(num_samp, dtype=bool)
    mask = csr_matrix((v, (i, j)), shape=(n, n))

    if directed:
        if not sample_diag:
            mask.setdiag(0)
            mask.eliminate_zeros()
    else:
        # For undir graphs we only look at the upper triangle
        if sample_diag:
            mask = triu(mask, k=0)
        else:
            mask = triu(mask, k=1)

    # Extract the linear indices of the nonzeros in a
    lin_indx_samp = np.ravel_multi_index(mask.nonzero(), (n, n))

    # All positive edges sampled in mask will stay in aux
    aux = mask.multiply(a)
    pos_e = np.array(aux.nonzero()).T

    # The rest of the lin indx not positive are negative
    lin_indx_ne = np.setdiff1d(lin_indx_samp, np.ravel_multi_index(aux.nonzero(), (n, n)))
    neg_e = np.array(np.unravel_index(lin_indx_ne, (n, n))).T

    # Return the node pairs
    return pos_e, neg_e


##################################################
#          Spanning tree computation             #
##################################################


def broder_alg(G, E):
    """
    Runs Andrei Broder's algorithm [1]_ to select uniformly at random a spanning tree of the input graph.
    The method works for directed and undirected networks. The edge directions in the resulting spanning tree
    are taken from the input edgelist (E). For node pairs where both edge directions exist, one is chosen at random.

    Parameters
    ----------
    G : graph
       A NetworkX graph or digraph.
    E : set
       A set of all directed or undirected edges in G.

    Returns
    -------
    train_E : set
       A set of edges of G that form the random spanning tree.

    See Also
    --------
    evalne.utils.split_train_test.wilson_alg :
        A more computationally efficient approach for selecting a spanning tree.

    References
    ----------
    .. [1] A. Broder, "Generating Random Spanning Trees", Proc. of the 30th Annual Symposium
           on Foundations of Computer Science, pp. 442--447, 1989.
    """
    # Create two partitions, S and T. Initially store all nodes in S.
    S = set(G.nodes)
    T = set()

    # Pick a random node as the "current node" and mark it as visited.
    current_node = random.sample(S, 1).pop()
    S.remove(current_node)
    T.add(current_node)

    # Perform random walk on the graph
    train_E = set()
    while S:
        if G.is_directed():
            neighbour_node = random.sample(list(G.successors(current_node)) + list(G.predecessors(current_node)), 1).pop()
        else:
            neighbour_node = random.sample(list(G.neighbors(current_node)), 1).pop()
        if neighbour_node not in T:
            S.remove(neighbour_node)
            T.add(neighbour_node)
            if random.random() < 0.5:
                if (current_node, neighbour_node) in E:
                    train_E.add((current_node, neighbour_node))
                else:
                    train_E.add((neighbour_node, current_node))
            else:
                if (neighbour_node, current_node) in E:
                    train_E.add((neighbour_node, current_node))
                else:
                    train_E.add((current_node, neighbour_node))
        current_node = neighbour_node

    # Return the set of edges constituting the spanning tree
    return train_E


def wilson_alg(G, E):
    """
    Runs Willson's algorithm [2]_ also known as loop erasing random walk to select uniformly at random a spanning tree
    of the input graph. The method works for directed and undirected networks [3]_. The edge directions in the resulting
    spanning tree are taken from the input edgelist (E). For node pairs where both edge directions exist, one is
    chosen at random.

    Parameters
    ----------
    G : graph
       A NetworkX graph or digraph.
    E : set
       A set of all directed or undirected edges in G.

    Returns
    -------
    train_E : set
       A set of edges of G that form the random spanning tree.

    See Also
    --------
    evalne.utils.split_train_test.broder_alg :
        A different approach for selecting a spanning tree that could be faster for specific graphs.

    References
    ----------
    .. [2] D. B. Wilson, "Generating Random Spanning Trees More Quickly than the Cover Time",
           In Proceedings of STOC, pp. 296--303, 1996.
    .. [3] J. G. Propp and D. B. Wilson, "How to Get a Perfectly Random Sample from a Generic
           Markov Chain and Generate a Random Spanning Tree of a Directed Graph",
           Journal of Algorithms 27, pp. 170--217, 1998.
    """
    # Stores the nodes which are part of the trees created by the LERW.
    intree = set()

    # A dictionary which works as a linked list and stores the spanning tree
    tree = dict()

    # Pick a random node as the root of the spanning tree and add it to intree
    # For undirected graphs this is the correct approach
    r = random.sample(G.nodes, 1).pop()
    intree.add(r)

    for node in G.nodes:
        i = node
        while i not in intree:
            # This random successor works for weighted and unweighted graphs because we just
            # want to select a bunch of edges from the graph, no matter what the weights are.
            if G.is_directed():
                tree[i] = random.sample(list(G.successors(i)) + list(G.predecessors(i)), 1).pop()
            else:
                tree[i] = random.sample(list(G.neighbors(i)), 1).pop()
            i = tree[i]
        i = node
        while i not in intree:
            intree.add(i)
            i = tree[i]

    # Create a set to store the train edges
    train_E = set()

    # This is only relevant for directed graphs to make the selection of edge direction equiprobable
    for e in set(zip(tree.keys(), tree.values())):
        if random.random() < 0.5:
            if e in E:
                train_E.add(e)
            else:
                train_E.add(e[::-1])
        else:
            if e[::-1] in E:
                train_E.add(e[::-1])
            else:
                train_E.add(e)

    # Return the edges of the random spanning tree
    return train_E


##################################################
#               Helper functions                 #
##################################################


def _sanity_check(G):
    """
    Helper function that checks if the input graphs contains a single connected component. Raises an error if not.

    Parameters
    ----------
    G : graph
       A NetworkX graph or digraph.

    Raises
    ------
    ValueError
        If the graph has more than one (weakly) connected component.
    """
    # Compute the number of connected components
    if G.is_directed():
        num_ccs = nx.number_weakly_connected_components(G)
    else:
        num_ccs = nx.number_connected_components(G)

    # Rise an error if more than one CC exists
    if num_ccs != 1:
        raise ValueError("Input graph should contain one (weakly) connected component. "
                         "This graph contains: " + str(num_ccs))


def store_edgelists(train_path, test_path, train_edges, test_edges):
    """
    Writes the train and test edgelists to files with the specified names.

    Parameters
    ----------
    train_path : string
       Indicates the path where the train data will be stored.
    test_path : string
       Indicates the path where the test data will be stored.
    train_edges : array_like
       Set of train edges.
    test_edges : array_like
       Set of test edges.
    """
    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])

    # Save the splits in different files
    np.savetxt(fname=train_path, X=train_edges, delimiter=',', fmt='%d')
    np.savetxt(fname=test_path, X=test_edges, delimiter=',', fmt='%d')


def store_train_test_splits(output_path, train_E, train_E_false, test_E, test_E_false, split_id=0):
    """
    Writes the sets of train and test edges and non-edges to separate files in the provided path. All files will share
    the same split number as an identifier. If any folder in the path does not exist, it will be generated.

    Parameters
    ----------
    output_path : string
       Indicates the path where data will be stored.
    train_E : set
       Set of train edges.
    train_E_false : set
       Set of train non-edges.
    test_E : set
       Set of test edges.
    test_E_false : set
       Set of test non-edges.
    split_id : int, optional
       The ID of train/test split to be stored. Default is 0.

    Returns
    -------
    filenames : list
        A list of strings, the names and paths of the 4 files where the train and test edges and non-edges are stored.

    See Also
    --------
    evalne.utils.preprocess.read_train_test : A function that can read the generated files.

    Notes
    -----
    This function generates 4 files under <output_path> named: 'trE_<split_id>.csv', 'negTrE_<split_id>.csv',
    'teE_<split_id>.csv', 'negTeE_<split_id>.csv' corresponding respectively to train edges train non-edges,
    test edges and test non-edges.

    Examples
    --------
    Writes some data to files under a folder named test:

    >>> train_E = ((1,2), (2,3))
    >>> train_E_false = ((-1,-2), (-2,-3))
    >>> test_E = ((10,20), (20,30))
    >>> test_E_false = ((-10,-20), (-20,-30))
    >>> store_train_test_splits("./test", train_E, train_E_false, test_E, test_E_false, split_id=0)
    ('./test/trE_0.csv', './test/negTrE_0.csv', './test/teE_0.csv', './test/negTeE_0.csv')

    """
    # Create path if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Convert edge-lists to numpy arrays
    train_E = np.array([list(edge_tuple) for edge_tuple in train_E])
    train_E_false = np.array([list(edge_tuple) for edge_tuple in train_E_false])
    test_E = np.array([list(edge_tuple) for edge_tuple in test_E])
    test_E_false = np.array([list(edge_tuple) for edge_tuple in test_E_false])

    filenames = (os.path.join(output_path, "trE_{}.csv".format(split_id)),
                 os.path.join(output_path, "negTrE_{}.csv".format(split_id)),
                 os.path.join(output_path, "teE_{}.csv".format(split_id)),
                 os.path.join(output_path, "negTeE_{}.csv".format(split_id)))

    # Save the splits in different files
    np.savetxt(fname=filenames[0], X=train_E, delimiter=',', fmt='%d')
    np.savetxt(fname=filenames[1], X=train_E_false, delimiter=',', fmt='%d')
    np.savetxt(fname=filenames[2], X=test_E, delimiter=',', fmt='%d')
    np.savetxt(fname=filenames[3], X=test_E_false, delimiter=',', fmt='%d')

    # Return the names given to the 4 files where data is stored
    return filenames


def check_overlap(filename, num_sets):
    """
    Shows the amount of overlap (shared elements) between sets of node pairs with different split IDs. This function
    assumes the existance of num_sets files sharing the same name and with split IDs starting from 0 to num_sets.

    Parameters
    ----------
    filename : string
       Indicates the path and name (without split ID) of the first set.
       The sets are assumed to have sequential split IDs starting at 0.
    num_sets : int
       The number of sets for which to check the overlap.

    See Also
    --------
    evalne.utils.split_train_test.store_train_test_splits :
        The filename is expected to follow the naming convention of this function without "_<split_id>.csv".

    Examples
    --------
    Checking the overlap of several files in the test folder containing train edges:

    >>> check_overlap("./test/tr_E", 2)
    Intersection of 2 sets is 10
    Union of 2 sets is 15
    Jaccard coefficient: 0.666666666667

    """
    # Load the first set and transform it into a list of tuples
    S = np.loadtxt(filename + "_0.csv", delimiter=',', dtype=int)
    S = set(map(tuple, S))

    # Initialize the intersection and union sets as all elements in first edge set
    intrs = S
    union = S

    # Sequentially add the rest of the sets and check overlap
    for i in range(num_sets - 1):
        # Read a new edge set
        S = np.loadtxt(filename + "_{}.csv".format(i + 1), delimiter=',', dtype=int)
        S = set(map(tuple, S))
        # Update intersection and union sets
        intrs = intrs & S
        union = union | S
        # Print the information on screen
        print("Intersection of {} sets is {}".format(i + 2, len(intrs)))
        print("Union of {} sets is {}".format(i + 2, len(union)))
        print("Jaccard coefficient: {}".format(len(intrs) / len(union)))
        print("")


def redges_false(train_E, test_E, output_path=None):
    """
    For directed graphs computes all non-edges (a->b) such that the opposite edge (a<-b) exists in the graph.
    It does this for both the train and test edge sets.

    Parameters
    ----------
    train_E : set
       The set of train edges.
    test_E : set
       The set of test edges.
    output_path : string, optional
        A path or file where to store the results. If None, results are not stored only returned. Default is None.

    Returns
    -------
    train_redges_false : set
        A set of train edges respecting the mentioned property.
    test_redges_false : set
        A set of test edges respecting the mentioned property.

    Notes
    -----
    These non-edges can be used to asses the performance of the embedding methods on predicting non-reciprocated edges.
    """
    # Reverse all train and test edges
    train_redges_false = set(tuple(reversed(edge_tuple)) for edge_tuple in train_E)
    test_redges_false = set(tuple(reversed(edge_tuple)) for edge_tuple in test_E)

    # Keep only the reversed edges which are not real train edges
    train_redges_false = train_redges_false - train_E

    # Keep only the test reversed edges which are not true edges in the graph
    test_redges_false = test_redges_false - train_E
    test_redges_false = test_redges_false - test_E

    if output_path is not None:
        # Store the reversed edges
        train_redges_false_np = np.array([list(edge_tuple) for edge_tuple in train_redges_false])
        test_redges_false_np = np.array([list(edge_tuple) for edge_tuple in test_redges_false])
        # Save the splits in different files
        np.savetxt(output_path, train_redges_false_np, delimiter=',', fmt='%d')
        np.savetxt(output_path, test_redges_false_np, delimiter=',', fmt='%d')

    # Return the computed sets
    return train_redges_false, test_redges_false
