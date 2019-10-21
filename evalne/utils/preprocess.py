#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# This code provides simple methods to preprocess graphs for the specific task of graph embeddings.
# Only undirected Graphs and Digraphs are supported, multi-graphs are not.

from __future__ import division
from __future__ import print_function

import collections

import networkx as nx
import numpy as np


def load_graph(input_path, delimiter=',', comments='#', directed=False):
    r"""
    Loads a directed or undirected graph from an edgelist. If the edgelist is weighted the provided graph will
    maintain those weights. For undirected graphs edges are sorted (smallID, bigID)
    
    Parameters
    ----------
    input_path : file or string
       File or filename to read.
    delimiter : string, optional
       The string used to separate values. Default is comma.
    comments : string, optional
       The character used to indicate the start of a comment. Default is '#'.
    directed : bool
       Indicated if the graph is directed or undirected.
    
    Returns
    -------
    G : graph
       A NetworkX graph
    """
    # Load edgelist
    E = np.loadtxt(input_path, delimiter=delimiter, comments=comments, dtype=int)

    # Create graph from the edgelist
    if directed:
        G = nx.DiGraph()
    else:
        # Make sure edges are sorted (smallNodeID, bigNodeID) for undirected graphs
        E.sort(axis=1)
        G = nx.Graph()

    if E.shape[1] == 2:
        # Create an unweighted graph
        G.add_edges_from(E)
    else:
        # Create an weighted graph
        G.add_weighted_edges_from(E[:, :3])

    # Return the nx graph
    return G


def infer_header(input_path, expected_lines):
    # Autodetect header of input as (num_lines_in_input - expected_lines)
    num_lines = sum(1 for _ in open(input_path))
    header_len = num_lines - expected_lines

    if header_len < 0:
        raise ValueError('Exception, not enough lines in input file! Expected {} lines, obtained {}.'
                         .format(expected_lines, num_lines))
    return header_len


def read_embeddings(input_path, nodes, embed_dim, delimiter=','):
    r"""
    Method that reads a file containing node or edge embeddings, and returns the results as dictionary of:
    {ID, embed_vect}. The file header is inferred base on the expected number of embeddings.

    Parameters
    ----------
    input_path : file or string
       File or filename to read.
    delimiter : string, optional
       The string used to separate values. Default is comma.
    comments : string, optional
       The character used to indicate the start of a comment. Default is '#'.
    directed : bool
       Indicated if the graph is directed or undirected.

    Returns
    -------
    G : graph
       A NetworkX graph
    """
    emb_skiprows = infer_header(input_path, len(nodes))

    # Read the embeddings
    X = np.genfromtxt(input_path, delimiter=delimiter, dtype=float, skip_header=emb_skiprows, autostrip=True)

    if X.ndim == 1:
        raise ValueError('Error encountered while reading node embeddings. Check output delimiter of evaluated method.')

    if X.shape[1] == embed_dim:
        # Assume embeddings given as matrix [X_0, X_1, ..., X_D] where rows correspond to sorted node id
        keys = map(str, sorted(nodes))
        X = dict(zip(keys, X))
    elif X.shape[1] == embed_dim + 1:
        # Assume first col is node id and rest are embedding features [id, X_0, X_1, ..., X_D]
        keys = map(str, np.array(X[:, 0], dtype=int))
        X = dict(zip(keys, X[:, 1:]))
    else:
        raise ValueError('Incorrect embedding dimension for evaluated method! Expected: {} or {} Received: {}'
                         .format(embed_dim, embed_dim + 1, X.shape[1]))
    return X


def read_predictions(input_path, num_pred, delimiter=','):
    emb_skiprows = infer_header(input_path, num_pred)

    # Read the embeddings
    X = np.genfromtxt(input_path, delimiter=delimiter, dtype=float, skip_header=emb_skiprows, autostrip=True)

    if X.ndim != 1:
        # If output is a matrix, assume last column has predictions in the same order as the edgelist.
        X = X[:, -1]

    return X


def save_graph(G, output_path, delimiter=',', write_stats=True, write_weights=False, write_dir=True):
    r"""
    Saves a graph to a file as an edgelist of weighted edgelist. If the stats parameter is set to True the file
    will include several lines containing the same basic graph statistics as provided by the get_stats function.
    For undirected graphs, the method stores both directions of every edge.

    Parameters
    ----------
    G : graph
       A NetworkX graph
    output_path : file or string
       File or filename to write. If a file is provided, it must be
       opened in 'wb' mode.
    delimiter : string, optional
       The string used to separate values. Default is ','.
    write_stats : bool, optional
        Sets if graph statistics should be added to the edgelist or not. Default is True.
    write_weights : bool, optional
        If True data will be stored as weighted edgelist (e.g. triplets src, dst, weight) otherwise as normal edgelist.
        If the graph edges have no weight attribute and this parameter is set to True,
        a weight of 1 will be assigned to each edge. Default is False.
    write_dir : bool, optional
        This option is only relevant for undirected graphs. If False, the graph will be stored with a single
        direction of the edges. If True, both directions of edges will be stored. Default is True.
    """
    # Write the graph stats in the file if required
    if write_stats:
        get_stats(G, output_path)

    # Open the file where data should be stored
    f = open(output_path, 'a+b')

    # Write the graph to a file and use both edge directions if graph is undirected
    if G.is_directed():
        # Store edgelist
        if write_weights:
            J = nx.DiGraph()
            J.add_weighted_edges_from(G.edges.data('weight', 1))
            nx.write_weighted_edgelist(J, f, delimiter=delimiter)
        else:
            nx.write_edgelist(G, f, delimiter=delimiter, data=False)
    else:
        if write_dir:
            H = nx.to_directed(G)
            J = nx.DiGraph()
        else:
            H = G
            J = nx.DiGraph()
        # Store edgelist
        if write_weights:
            J.add_weighted_edges_from(H.edges.data('weight', 1))
            nx.write_weighted_edgelist(J, f, delimiter=delimiter)
        else:
            nx.write_edgelist(H, f, delimiter=delimiter, data=False)


def get_stats(G, output_path=None, all_stats=False):
    r"""
    Prints or stores some basic statistics about the graph commonly used in network embedding literature.
    If an output file path is provided the results are written in that file.
    
    Parameters
    ----------
    G : graph
        A NetworkX graph
    output_path : file or string
        File or filename to write.
    all_stats : bool
        Sets if all stats or a small subset of them should be shown.
    """
    # Compute the number of nodes and edges of the graph
    N = len(G.nodes)
    M = len(G.edges)

    # Compute average degree and deg1 and deg2 num nodes
    degs = np.array(G.degree)[:, 1]
    avgdeg = sum(degs)/N
    counts = collections.Counter(degs)
    degdict = collections.OrderedDict(sorted(counts.items()))
    deg1 = degdict.get(1, 0)
    deg2 = degdict.get(2, 0)

    if all_stats:
        x = np.log(np.array(degdict.keys()))    # degrees
        y = np.log(np.array(degdict.values()))  # frequencies
        # the power-law coef. is the slope of a linear moder fitted to the loglog data which has closed-form solution
        plawcoef = np.abs(np.cov(x, y) / np.var(x))[0, 1]
        cc = nx.average_clustering(G)
        dens = nx.density(G)
        if G.is_directed():
            diam = nx.diameter(G) if nx.is_strongly_connected(G) else float('inf')
        else:
            diam = nx.diameter(G)

    # Print or write to file the graph info
    if output_path is None:
        # Print some basic info about the graph
        if G.is_directed():
            num_ccs = nx.number_weakly_connected_components(G)
            Gcc = max(nx.weakly_connected_component_subgraphs(G), key=len)
            Ncc = len(Gcc.nodes)
            Mcc = len(Gcc.edges)
            print("Directed and unweighted graph")
            print("Num. nodes: {}".format(N))
            print("Num. edges: {}".format(M))
            print("Num. weakly connected components: {}".format(num_ccs))
            print("Num. nodes in largest weakly CC: {} ({} % of total)".format(Ncc, Ncc * 100.0 / N))
            print("Num. edges in largest weakly CC: {} ({} % of total)".format(Mcc, Mcc * 100.0 / M))
        else:
            num_ccs = nx.number_connected_components(G)
            Gcc = max(nx.connected_component_subgraphs(G), key=len)
            Ncc = len(Gcc.nodes)
            Mcc = len(Gcc.edges)
            print("Undirected and unweighted graph")
            print("Num. nodes: {}".format(N))
            print("Num. edges: {}".format(M))
            print("Num. connected components: {}".format(num_ccs))
            print("Num. nodes in largest weakly CC: {} ({} % of total)".format(Ncc, Ncc * 100.0 / N))
            print("Num. edges in largest weakly CC: {} ({} % of total)".format(Mcc, Mcc * 100.0 / M))
        if all_stats:
            print("Clustering coefficient: {}".format(cc))
            print("Diameter: {}".format(diam))
            print("Density: {}".format(dens))
            print("Power-law coefficient: {}".format(plawcoef))
        print("Avg. node degree: {}".format(avgdeg))
        print("Num. degree 1 nodes: {}".format(deg1))
        print("Num. degree 2 nodes: {}".format(deg2))
        print("Num. self loops: {}".format(G.number_of_selfloops()))
        print("")
    else:
        # Write the info to the provided file
        f = open(output_path, 'w+b')
        if G.is_directed():
            num_ccs = nx.number_weakly_connected_components(G)
            Gcc = max(nx.weakly_connected_component_subgraphs(G), key=len)
            Ncc = len(Gcc.nodes)
            Mcc = len(Gcc.edges)
            f.write("# Directed and unweighted graph".encode())
            f.write("\n# Num. nodes: {}".format(N).encode())
            f.write("\n# Num. edges: {}".format(M).encode())
            f.write("\n# Num. weakly connected components: {}".format(num_ccs).encode())
            f.write("\n# Num. nodes in largest weakly CC: {} ({} % of total)".format(Ncc, Ncc * 100.0 / N).encode())
            f.write("\n# Num. edges in largest weakly CC: {} ({} % of total)".format(Mcc, Mcc * 100.0 / M).encode())
        else:
            num_ccs = nx.number_connected_components(G)
            Gcc = max(nx.connected_component_subgraphs(G), key=len)
            Ncc = len(Gcc.nodes)
            Mcc = len(Gcc.edges)
            f.write("# Undirected and unweighted graph".encode())
            f.write("\n# Num. nodes: {}".format(N).encode())
            f.write("\n# Num. edges: {}".format(M).encode())
            f.write("\n# Num. connected components: {}".format(num_ccs).encode())
            f.write("\n# Num. nodes in largest CC: {} ({} % of total)".format(Ncc, Ncc * 100.0 / N).encode())
            f.write("\n# Num. edges in largest CC: {} ({} % of total)".format(Mcc, Mcc * 100.0 / M).encode())
        if all_stats:
            f.write("\n# Clustering coefficient: {}".format(cc).encode())
            f.write("\n# Diameter: {}".format(diam).encode())
            f.write("\n# Density: {}".format(dens).encode())
            f.write("\n# Power-law coefficient: {}".format(plawcoef).encode())
        f.write("\n# Avg. node degree: {}".format(avgdeg).encode())
        f.write("\n# Num. degree 1 nodes: {}".format(deg1).encode())
        f.write("\n# Num. degree 2 nodes: {}".format(deg2).encode())
        f.write("\n# Num. self loops: {}".format(G.number_of_selfloops()).encode())
        f.write("\n".encode())
        f.close()
        

def prep_graph(G, relabel=True, del_self_loops=True, maincc=True):
    r"""
    Preprocess a graphs according to the parameters provided.
    By default the (digraphs) graphs are restricted to their main (weakly) connected component.
    Trying to embed graphs with several CCs may cause some algorithms to put them infinitely far away.

    Parameters
    ----------
    G : graph
        A NetworkX graph
    relabel : bool, optional
        Determines if the nodes are relabeled with consecutive integers 0..N
    del_self_loops : bool, optional
        Determines if self loops should be deleted from the graph. Default is True.
    maincc : bool, optional
        Determines if the graphs should be restricted to the main connected component or not. Default is True.

    Returns
    -------
    G : graph
       A preprocessed NetworkX graph
    Ids : list of tuples
       A list of (OldNodeID, NewNodeID)
    """
    # Remove self loops
    if del_self_loops:
        G.remove_edges_from(G.selfloop_edges())

    # Restrict graph to its main connected component
    if maincc:
        if G.is_directed():
            Gcc = max(nx.weakly_connected_component_subgraphs(G), key=len)
        else:
            Gcc = max(nx.connected_component_subgraphs(G), key=len)
    else:
        Gcc = G

    # Relabel graph nodes in 0...N
    if relabel:
        Grl = nx.convert_node_labels_to_integers(Gcc, first_label=0, ordering='sorted')
        # A list of (oldNodeID, newNodeID)
        ids = list(zip(sorted(Gcc.nodes), sorted(Grl.nodes)))
        return Grl, ids
    else:
        return Gcc, None


def relabel_nodes(train_E, test_E, directed):
    r"""
    For given sets of train and test edges, the method returns relabeled sets with nodes being integers in 0...N.
    Additionally, the method return a graph G containing all edges in the train and test sets and the same node labels.

    Parameters
    ----------
    train_E : set
        The set of train edges.
    test_E : set
        The set of test edges.
    directed: bool
        Indicates if the returned graph should be directed or undirected.

    Returns
    -------
    train_false_E : set
        The set of false train edges
    test_false_E : set
        The set of false test edges
    G : graph
        A NetworkX graph with relabeled nodes containing the edges in train and test.
    mapping : dict
        A dictionary containing old node id's as key and new id's as values.
    """

    E = train_E | test_E

    if directed:
        H = nx.DiGraph()
        H.add_edges_from(E)
    else:
        H = nx.Graph()
        H.add_edges_from(E)

    mapping = dict(zip(H.nodes(), range(len(H.nodes()))))
    nx.relabel_nodes(H, mapping, copy=False)

    tr_E = set()
    for (src, dst) in train_E:
        tr_E.add((mapping[src], mapping[dst]))

    te_E = set()
    for (src, dst) in test_E:
        te_E.add((mapping[src], mapping[dst]))

    return tr_E, te_E, H, mapping


def get_redges_false(G, output_path=None):
    r"""
    For directed graphs returns a list of all non-edges for which the opposite edge exists in the graph.
    E.g. returns all pairs of non-edges (a -> b) such that (b -> a) exists in the graph.

    Parameters
    ----------
    G : graph
       A NetworkX graph
    output_path : string
        A path or file where to store the results

    Returns
    -------
    redges_false : set of tuples
       A set of edges respecting the mentioned property
    """
    if not G.is_directed():
        raise ValueError("Function only defined for directed graphs.")

    # Get graph edges as a set E
    E = set(G.edges)

    # Reverse all edges
    redges_false = set(tuple(reversed(edge_tuple)) for edge_tuple in E)

    # Keep only the reversed edges which are not real edges
    redges_false = redges_false - E

    # Save the data if a path is provided
    if output_path is not None:
        redges_false_np = np.array([list(edge_tuple) for edge_tuple in redges_false])
        np.savetxt(output_path, redges_false_np, delimiter=',', fmt='%d')

    # Return the sets of edges
    return redges_false


def read_train_test(filename, split):
    """
    Reads the sets of true and false train and test edges that share the given filename and split id.
    The method assumes that the splits have been generated with this tool.

    Parameters
    ----------
    filename : basestring
        The name shared by all the splits i.e. until '_trE_{}.csv'
    split : int
        The id of the splits to be read.

    Returns
    -------
    train_E : set
       Set of train edges
    train_E_false : set
       Set of train non-edges
    test_E : set
       Set of test edges
    test_E_false : set
       Set of test non-edges
    """
    # Generate file names
    filenames = ["{}_trE_{}.csv".format(filename, split), "{}_negTrE_{}.csv".format(filename, split),
                 "{}_teE_{}.csv".format(filename, split), "{}_negTeE_{}.csv".format(filename, split)]

    # Read the data
    train_E = np.loadtxt(filenames[0], delimiter=',', dtype=int)
    train_E_false = np.loadtxt(filenames[1], delimiter=',', dtype=int)
    test_E = np.loadtxt(filenames[2], delimiter=',', dtype=int)
    test_E_false = np.loadtxt(filenames[3], delimiter=',', dtype=int)

    return train_E, train_E_false, test_E, test_E_false


def prune_nodes(G, threshold):
    """
    Removes all nodes from the graph whose degree is strictly smaller that the threshold.
    This could result in a disconnected graph.

    Parameters
    ----------
    G : graph
        A NetworkX graph
    threshold : int
        All nodes with degree lower than this value will be removed from the graph.

    Returns
    -------
    H : graph
       A copy of the original graph with some nodes removed.
    """
    # Create a copy of the original graph
    H = G.copy()

    # Loop over the node, degree pairs and remove the corresponding ones
    for node, deg in G.degree:
        if deg < threshold:
            H.remove_node(node)

    # Return the resulting graph
    return H
