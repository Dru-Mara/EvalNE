#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# This file provides simple methods to: (1) read and write graphs, embeddings, node similarities, ... to/from files and
# (2) preprocess graphs for the specific task of graph embedding. NetworkX graphs and digraphs are supported,
# multi-graphs are not.

from __future__ import division
from __future__ import print_function

import os
import collections
import logging

import networkx as nx
import numpy as np


##################################################
#                      I/O                       #
##################################################


def save_graph(G, output_path, delimiter=',', write_stats=True, write_weights=False, write_dir=True):
    """
    Saves a graph to a file as an edgelist or weighted edgelist. If write_stats is True the file will include a header
    containing basic graph statistics as provided by the get_stats function.

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    output_path : file or string
        File or filename to write. If a file is provided, it must be opened in 'wb' mode.
    delimiter : string, optional
        The string to use for separating values in the output file. Default is ','.
    write_stats : bool, optional
        Adds basic graph statistics to the file as a header or not. Default is True.
    write_weights : bool, optional
        If True data will be stored as weighted edgelist i.e. triplets (src, dst, weight), otherwise, as regular
        (src, dst) pairs. For unweighted graphs, setting this parameter to True will add weight 1 to all edges.
        Default is False.
    write_dir : bool, optional
        This parameter is only relevant for undirected graphs. If True, it forces the method to write both edge
        directions in the file i.e. (src, dst) and (dst, src). If False, only one direction is stored. Default is True.

    See Also
    --------
    evalne.utils.split_train_test.get_stats : Function used to compute simple statistics to add as file header.
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


def load_graph(input_path, delimiter=',', comments='#', directed=False, datatype=float):
    """
    Reads a directed or undirected edgelist and returns a graph. If the edgelist is weighted the graph will maintain
    those weights. Nodes are read as int and weights as float.

    Parameters
    ----------
    input_path : file or string
        File or filename to read.
    delimiter : string, optional
        The string used to separate values in the input file. Default is ','.
    comments : string, optional
        The character used to indicate the start of a comment. Default is '#'.
    directed : bool
        Indicates if the method should return a graph or a digraph.
    datatype : int or float, optional
        The type of the graph weights. Default is float.

    Returns
    -------
    G : graph
        A NetworkX graph or digraph.
    """
    if directed:
        G = nx.read_edgelist(input_path, delimiter=delimiter, comments=comments, create_using=nx.DiGraph,
                             nodetype=int, data=(("weight", datatype),))
    else:
        G = nx.read_edgelist(input_path, delimiter=delimiter, comments=comments, create_using=nx.Graph,
                             nodetype=int, data=(("weight", datatype),))

    if len(G.edges) == 0:
        raise ValueError("The graph is empty, the delimiter specified might be incorrect!")

    return G


def read_labels(input_path, delimiter=',', comments='#', idx_mapping=None):
    """
    Reads node labels from a file and returns them as a numpy ndarray where the first column contains nodeIDs and the
    second, node attributes. If idx_mapping is provided, the original indices are re-mapped to new indices.

    Parameters
    ----------
    input_path : file or string
        File or filename to read. File is assumed to contain in each line a nodeID and label pair.
    delimiter : string, optional
        The string used to separate values in the input file. Default is ','.
    comments : string, optional
        The character used to indicate the start of a comment. Default is '#'.
    idx_mapping : list of tuples
        A list of (OldNodeID, NewNodeID).

    Returns
    -------
    labels : ndarray
        A column vector of nodeID and label pairs.
    """
    # Read the lables from a file
    labels = np.loadtxt(input_path, delimiter=delimiter, comments=comments, dtype=float)

    # Remap them if necessary
    if idx_mapping is not None:
        res = list()
        # Make labels a dict
        lab_d = dict(labels)
        for (oldnid, newnid) in idx_mapping:
            res.append((newnid, lab_d.get(oldnid)))
        return np.array(res)
    else:
        return labels[labels[:, 0]]


def read_node_embeddings(input_path, nodes, embed_dim, delimiter=',', method_name=None):
    """
    Reads node embeddings from a file (see Notes for expected input file structure) and returns the results as a
    dictionary. Keys correspond to nodeIDs and values to vectors. File headers are inferred base on the expected
    number of embeddings and ignored.

    Parameters
    ----------
    input_path : file or string
        File or filename to read.
    nodes : array_like
        The network nodes for which embeddings are expected.
    embed_dim : int
        The expected dimensionality of the node embeddings.
    delimiter : string, optional
        The string used to separate values in the input file. Default is ','.
    method_name : string, optional
        A string indicating the name of the method being evaluated. If provided will be used when logging an error.
        Default is None.

    Returns
    -------
    X : dict
       A dictionary of {`nodeID`: embed_vect, `nodeID`: embed_vect, ...}. Keys are type string and values array_like.

    Raises
    ------
    ValueError
        If the number of embeddings in the file is less than len(nodes).
        If the input data dimensions are not correct.

    Notes
    -----
    The method accepts files where each line corresponds to the embedding of one node. The nodeID can be present as
    the first value in each line. If the nodeID is not present, embeddings are assumed to be written in the file in
    ascending nodeID order.
    """
    # Check how many rows are in the header
    if method_name is not None:
        emb_skiprows = infer_header(input_path, len(nodes), method_name)
    else:
        emb_skiprows = infer_header(input_path, len(nodes))

    # Read the embeddings
    X = np.genfromtxt(input_path, delimiter=delimiter, dtype=float, skip_header=emb_skiprows, autostrip=True)

    # If output is just a vector rise error
    if X.ndim == 1:
        raise ValueError('Error encountered while reading node embeddings. Check output delimiter of the method.')

    # If output is a matrix check its dimensions
    if X.shape[1] == embed_dim:
        # Assume embeddings given as matrix [X_0, X_1, ..., X_D] where rows correspond to sorted node id
        keys = map(str, sorted(nodes))
        X = dict(zip(keys, X))
    elif X.shape[1] == embed_dim + 1:
        # Assume first col is node id and rest are embedding features [id, X_0, X_1, ..., X_D]
        if method_name is not None:
            logging.warning('Output provided by method `{}` contains {} columns, {} expected! '
                            'Taking first column as nodeID...'.format(method_name, X.shape[1], embed_dim))
        else:
            logging.warning('Output of evaluated method contains {} columns, {} expected! '
                            'Taking first column as nodeID...'.format(X.shape[1], embed_dim))
        keys = map(str, np.array(X[:, 0], dtype=int))
        X = dict(zip(keys, X[:, 1:]))
    else:
        raise ValueError('Incorrect embedding dimension for the evaluated method! Expected: {} or {} Received: {}'
                         .format(embed_dim, embed_dim + 1, X.shape[1]))

    # Return the embeddings
    return X


def read_edge_embeddings(input_path, ebunch_len, embed_dim, delimiter=',', method_name=None):
    """
    Reads node-pair or edge embeddings from a file and returns the results as a matrix. Each line in the file is
    assumed to represent the embedding of one node-pair. File headers are inferred base on the expected number of
    embeddings and ignored.

    Parameters
    ----------
    input_path : file or string
        File or filename to read.
    ebunch_len : int
        The number of embeddings expected.
    embed_dim : int
        The expected dimensionality of the embeddings.
    delimiter : string, optional
        The string used to separate values in the input file. Default is ','.
    method_name : string, optional
        A string indicating the name of the method being evaluated. If provided will be used when logging an error.
        Default is None.

    Returns
    -------
    Y : ndarray
        A column vector containing embeddings as rows.

    Raises
    ------
    ValueError
        If the number of embeddings in the file is less than ebunch_len.
        If the input data dimensions are not correct.
    """
    # Check how many rows are in the header
    if method_name is not None:
        emb_skiprows = infer_header(input_path, ebunch_len, method_name)
    else:
        emb_skiprows = infer_header(input_path, ebunch_len)

    # Read the embeddings
    Y = np.genfromtxt(input_path, delimiter=delimiter, dtype=float, skip_header=emb_skiprows, autostrip=True)

    # Check embedding dimensions
    if Y.ndim == 1 and len(Y) != embed_dim or Y.ndim == 2 and Y.shape[1] != embed_dim:
        raise ValueError('Incorrect edge embedding dimension for method {}! Expected dims: ({},{}), Received dims: {}'
                         .format(method_name, ebunch_len, embed_dim, Y.shape))

    # Return the embeddings
    return Y


def read_predictions(input_path, ebunch_len, delimiter=',', method_name=None):
    """
    Reads node similarities from a file (see Notes for expected input file structure) and returns the results as a
    vector. File headers are inferred base on the expected number of predictions and ignored.

    Parameters
    ----------
    input_path : file or string
        File or filename to read.
    ebunch_len : int
        The number of predictions expected.
    delimiter : string, optional
        The string used to separate values in the input file. Default is ','.
    method_name : string, optional
        A string indicating the name of the method being evaluated. If provided will be used when logging an error.
        Default is None.

    Returns
    -------
    Y : ndarray
        A column vector containing node similarities.

    Raises
    ------
    ValueError
        If the number of predictions in the file is less than ebunch_len.
        If the input data dimensions are not correct.

    Notes
    -----
    The method accepts files where all similarities are given in a single line in the file or one per line. In both
    case, the expected length of these "vectors" is ebunch_len. For input files containing similarities in rows, the
    method expects the last element of each row to be the similarity. This is useful for methods that return triplets:
    src, dst, sim.
    """
    # Predictions could be returned as col or row vectors
    try:
        # Check how many rows are in the header
        if method_name is not None:
            emb_skiprows = infer_header(input_path, ebunch_len, method_name)
        else:
            emb_skiprows = infer_header(input_path, ebunch_len)
    except ValueError:
        # Predictions could have been returned as a single vector
        if method_name is not None:
            emb_skiprows = infer_header(input_path, 1, method_name)
        else:
            emb_skiprows = infer_header(input_path, 1)

    # Read the embeddings
    Y = np.genfromtxt(input_path, delimiter=delimiter, dtype=float, skip_header=emb_skiprows, autostrip=True)

    # Check embedding dimensions
    if Y.ndim == 0:
        if ebunch_len == 1:
            Y = [Y.item()]
        else:
            if method_name is not None:
                raise ValueError('Incorrect number of predictions for method `{}`! Expected: {}, Received: {}'
                                 .format(method_name, ebunch_len, 1))
            else:
                raise ValueError('Incorrect number of predictions for evaluated method! Expected: {}, Received: {}'
                                 .format(method_name, ebunch_len, 1))

    if Y.ndim == 1 and len(Y) != ebunch_len or Y.ndim == 2 and Y.shape[0] != ebunch_len:
        if method_name is not None:
            raise ValueError('Incorrect number of predictions for method `{}`! Expected: {}, Received: {}'
                             .format(method_name, ebunch_len, Y.shape))
        else:
            raise ValueError('Incorrect predictions dimension for evaluated method! Expected: {}, Received: {}'
                             .format(ebunch_len, Y.shape))

    if Y.ndim == 2:
        if method_name is not None:
            logging.warning('Output provided by method `{}` is two dimensional. Taking last column as predictions... '
                            .format(method_name))
        else:
            logging.warning('Output of evaluated method is two dimensional. Taking last column as predictions... ')
        return Y[:, -1]
    else:
        return Y


def read_train_test(input_path, split):
    """
    Reads the sets of train and test edges and non-edges from the given path that share the given split ID.
    The method assumes that these files follow the naming convention of store_train_test_splits().

    Parameters
    ----------
    input_path : string
        The path where the input data can be found.
    split : int
        The ID of the splits to be read.

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

    See Also
    --------
    evalne.utils.split_train_test.store_train_test_splits :
        The files in the provided input path are expected to follow the naming convention of this function.
     """
    # Generate file names
    filenames = (os.path.join(input_path, "trE_{}.csv".format(split)),
                 os.path.join(input_path, "negTrE_{}.csv".format(split)),
                 os.path.join(input_path, "teE_{}.csv".format(split)),
                 os.path.join(input_path, "negTeE_{}.csv".format(split)))

    # Read the data
    train_E = np.loadtxt(filenames[0], delimiter=',', dtype=int)
    train_E_false = np.loadtxt(filenames[1], delimiter=',', dtype=int)
    test_E = np.loadtxt(filenames[2], delimiter=',', dtype=int)
    test_E_false = np.loadtxt(filenames[3], delimiter=',', dtype=int)

    return train_E, train_E_false, test_E, test_E_false


##################################################
#                  Preprocess                    #
##################################################


def prep_graph(G, relabel=True, del_self_loops=True, maincc=True):
    """
    Preprocess a graph according to the input parameters.

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    relabel : bool, optional
        Determines if the nodes should be relabeled with consecutive integers 0...N. Default is True.
    del_self_loops : bool, optional
        Determines if self loops should be deleted. Default is True.
    maincc : bool, optional
        Determines if the graphs (digraphs) should be restricted to the main (weakly) connected component or not.
        Default is True.

    Returns
    -------
    G : graph
       A preprocessed NetworkX graph or digraph.
    Ids : list of tuples
       A list of (OldNodeID, NewNodeID). Returns None if relabel=False.

    Notes
    -----
    For some methods trying to embed graphs with several connected components might result in inconsistent behaviours.
    This is the main reason for setting maincc=True.
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
    """
    For given sets of train and test edges, the method returns relabeled sets with nodes being integers in 0...N.
    Additionally, the method returns a graph containing all edges in the train and test and nodes in 0...N.

    Parameters
    ----------
    train_E : set
        The set of train edges.
    test_E : set
        The set of test edges.
    directed: bool
        Indicates if the method should return a graph or a digraph.

    Returns
    -------
    train_E : set
        The set of train edges
    test_E : set
        The set of test edges
    G : graph
        A NetworkX graph or digraph with relabeled nodes containing the edges in train and test.
    mapping : dict
        A dictionary containing old node id's as key and new id's as values.
    """

    # Create the set of all edges
    E = train_E | test_E

    # Create a graph
    if directed:
        H = nx.DiGraph()
        H.add_edges_from(E)
    else:
        H = nx.Graph()
        H.add_edges_from(E)

    # Save the mapping and relabel the graph
    mapping = dict(zip(H.nodes(), range(len(H.nodes()))))
    nx.relabel_nodes(H, mapping, copy=False)

    # Generate the new train set
    tr_E = set()
    for (src, dst) in train_E:
        tr_E.add((mapping[src], mapping[dst]))

    # Generate the new test set
    te_E = set()
    for (src, dst) in test_E:
        te_E.add((mapping[src], mapping[dst]))

    # Return the results
    return tr_E, te_E, H, mapping


def prune_nodes(G, threshold):
    """
    Removes all nodes from the graph whose degree is strictly smaller that the threshold. This can result in a
    disconnected graph.

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    threshold : int
        All nodes with degree lower than this value will be removed from the graph.

    Returns
    -------
    H : graph
       A copy of the original graph or digraph with removed nodes.
    """
    # Create a copy of the original graph
    H = G.copy()

    # Loop over the node, degree pairs and remove the corresponding ones
    for node, deg in G.degree:
        if deg < threshold:
            H.remove_node(node)

    # Return the resulting graph
    return H


def get_redges_false(G, output_path=None):
    """
    For directed graphs returns a list of all non-edges for which the opposite edge exists in the graph.
    E.g. returns all pairs of non-edges (a -> b) such that (b -> a) exists in the graph.

    Parameters
    ----------
    G : graph
       A NetworkX digraph.
    output_path : string
        A path or file where to store the results.

    Returns
    -------
    redges_false : set
       A set of node pairs respecting the mentioned property.
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


##################################################
#                     Utils                      #
##################################################


def infer_header(input_path, expected_lines, method_name=None):
    """
    Method that infers the length of the header of a given file from the number of lines and of expected lines.

    Parameters
    ----------
    input_path : file or string
        File or filename to read.
    expected_lines : int
        Number of expected lines in the input file.
    method_name : string, optional
        A string indicating the name of the method being evaluated. If provided will be used when logging an error.
        Default is None.

    Returns
    -------
    header_len : int
        The length of the header.

    Raises
    ------
    ValueError
        If more lines are expected than those in the file.
    """
    # Autodetect header of input as (num_lines_in_input - expected_lines)
    num_lines = sum(1 for _ in open(input_path))
    header_len = num_lines - expected_lines

    if header_len < 0:
        raise ValueError('Exception, not enough lines in input file! Expected {} lines, obtained {}.'
                         .format(expected_lines, num_lines))
    elif header_len > 0:
        if method_name is not None:
            logging.warning('Output of method `{}` contains {} more lines than expected. Will consider them part '
                            'of the header and ignore them... Expected num_lines {}, obtained lines {}.'
                            .format(method_name, header_len, expected_lines, num_lines))
        else:
            logging.warning('Output contains {} more lines than expected. Will consider them part '
                            'of the header and ignore them... Expected num_lines {}, obtained lines {}.'
                            .format(header_len, expected_lines, num_lines))

    # Return the header length
    return header_len


def get_stats(G, output_path=None, all_stats=False):
    """
    Prints or stores some basic statistics about the graph. If an output path is provided the results are written in
    said file.

    Parameters
    ----------
    G : graph
        A NetworkX graph or digraph.
    output_path : file or string, optional
        File or filename to write. Default is None
    all_stats : bool, optional
        Sets if all stats or a small subset of them should be shown. Computing all stats can be very slow.
        Default is False.
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
            print("Directed graph")
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
            print("Undirected graph")
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
            f.write("# Directed graph".encode())
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
            f.write("# Undirected graph".encode())
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
