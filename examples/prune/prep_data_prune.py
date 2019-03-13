#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# This code preprocessed the Facebook wall post and the Webspam datasets in order to produce edgelists
# which can be then used to replicate the paper experiments using EvalNE.

from __future__ import division

import os
import networkx as nx

from sys import argv
from evalne.preprocessing import preprocess as pp


def main():
    # Check cmd args
    if len(argv) != 3:
        print("ERROR: wrong number of parameters")
        print("Usage: prep_data_prune.py <facebook_path> <webspam_path>")
        exit(-1)

    # Extract the dataset names and paths
    fb_path, fb_name = os.path.split(argv[1])
    ws_path, ws_name = os.path.split(argv[2])

    # Preprocess FB graph
    G1 = prep_fb(argv[1])

    # Store FB graph to a file
    pp.save_graph(G1, output_path=fb_path + "/prep_graph_slfloops.edgelist", delimiter=',', write_stats=True)

    # Preprocess WS graph
    G2 = prep_ws(argv[2])

    # Store preprocessed graph to a file
    pp.save_graph(G2, output_path=ws_path + "/prep_graph_slfloops.edgelist", delimiter=',', write_stats=True)

    print("Preprocessing finished.")


def prep_fb(inpath):
    """
    Preprocess facebook wall post graph.
    """
    # Load a graph
    G = pp.load_graph(inpath, delimiter='\t', comments='#', directed=True)

    # The FB graph is stores as destination, origin so needs to be reversed
    G = G.reverse()

    # Preprocess the graph
    G, ids = pp.prep_graph(G, relabel=True, del_self_loops=False)

    # Return the preprocessed graph
    return G


def prep_ws(inpath):
    """
    Preprocess web spam graph.
    """
    # Create an empty digraph
    G = nx.DiGraph()

    # Read the file and create the graph
    src = 0
    f = open(inpath, 'r')
    for line in f:
        if src != 0:
            arr = line.split()
            for dst in arr:
                dst_id = int(dst.split(':')[0])
                # We consider the graph unweighted
                G.add_edge(src, dst_id)
        src += 1
    # G.add_node(src-2)

    # Preprocess the graph
    G, ids = pp.prep_graph(G, relabel=True, del_self_loops=False)

    # Return the preprocessed graph
    return G


if __name__ == "__main__":
    main()