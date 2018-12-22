#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

from __future__ import division

import random

from evalne.preprocessing import preprocess as pp
from evalne.preprocessing import split_train_test as stt


def test():
    # Variables
    dataset_path = "./data/"
    output_path = "./data/"
    test_name = "network.edgelist"

    # Load a graph
    G = pp.load_graph(dataset_path + test_name, delimiter=',', comments='#', directed=True)

    # Print some stats
    print("")
    print("Original graph stats:")
    print("-----------------------------------------")
    pp.get_stats(G)

    # Save the graph
    pp.save_graph(G, output_path + "orig_graph.edgelist", delimiter=",")

    # Load the saved graph
    G2 = pp.load_graph(output_path + "orig_graph.edgelist", delimiter=",", comments='#', directed=True)

    # Stats comparison
    print("Has the same stats after being loaded?:")
    print("-----------------------------------------")
    pp.get_stats(G2)

    # Preprocess the graph
    GP, ids = pp.prep_graph(G2, del_self_loops=False, relabel=True)

    print("Preprocessed graph stats (restricted to main cc):")
    print("-----------------------------------------")
    pp.get_stats(GP)

    pp.save_graph(GP, output_path + "prep_graph.edgelist", delimiter=",")

    print("Sample of 10 (oldNodeID, newNodeID):")
    print("-----------------------------------------")
    print(ids[0:10])

    pp.get_redges_false(GP, output_path + "redges_false.csv")


def test_split():
    # Variables
    dataset_path = "./data/"
    output_path = "./data/"
    test_name = "network.edgelist"
    subgraph_size = 1000

    # Load a graph
    G = pp.load_graph(dataset_path + test_name, delimiter="\t", comments='#', directed=True)

    # Restrict graph to a sub-graph of 'subgraph_size' nodes
    SG = G.subgraph(random.sample(G.nodes, subgraph_size)).copy()

    # Preprocess the graph
    SG, ids = pp.prep_graph(SG, relabel=True, del_self_loops=True)

    # Get stats of the preprocessed subgraph
    pp.save_graph(SG, output_path + "prep_graph.edgelist", delimiter=",")

    # Alternatively, train/test splits can be computed one at a time
    train_E, test_E = stt.split_train_test(SG, train_frac=0.51, seed=99)

    print(train_E)

    # Compute set of false edges
    train_E_false, test_E_false = stt.generate_false_edges_owa(SG, train_E=train_E, test_E=test_E, num_fe_train=None,
                                                               num_fe_test=None, seed=99)
    # train_E_false, test_E_false = stt.generate_false_edges_owa(G, train_E=train_E, test_E=test_E, num_fe_train=None,
    #                                                          num_fe_test=None, seed=99)


if __name__ == "__main__":

    # Run the tests
    # print(np.__version__)
    # print(sp.__version__)
    # print(nx.__version__)
    # print(sk.__version__)
    test()
    test_split()


