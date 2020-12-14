#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

from __future__ import division

import random
import time

import networkx as nx

from evalne.utils import preprocess as pp
from evalne.utils import split_train_test as stt


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
    subgraph_size = 400
    train_frac = 0.5
    directed = True

    # Load a graph
    G = pp.load_graph(dataset_path + test_name, delimiter=",", comments='#', directed=directed)

    # Restrict graph to a sub-graph of 'subgraph_size' nodes
    SG = G.subgraph(random.sample(G.nodes, subgraph_size)).copy()

    # Preprocess the graph
    PSG, ids = pp.prep_graph(SG, relabel=True, del_self_loops=True, maincc=True)

    # Save the preprocessed graph
    pp.save_graph(PSG, output_path + "prep_graph.edgelist", delimiter=",")

    # Compute train/test splits
    start = time.time()
    train_stt, test_stt = stt.split_train_test(PSG, train_frac=train_frac)
    end = time.time() - start
    print("Exec time stt: {}".format(end))

    # Check that the train graph generated with stt has one single cc
    if directed:
        TG_stt = nx.DiGraph()
        TG_stt.add_edges_from(train_stt)
        print("Number of weakly CCs with stt: {}".format(nx.number_weakly_connected_components(TG_stt)))
    else:
        TG_stt = nx.Graph()
        TG_stt.add_edges_from(train_stt)
        print("Number of CCs with stt: {}".format(nx.number_connected_components(TG_stt)))
    print("Number train edges stt: {}".format(len(train_stt)))
    print("Number test edges stt: {}".format(len(test_stt)))
    print("Number of nodes in train graph: {}".format(len(TG_stt.nodes)))

    # Preprocess the graph
    PSG, ids = pp.prep_graph(SG, relabel=True, del_self_loops=True, maincc=False)

    # Compute train/test splits
    start = time.time()
    train_rstt, test_rstt = stt.rand_split_train_test(PSG, train_frac=train_frac)
    end = time.time() - start
    print("\nExec time rand_stt: {}".format(end))

    # Check that the train graph generated with rstt has one single cc
    if directed:
        TG_rstt = nx.DiGraph()
        TG_rstt.add_edges_from(train_rstt)
        print("Number of weakly CCs with rstt: {}".format(nx.number_weakly_connected_components(TG_rstt)))
    else:
        TG_rstt = nx.Graph()
        TG_rstt.add_edges_from(train_rstt)
        print("Number of CCs with rstt: {}".format(nx.number_connected_components(TG_rstt)))
    print("Number train edges rstt: {}".format(len(train_rstt)))
    print("Number test edges rstt: {}".format(len(test_rstt)))
    print("Number of nodes in train graph: {}".format(len(TG_rstt.nodes)))

    # Compute set of false edges
    # train_E_false, test_E_false = stt.generate_false_edges_owa(SG, train_E=train_E, test_E=test_E, num_fe_train=0,
    #                                                           num_fe_test=100)
    # train_E_false, test_E_false = stt.generate_false_edges_owa(G, train_E=train_E, test_E=test_E, num_fe_train=None,
    #                                                          num_fe_test=None, seed=99)

    # Store the edge splits generated
    # stt.store_train_test_splits('./', train_E, train_E_false, test_E, test_E_false, split_id=0)


if __name__ == "__main__":

    test()
    test_split()


