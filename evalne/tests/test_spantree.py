#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

from __future__ import division

import time

import networkx as nx

from evalne.utils import preprocess as pp
from evalne.utils import split_train_test as stt


def test_stt():
    # Variables
    dataset_path = "./data/"
    test_name = "network.edgelist"
    frac = 0.5

    # Load a graph
    G = pp.load_graph(dataset_path + test_name, delimiter=",", comments='#', directed=False)

    # Preprocess the graph for stt alg.
    SG, ids = pp.prep_graph(G, relabel=True, del_self_loops=True, maincc=True)

    # Split train/test using stt
    start = time.time()
    train_E, test_E = stt.split_train_test(SG, train_frac=frac)
    end1 = time.time() - start

    # Compute the false edges
    train_E_false, test_E_false = stt.generate_false_edges_owa(SG, train_E=train_E, test_E=test_E,
                                                               num_fe_train=None, num_fe_test=None)
    # Store data to file
    _ = stt.store_train_test_splits(dataset_path + "stt_frac_" + str(frac),
                                    train_E=train_E, train_E_false=train_E_false, test_E=test_E,
                                    test_E_false=test_E_false, split_id=0)

    # Split train/test using rstt
    start = time.time()
    tr_E, te_E = stt.rand_split_train_test(G, train_frac=frac)
    end2 = time.time() - start

    train_E, test_E, J, mp = pp.relabel_nodes(tr_E, te_E, G.is_directed())

    print("Number of nodes in G: {}".format(len(G.nodes())))
    print("Number of nodes in J: {}".format(len(J.nodes())))
    print("Are nodes in J sequential integers? {}".format(not len(set(J.nodes()) - set(range(len(J.nodes()))))))

    checks = list()
    queries = 200
    # Check if the mapping is correct
    for i in range(queries):
        ag = tr_E.pop()                 # a random element from train
        aj = (mp[ag[0]], mp[ag[1]])     # check what it maps to in J
        checks.append(aj in train_E)
        # print("Random tuple from G: {}".format(ag))
        # print("The tuple maps in J to: {}".format(aj))
        # print("Is that tuple in the new train?: {}".format(aj in train_E))

    print("For train edges out of {} samples, {} were in the relabeled train_E".format(queries, sum(checks)))

    checks = list()
    # Check if the mapping is correct
    for i in range(queries):
        ag = te_E.pop()                 # a random element from test
        aj = (mp[ag[0]], mp[ag[1]])     # check what it maps to in J
        checks.append(aj in test_E)
        # print("Random tuple from G: {}".format(ag))
        # print("The tuple maps in J to: {}".format(aj))
        # print("Is that tuple in the new train?: {}".format(aj in train_E))

    print("For test edges out of {} samples, {} were in the relabeled test_E".format(queries, sum(checks)))

    # Compute the false edges
    train_E_false, test_E_false = stt.generate_false_edges_owa(J, train_E=train_E, test_E=test_E,
                                                               num_fe_train=None, num_fe_test=None)
    # Store data to file
    _ = stt.store_train_test_splits(dataset_path + "rstt_frac_" + str(frac),
                                    train_E=train_E, train_E_false=train_E_false, test_E=test_E,
                                    test_E_false=test_E_false, split_id=0)


def test_split():
    # Variables
    dataset_path = "./data/"
    test_name = "network.edgelist"

    # Load a graph
    SG = pp.load_graph(dataset_path + test_name, delimiter=",", comments='#', directed=False)

    # Preprocess the graph
    SG, ids = pp.prep_graph(SG, relabel=True, del_self_loops=True)
    print("Number of CCs input: {}".format(nx.number_connected_components(SG)))

    # Store the edges in the graphs as a set E
    E = set(SG.edges())

    # Use LERW approach to get the ST
    start = time.time()
    train_lerw = stt.wilson_alg(SG, E)
    end1 = time.time() - start

    # Use BRO approach to get the ST
    start = time.time()
    train_bro = stt.broder_alg(SG, E)
    end2 = time.time() - start

    print("LERW time: {}".format(end1))
    print("Bro time: {}".format(end2))

    print("Num tr_e lerw: {}".format(len(train_lerw)))
    print("Num tr_e bro: {}".format(len(train_bro)))

    print("All tr_e in E for lerw?: {}".format(train_lerw - E))
    print("All tr_e in E for bro?: {}".format(train_bro - E))

    # Check that the graph generated with lerw has indeed one single cc
    TG_lerw = nx.Graph()
    TG_lerw.add_edges_from(train_lerw)
    print("Number of CCs with lerw: {}".format(nx.number_connected_components(TG_lerw)))

    # Check that the graph generated with broder algorithm has indeed one single cc
    TG_bro = nx.Graph()
    TG_bro.add_edges_from(train_bro)
    print("Number of CCs with lerw: {}".format(nx.number_connected_components(TG_bro)))


if __name__ == "__main__":

    # Run the tests
    test_split()
    test_stt()

