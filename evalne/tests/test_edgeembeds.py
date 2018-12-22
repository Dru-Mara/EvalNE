#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

import timeit

import numpy as np

from evalne.evaluation import edge_embeddings


def test():

    a = {'1': np.array([0, 0, 0, 0]), '2': np.array([2, 2, 2, 2]), '3': np.array([1, 1, -1, -1])}
    e = ((2, 1), (1, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2))
    ee = edge_embeddings.compute_edge_embeddings(a, e, "average")
    print('Input node embeddings:')
    print(a)
    print('Ebunch:')
    print(e)
    print('Output edge embeddings:')
    print(ee)


def time_test():

    # Create a dictionary simulating the node embeddings
    keys = map(str, range(100))
    vals = np.random.randn(100, 10)
    d = dict(zip(keys, vals))

    # Create set of edges
    num_edges = 1000000
    edges = list(zip(np.random.randint(0, 100, num_edges), np.random.randint(0, 100, num_edges)))

    start = timeit.default_timer()
    res = edge_embeddings.compute_edge_embeddings(d, edges, "average")
    end = timeit.default_timer() - start

    print("Processed in: {}".format(end))


if __name__ == "__main__":

    # Check cmd args
    # if len(argv) != 4:
    #     print "ERROR: wrong number of parameters"
    #     print "Usage: test_baselines.py <edgelist_path> <directed> <method>"
    #     print "Example: test_baselines.py ../../../data/Facebook-wallposts/FB-subs2-prep.edgelist, true, 'cn'"
    #     exit(-1)
    test()
    time_test()
