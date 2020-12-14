#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

from __future__ import division

import random
from time import time

import numpy as np

from evalne.evaluation import evaluator
from evalne.evaluation import score
from evalne.evaluation import split
from evalne.methods import katz
from evalne.utils import preprocess as pp


# TODO: there are big differences between katz exact and approx. Exact probably is wrong.


def test_katz(nee):
    # Evaluate exact katz implementation
    exact = katz.Katz(nee.traintest_split.TG)
    train_pred = exact.predict(nee.traintest_split.train_edges)
    test_pred = exact.predict(nee.traintest_split.test_edges)
    ms = score.Results(method='Katz', params=exact.get_params(),
                       train_pred=train_pred, train_labels=nee.traintest_split.train_labels,
                       test_pred=test_pred, test_labels=nee.traintest_split.test_labels)
    ms.pretty_print(results='test', precatk_vals=[2, 4, 6, 10, 100, 1000])
    # ms.plot()

    # # Evaluate approx katz implementation
    # approx = katz.KatzApprox(nee.traintest_split.TG)
    # train_pred = approx.fit_predict(nee.traintest_split.train_edges)
    # test_pred = approx.fit_predict(nee.traintest_split.test_edges)
    # ms = score.Results(method='Katz', params=approx.get_params(),
    #                    train_pred=train_pred, train_labels=nee.traintest_split.train_labels,
    #                    test_pred=test_pred, test_labels=nee.traintest_split.test_labels)
    # ms.pretty_print(results='test', precatk_vals=[2, 4, 6, 10, 100, 1000])
    # # ms.plot()


def test_baselines(nee, directed):
    """
    Experiment to test the baselines.
    """
    print('Evaluating baselines...')

    # Set the baselines
    methods = ['random_prediction', 'common_neighbours', 'jaccard_coefficient', 'adamic_adar_index',
               'preferential_attachment', 'resource_allocation_index']

    # Results list
    results = list()

    # Evaluate baseline methods
    for method in methods:
        if directed:
            results.append(nee.evaluate_baseline(method=method, neighbourhood="in"))
            results.append(nee.evaluate_baseline(method=method, neighbourhood="out"))
        else:
            results.append(nee.evaluate_baseline(method=method))

    for result in results:
        result.pretty_print()

    results[0].save_predictions('predictions.txt')


def run_test():

    random.seed(42)
    np.random.seed(42)

    # Set some variables
    filename = "./data/network.edgelist"
    directed = False

    # Load the test graph
    G = pp.load_graph(filename, delimiter=",", comments='#', directed=directed)
    G, ids = pp.prep_graph(G)

    # Print some stars about the graph
    pp.get_stats(G)

    # Generate one train/test split with all edges in train set
    start = time()
    traintest_split = split.EvalSplit()
    traintest_split.compute_splits(G, train_frac=0.9)
    end = time() - start
    print("\nSplits computed in {} sec".format(end))

    # Create an evaluator
    nee = evaluator.LPEvaluator(traintest_split)

    # Test baselines
    start = time()
    test_baselines(nee, directed)
    end = time() - start
    print("\nBaselines computed in {} sec".format(end))

    # Test Katz
    start = time()
    test_katz(nee)
    end = time() - start
    print("\nKatz computed in {} sec".format(end))


if __name__ == "__main__":
    run_test()

