#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

from __future__ import division

from time import time

from evalne.evaluation import evaluator
from evalne.evaluation import score
from evalne.methods import katz
from evalne.preprocessing import preprocess as pp


# TODO: there are big differences between katz exact and approx. Exact probably is wrong.


def test_katz(nee):
    # Evaluate exact katz implementation
    exact = katz.Katz(nee.traintest_split.TG)
    train_pred = exact.predict(nee.traintest_split.train_edges)
    test_pred = exact.predict(nee.traintest_split.test_edges)
    ms = score.Results(method='Katz', params={},
                       train_pred=train_pred, train_labels=nee.traintest_split.train_labels,
                       test_pred=test_pred, test_labels=nee.traintest_split.test_labels)
    ms.pretty_print(results='test')
    # ms.plot()

    # Evaluate approx katz implementation
    approx = katz.KatzApprox(nee.traintest_split.TG)
    train_pred = approx.fit_predict(nee.traintest_split.train_edges)
    test_pred = approx.fit_predict(nee.traintest_split.test_edges)
    ms = score.Results(method='Katz', params={},
                       train_pred=train_pred, train_labels=nee.traintest_split.train_labels,
                       test_pred=test_pred, test_labels=nee.traintest_split.test_labels)
    ms.pretty_print(results='test')
    # ms.plot()


def test_baselines(nee, directed):
    """
    Experiment to test the baselines.
    """
    print('Evaluating baselines...')

    # Set the baselines
    methods = ['random_prediction', 'common_neighbours', 'jaccard_coefficient', 'adamic_adar_index',
               'preferential_attachment', 'resource_allocation_index']

    # Evaluate baseline methods
    if directed:
        nee.evaluate_baseline(methods=methods, neighbourhood="in")
        nee.evaluate_baseline(methods=methods, neighbourhood="out")
    else:
        nee.evaluate_baseline(methods=methods)

    # results = nee.get_results()
    # for result in results:
    #    print(result.test_scores.auroc())


def run_test():

    # Set some variables
    filename = "./data/network.edgelist"
    directed = False

    # Load the test graph
    G = pp.load_graph(filename, delimiter=",", comments='#', directed=directed)

    # Print some stars about the graph
    pp.get_stats(G)

    # Create an evaluator
    nee = evaluator.Evaluator()

    # Generate one train/test split with all edges in train set
    start = time()
    train_E, train_E_false, test_E, test_E_false = nee.traintest_split.compute_splits(G, train_frac=0.9)
    # nee.traintest_split.read_splits('./data/data', 0, directed, verbose=False)
    end = time() - start
    print("\nSplits computed in {} sec".format(end))

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

