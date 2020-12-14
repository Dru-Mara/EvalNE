#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# This example shows how to use the LPEvaluator class to analyse several algorithms on the input networks.
# Takes raw graph as input, preprocesses it, computes tr/te splits and runs the algorithms on these splits.
# In this case the parameters are provided by the user and no config file is used.
# Similar evaluations can be run for network reconstruction and sign prediction by importing the appropriate classes.

from __future__ import division

import os
import random

import numpy as np

from evalne.evaluation.evaluator import LPEvaluator
from evalne.evaluation.score import Scoresheet
from evalne.evaluation.split import LPEvalSplit
from evalne.utils import preprocess as pp

# NOTE: The example `as is`, only evaluates baseline methods. To evaluate the OpenNE methods, PRUNE and Metapath2vec
# these must be first installed. Then the correct paths must be set in the commands_other variable.
# Finally, the following parameter can be set to True.
run_other_methods = False


def main():
    # Initialize some parameters
    inpath = list()
    nw_names = ['test_network', 'blogCatalog']   # Stores the names of the networks evaluated
    inpath.append("../../evalne/tests/data/network.edgelist")
    # inpath.append("../../../data/BlogCatalog/blog.edgelist")
    outpath = "./output/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    directed = False		        # indicates if the graphs are directed or undirected
    delimiters = (',', '\t')		# indicates the delimiter in the original graph
    repeats = 2		                # number of time the experiment will be repeated

    # Create a scoresheet to store the results
    scoresheet = Scoresheet(tr_te='test')

    for i in range(len(inpath)):

        # Create folders for the evaluation results (one per input network)
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        # Load and preprocess the graph
        G = preprocess(inpath[i], outpath, delimiters[i], directed)

        # For each repeat of the experiment generate new data splits
        for repeat in range(repeats):
            print('Repetition {} of experiment'.format(repeat))

            # Generate one train/test split with default parameters
            traintest_split = LPEvalSplit()
            traintest_split.compute_splits(G, nw_name=nw_names[i], train_frac=0.8, split_id=repeat)

            trainvalid_split = LPEvalSplit()
            trainvalid_split.compute_splits(traintest_split.TG, nw_name=nw_names[i], train_frac=0.9, split_id=repeat)

            # Create an evaluator
            nee = LPEvaluator(traintest_split, trainvalid_split)

            # Evaluate baselines
            eval_baselines(nee, directed, scoresheet)

            # Evaluate other NE methods
            if run_other_methods:
                eval_other(nee, scoresheet)

    print("\nEvaluation results:")
    print("-------------------")

    # Print results averaged over exp repeats
    scoresheet.print_tabular(metric='auroc')

    # Write results averaged over exp repeats to a single file
    scoresheet.write_tabular(filename=os.path.join(outpath, 'eval_output.txt'), metric='auroc')

    # Store the Scoresheet object for later analysis
    scoresheet.write_pickle(os.path.join(outpath, 'eval.pkl'))

    print("Evaluation results are also stored in a folder named `output` in the current directory.")
    print("End of evaluation")


def preprocess(inpath, outpath, delimiter, directed):
    """
    Graph preprocessing routine.
    """
    print('Preprocessing graph...')

    # Load a graph
    G = pp.load_graph(inpath, delimiter=delimiter, comments='#', directed=directed)

    # Preprocess the graph
    G, ids = pp.prep_graph(G, relabel=True, del_self_loops=True)

    # Store preprocessed graph to a file
    pp.save_graph(G, output_path=outpath + "prep_graph.edgelist", delimiter=',', write_stats=True)

    # Return the preprocessed graph
    return G


def eval_baselines(nee, directed, scoresheet):
    """
    Experiment to test the baselines.
    """
    print('Evaluating baselines...')

    # Set the baselines
    methods = ['common_neighbours', 'jaccard_coefficient', 'cosine_similarity', 'lhn_index', 'topological_overlap',
               'adamic_adar_index', 'resource_allocation_index', 'preferential_attachment', 'random_prediction',
               'all_baselines']

    # Evaluate baseline methods
    for method in methods:
        if directed:
            result = nee.evaluate_baseline(method=method, neighbourhood="in")
            scoresheet.log_results(result)
            result = nee.evaluate_baseline(method=method, neighbourhood="out")
            scoresheet.log_results(result)
        else:
            result = nee.evaluate_baseline(method=method)
            scoresheet.log_results(result)


def eval_other(nee, scoresheet):
    """
    Experiment to test other embedding methods not integrated in the library.
    """
    print('Evaluating Embedding methods...')

    # Set edge embedding methods
    # Other options: 'weighted_l1', 'weighted_l2'
    edge_embedding_methods = ['average', 'hadamard']

    # Evaluate non OpenNE method
    # -------------------------------
    # Set the methods
    methods_other = ['PRUNE', 'metapath2vec++']

    # Set the method types
    method_type = ['ne', 'ne']

    # Set the commands
    commands_other = [
        'python ../../methods/PRUNE/src/main.py --inputgraph {} --output {} --dimension {}',
        '../../methods/metapath2vec/metapath2vec -min-count 1 -iter 20 -samples 100 -train {} -output {} -size {}']

    # Set delimiters for the in and out files required by the methods
    input_delim = [' ', ' ']
    output_delim = [',', ' ']

    for i in range(len(methods_other)):
        # Evaluate the method
        results = nee.evaluate_cmd(method_name=methods_other[i], method_type=method_type[i], command=commands_other[i],
                                   edge_embedding_methods=edge_embedding_methods,
                                   input_delim=input_delim[i], output_delim=output_delim[i])
        # Log the list of results
        scoresheet.log_results(results)

    # Evaluate methods from OpenNE
    # ----------------------------
    # Set the methods
    methods = ['node2vec', 'deepwalk', 'line']

    # Set the commands
    commands = [
        'python -m openne --method node2vec --graph-format edgelist --epochs 100 --number-walks 10',
        'python -m openne --method deepWalk --graph-format edgelist --epochs 100 --number-walks 10 --walk-length 80',
        'python -m openne --method line --graph-format edgelist --epochs 100']

    # Set parameters to be tuned
    tune_params = ['--p 0.5 1 --q 0.5 1', None, None]

    # For each method evaluate
    for i in range(len(methods)):
        command = commands[i] + " --input {} --output {} --representation-size {}"
        results = nee.evaluate_cmd(method_name=methods[i], method_type='ne', command=command,
                                   edge_embedding_methods=edge_embedding_methods, input_delim=' ', output_delim=' ',
                                   tune_params=tune_params[i])
        # Log the list of results
        scoresheet.log_results(results)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
