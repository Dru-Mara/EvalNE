#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# This example shows how to use the Evaluator class to analyse several algorithms on the input networks.
# Takes raw graph as input, preprocesses it, computes tr/te splits and runs the algorithms on these splits.
# In this case the parameters are provided by the user and no config file is used.

# NOTE: In order to run this experiment `as is` OpenNE is required.
# NOTE: The user must also check that the paths for `other methods` are correctly set.

from __future__ import division

import os
import random

import numpy as np

from evalne.evaluation import evaluator
from evalne.preprocessing import preprocess as pp


def main():
    # Initialize some parameters
    inpath = list()
    inpath.append("../evalne/tests/data/network.edgelist")
    # inpath.append("../../data/BlogCatalog/blog.edgelist")
    directeds = (False, False)		# indicates if the graphs are directed or undirected
    delimiters = (',', '\t')		# indicates the delimiter in the original graph
    repeats = 1		                # number of time the experiment will be repeated

    for i in range(len(inpath)):
        # Create output folders
        outpath = "./output/"
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        # Create folders for the train/test splits
        outpath = outpath + inpath[i].split("/")[-2] + "/"
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        # Create an evaluator
        nee = evaluator.Evaluator()

        # Load and preprocess the graph
        G = preprocess(inpath[i], outpath, delimiters[i], directeds[i])
        
        # For each repeat of the experiment generate new data splits
        for repeat in range(repeats):
            print('Repetition {} of experiment'.format(repeat))

            # Generate one train/test split with default parameters
            nee.traintest_split.compute_splits(G, train_frac=0.9, seed=repeat)

            # Evaluate baselines
            eval_baselines(nee, directeds[i])

            # Evaluate other NE methods
            eval_other(nee)

            # Check out the scores and store them to a file
            check_scores(nee, outpath, repeat)

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


def eval_baselines(nee, directed):
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


def eval_other(nee):
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

    # Set the commands
    commands_other = [
        'python ../../methods/PRUNE/src/main.py --inputgraph {} --output {} --dimension {}',
        '../../methods/metapath2vec/metapath2vec -min-count 1 -iter 20 -samples 100 -train {} -output {} -size {}']

    # Set delimiters for the in and out files required by the methods
    input_delim = [' ', ' ']
    output_delim = [',', ' ']

    for i in range(len(methods_other)):
        # Evaluate the method
        nee.evaluate_ne_cmd(method_name=methods_other[i], command=commands_other[i],
                            edge_embedding_methods=edge_embedding_methods,
                            input_delim=input_delim[i], emb_delim=output_delim[i],)

    # Evaluate methods from OpenNE
    # ----------------------------
    # Set the methods
    methods = ['node2vec', 'deepwalk', 'line']

    # Set the commands
    commands = [
        'python -m openne --method node2vec --graph-format edgelist --epochs 100 --p 0.25 --q 0.25 --number-walks 10',
        'python -m openne --method deepWalk --graph-format edgelist --epochs 100 --number-walks 10 --walk-length 80',
        'python -m openne --method line --graph-format edgelist --epochs 100']

    # Set parameters to be tuned
    tune_params = ['--p 0.5 1 --q 0.5 1', None, None]

    # For each method evaluate
    for i in range(len(methods)):
        command = commands[i] + " --input {} --output {} --representation-size {}"
        nee.evaluate_ne_cmd(method_name=methods[i], command=command, edge_embedding_methods=edge_embedding_methods,
                            input_delim=' ', emb_delim=' ', tune_params=tune_params[i])


def check_scores(nee, outpath, repeat):
    # Create a file for the results (one for each repetition of the exp)
    outfile = outpath + "eval_output" + "_rep_" + str(repeat) + ".txt"

    # Check the results
    results = nee.get_results()

    # Store the scoresheets
    for result in results:
        result.save(outfile)

    # Reset the scoresheets
    nee.reset_results()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
