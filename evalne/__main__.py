#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

import os
import random
import time
from sys import argv

import numpy as np

from evalne.evaluation.evaluator import *
from evalne.preprocessing import preprocess as pp
from evalne.preprocessing import split_train_test as stt


def main():
    # Start timer
    start = time.time()

    # Simple check of input args
    if len(argv) != 2:
        print("Error: Wrong number of parameters!")
        print("Usage: python evalne <path_to_ini_file>")
        exit(-1)

    # Create evaluation setup and pass ini file
    setup = EvalSetup(argv[1])

    # Evaluate methods using the setup
    evaluate(setup)

    # Get execution time
    end = time.time() - start
    print("Precessed in {} seconds".format(str(end)))


def evaluate(setup):
    # Get input and output paths
    inpaths = setup.inpaths
    outpaths = setup.outpaths

    # Auxiliary variable to store the tabular results
    results = list()
    method_names = list()

    # Loop over all input networks
    for i in range(len(inpaths)):
        if setup.verbose:
            print('\nEvaluating {} network...'.format(setup.names[i]))
            print('-------------------------------------')

        # Create output path if needed
        if not os.path.exists(outpaths[i]):
            os.makedirs(outpaths[i])

        # Create training path if needed
        if setup.traintest_path is not None:
            if not os.path.exists(outpaths[i] + setup.traintest_path):
                os.makedirs(outpaths[i] + setup.traintest_path)

        # Create an evaluator
        nee = Evaluator(setup.embed_dim, setup.lp_model)

        # Load and preprocess the graph
        G = preprocess(setup, i)

        # For each repeat of the experiment generate new data splits
        for repeat in range(setup.exp_repeats):
            if setup.verbose:
                print('Repetition {} of experiment'.format(repeat))

            # Generate one train/test split
            train_E, train_E_false, test_E, test_E_false = \
                nee.traintest_split.compute_splits(G, train_frac=setup.train_frac,
                                                   fast_split=setup.fast_split,
                                                   owa=setup.owa,
                                                   num_fe_train=setup.num_fe_train,
                                                   num_fe_test=setup.num_fe_test,
                                                   seed=repeat, verbose=setup.verbose)

            # Store train/test splits
            if setup.traintest_path is not None:
                stt.store_train_test_splits(output_path=outpaths[i] + setup.traintest_path, train_E=train_E,
                                            train_E_false=train_E_false, test_E=test_E, test_E_false=test_E_false,
                                            split=repeat)

            # Evaluate baselines
            if setup.lp_baselines is not None:
                eval_baselines(setup, nee, i)

            # Evaluate other NE methods
            if setup.methods_opne is not None or setup.methods_other is not None:
                eval_other(setup, nee)

            # Store the results in tabular format
            if setup.scores is not None and setup.scores != 'all':
                names, res = get_scores(setup, nee)
                results.append(res)
                method_names.append(names)

            # Store the plots or complete output
            store_scores(setup, nee, i, repeat)

            # Reset the scoresheets
            nee.reset_results()

    if setup.scores is not None and setup.scores != 'all':
        store_tabular(setup, results, method_names)

    print("End of experiment")


def preprocess(setup, i):
    """
    Graph preprocessing rutine.
    """
    if setup.verbose:
        print('Preprocesing graph...')

    # Load a graph
    G = pp.load_graph(setup.inpaths[i], delimiter=setup.separators[i], comments=setup.comments[i],
                      directed=setup.directed[i])

    # Preprocess the graph
    G, ids = pp.prep_graph(G, relabel=setup.relabel, del_self_loops=setup.del_selfloops)

    if setup.prep_nw_name is not None:
        # Store preprocessed graph to a file
        pp.save_graph(G, output_path=setup.outpaths[i] + setup.prep_nw_name, delimiter=setup.delimiter,
                      write_stats=setup.write_stats)

    # Return the preprocessed graph
    return G


def eval_baselines(setup, nee, i):
    """
    Experiment to test the baselines.
    """
    if setup.verbose:
        print('Evaluating baselines...')

    # Set the baselines
    methods = setup.lp_baselines

    # Evaluate baseline methods
    if setup.directed[i]:
        if setup.verbose:
            print('Input {} network is directed. Running baselines for all neighbourhoods specified...'
                  .format(setup.names[i]))
        for neigh in setup.neighbourhood:
            nee.evaluate_baseline(methods=methods, neighbourhood=neigh)
    else:
        if setup.verbose:
            print('Input {} network is undirected. Running standard baselines...'.format(setup.names[i]))
        nee.evaluate_baseline(methods=methods)


def eval_other(setup, nee):
    """
    Experiment to test other embedding methods not integrated in the library.
    """
    if setup.verbose:
        print('Evaluating Embedding methods...')

    if setup.methods_other is not None:
        # Evaluate non OpenNE method
        # -------------------------------
        for i in range(len(setup.methods_other)):
            # Evaluate the method
            nee.evaluate_ne_cmd(method_name=setup.names_other[i], command=setup.methods_other[i],
                                edge_embedding_methods=setup.edge_embedding_methods,
                                input_delim=setup.input_delim_other[i], emb_delim=setup.output_delim_other[i],
                                tune_params=setup.tune_params_other[i], maximize=setup.maximize, verbose=setup.verbose)

    if setup.methods_opne is not None:
        # Evaluate methods from OpenNE
        # ----------------------------
        for i in range(len(setup.methods_opne)):
            command = setup.methods_opne[i] + " --input {} --output {} --representation-size {}"
            nee.evaluate_ne_cmd(method_name=setup.names_opne[i], command=command, input_delim=' ',
                                edge_embedding_methods=setup.edge_embedding_methods, emb_delim=' ',
                                tune_params=setup.tune_params_opne[i], maximize=setup.maximize, verbose=setup.verbose)


def store_scores(setup, nee, i, repeat):
    # Check the scoresheets
    results = nee.get_results()

    # Set the output file names
    filename = setup.outpaths[i] + setup.names[i] + "_rep_" + str(repeat)

    for result in results:
        if setup.curves is not None:
            # Plot all curves
            result.plot(setup.curves, filename + '_' + result.method)

        if setup.scores == 'all':
            result.save(filename + '.txt')


def store_tabular(setup, results, method_names):
    # Set output name
    filename = "./eval_output.txt"

    # Compute the average score over all repeats
    avg_res = np.mean(np.array(results).reshape(len(setup.names), setup.exp_repeats, len(method_names[0])), 1)
    avg_res = avg_res.transpose()

    f = open(filename, 'a+b')
    headder = '\nAlg.\\Network'
    for name in setup.names:
        headder += '\t' + name
    f.write((headder + '\n').encode())

    # Write the data to the file
    for i in range(avg_res.shape[0]):
        f.write(method_names[0][i].encode())
        for j in range(avg_res.shape[1]):
            f.write(('\t' + str(avg_res[i][j])).encode())
        f.write('\n'.encode())

    # Close the file
    f.close()


def get_scores(setup, nee):
    # Check the scoresheets
    results = nee.get_results()

    res = list()
    names = list()
    for result in results:
        func = getattr(result.test_scores, str(setup.scores))
        res.append(func())
        if result.params['edge_embed_method'] is not None:
            names.append(result.method + '-' + result.params['edge_embed_method'])
        else:
            names.append(result.method)

    return names, res


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
