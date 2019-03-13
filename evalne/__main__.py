#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

import os
import random
import time
import numpy as np

from sys import argv

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
    # Set the random seed
    random.seed(setup.seed)
    np.random.seed(setup.seed)

    # Get input and output paths
    inpaths = setup.inpaths
    outpaths = setup.outpaths

    # Auxiliary variable to store the tabular results
    results = list()
    names = list()

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

        res = list()

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
                                                   split_id=repeat, verbose=setup.verbose)

            # Store train/test splits
            if setup.traintest_path is not None:
                stt.store_train_test_splits(output_path=outpaths[i] + setup.traintest_path, train_E=train_E,
                                            train_E_false=train_E_false, test_E=test_E, test_E_false=test_E_false,
                                            split_id=repeat)

            # Evaluate baselines
            if setup.lp_baselines is not None:
                eval_baselines(setup, nee, i)

            # Evaluate other NE methods
            if setup.methods_opne is not None or setup.methods_other is not None:
                eval_other(setup, nee)

            # Write method results to one file per experiment repeat
            # check_scores(setup, nee, repeat)

            # Store the results and average over the exp. repeats
            names, res = get_scores(setup, nee, res, i, repeat)

            # Reset the results list
            nee.reset_results()

        results.append(res)

    # Store the results
    if setup.scores is not None:
        write_output(setup, results, names)

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
                      write_stats=setup.write_stats, write_weights=False, write_dir=True)

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
            nee.evaluate_cmd(method_name=setup.names_other[i], method_type=setup.embtype_other[i],
                             command=setup.methods_other[i], edge_embedding_methods=setup.edge_embedding_methods,
                             input_delim=setup.input_delim_other[i], output_delim=setup.output_delim_other[i],
                             tune_params=setup.tune_params_other[i], maximize=setup.maximize,
                             write_weights=setup.write_weights_other[i], write_dir=setup.write_dir_other[i],
                             verbose=setup.verbose)

    if setup.methods_opne is not None:
        # Evaluate methods from OpenNE
        # ----------------------------
        for i in range(len(setup.methods_opne)):
            command = setup.methods_opne[i] + " --input {} --output {} --representation-size {}"
            nee.evaluate_cmd(method_name=setup.names_opne[i], method_type='ne', command=command, input_delim=' ',
                             edge_embedding_methods=setup.edge_embedding_methods, output_delim=' ',
                             tune_params=setup.tune_params_opne[i], maximize=setup.maximize, write_weights=False,
                             write_dir=True, verbose=setup.verbose)


def get_scores(setup, nee, res, nw_indx, repeat):
    # Check the method results
    results = nee.get_results()
    names = list()

    # Set the output file names
    filename = setup.outpaths[nw_indx] + setup.names[nw_indx] + "_rep_" + str(repeat)

    for i in range(len(results)):

        # Update the res variable with the results of the current repeat
        if len(res) != len(results):
            res.append(results[i].get_all(precatk_vals=setup.precatk_vals))
        else:
            aux = results[i].get_all(precatk_vals=setup.precatk_vals)
            res[i] = (res[i][0], [res[i][1][k] + aux[1][k] for k in range(len(aux[1]))])

        # Add the method names to a list
        if 'edge_embed_method' in results[i].params:
            names.append(results[i].method + '-' + results[i].params['edge_embed_method'])
        else:
            names.append(results[i].method)

        # Plot the curves if needed
        if setup.curves is not None:
            # Plot all curves
            results[i].plot(filename=filename + '_' + results[i].method, curve=setup.curves)

    return names, res


def write_output(setup, results, method_names):
    # Set output name
    filename = "./eval_output.txt"

    f = open(filename, 'a+b')
    if setup.scores == 'all':
        for i in range(len(results)):
            f.write(('\n\n{} Network'.format(setup.names[i])).encode())
            f.write('\n---------------------------'.encode())
            for j in range(len(results[i])):
                f.write(('\n{}:'.format(method_names[j])).encode())
                f.write('\n '.encode())
                for k in range(len(results[i][j][0])):
                    f.write((results[i][j][0][k] + ':  \t ' +
                             str(np.around(results[i][j][1][k] / setup.exp_repeats, 4)) + '\n ').encode())

    else:
        # Find the metric's index
        try:
            index = results[0][0][0].index(setup.scores)
        except ValueError:
            raise ValueError('The selected metric in `setup.scores` does not exist!')

        # Write the header
        header = '\nAlg.\\Network'
        for name in setup.names:
            header += '\t' + name
        f.write((header + '\n').encode())

        # Write the data to the file
        for j in range(len(results[0])):
            f.write((method_names[j] + ': ').encode())
            for i in range(len(results)):
                f.write(('\t' + str(np.around(results[i][j][1][index] / setup.exp_repeats, 4))).encode())
            f.write('\n'.encode())

    # Close the file
    f.close()


def check_scores(setup, nee, repeat):
    # Create a file for the results (one for each repetition of the exp)
    outfile = "./eval_output" + "_rep_" + str(repeat) + ".txt"

    # Check the results
    results = nee.get_results()

    # Store the results
    for result in results:
        result.save(outfile)


if __name__ == "__main__":
    main()
