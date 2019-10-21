#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

import os
import random
import time
from datetime import datetime
from datetime import timedelta
import numpy as np
import logging
import networkx as nx

from tqdm import tqdm
from sys import argv
from evalne.evaluation.evaluator import LPEvaluator
from evalne.evaluation.evaluator import NREvaluator
from evalne.evaluation.evaluator import NCEvaluator
from evalne.evaluation.pipeline import EvalSetup
from evalne.evaluation.score import Scoresheet
from evalne.evaluation.split import EvalSplit
from evalne.utils import preprocess as pp
from evalne.utils import split_train_test as stt


def main():
    # Start timer
    start = time.time()

    # Simple check of input args
    if len(argv) != 2:
        print("Error: Wrong number of parameters!")
        print("Usage: python -m evalne <path_to_ini_file>")
        exit(-1)

    # Create evaluation setup and pass ini file
    setup = EvalSetup(argv[1])

    # Evaluate
    evaluate(setup)

    # Get execution time
    end = time.time() - start
    print("Evaluation finished in: {} ({:.2f} sec.)".format(str(timedelta(seconds=round(end))), end))


def evaluate(setup):
    # Set the random seed
    random.seed(setup.seed)
    np.random.seed(setup.seed)

    # Get input and output paths
    inpaths = setup.inpaths
    filename = '{}_eval_{}'.format(setup.task, datetime.now().strftime("%m%d_%H%M"))
    outpath = os.path.join(os.getcwd(), filename)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Logging configuration (file opened in append mode)
    logging.basicConfig(filename=os.path.join(outpath, 'eval.log'), format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%d-%m-%y %H:%M:%S', level=logging.INFO)
    logging.info('Evaluation start')

    # Create a Scoresheet object to store all results
    if setup.task == 'nr':
        scoresheet = Scoresheet(tr_te='train', precatk_vals=setup.precatk_vals)
    else:
        scoresheet = Scoresheet(tr_te='test', precatk_vals=setup.precatk_vals)

    # Initialize some variables
    edge_split_time = list()
    repeats = setup.lp_num_edge_splits if setup.task == 'lp' else 1
    t = tqdm(total=len(inpaths) * repeats)
    t.set_description(desc='Progress on {} task'.format(setup.task))

    # Loop over all input networks
    for i in range(len(inpaths)):
        logging.info('====== Evaluating {} network ======'.format(setup.names[i]))
        print('\nEvaluating {} network...'.format(setup.names[i]))
        print('=====================================')

        # Create path to store info per network if needed
        nw_outpath = os.path.join(outpath, setup.names[i])
        if setup.save_prep_nw or setup.curves != '':
            if not os.path.exists(nw_outpath):
                os.makedirs(nw_outpath)

        # Load and preprocess the graph
        G = preprocess(setup, nw_outpath, i)
        # labels = read_labels(setup.labelpath[i], ids)

        # For each repeat of the experiment generate new edge splits
        for repeat in range(repeats):
            logging.info('------ Repetition {} of experiment ------'.format(repeat))
            print('\nRepetition {} of experiment...'.format(repeat))
            print('-------------------------------------')

            traintest_split = EvalSplit()
            trainvalid_split = EvalSplit()
            split_time = time.time()
            if setup.task == 'lp':
                # For LP compute train/test and train/valid splits
                traintest_split.compute_splits(G, nw_name=setup.names[i], train_frac=setup.traintest_frac,
                                               split_alg=setup.split_alg, owa=setup.owa,
                                               fe_ratio=setup.fe_ratio, split_id=repeat, verbose=setup.verbose)
                trainvalid_split.compute_splits(traintest_split.TG, nw_name=setup.names[i],
                                                train_frac=setup.trainvalid_frac, split_alg=setup.split_alg,
                                                owa=setup.owa, fe_ratio=setup.fe_ratio, split_id=repeat,
                                                verbose=setup.verbose)
                # Create an LP evaluator
                nee = LPEvaluator(traintest_split, trainvalid_split, setup.embed_dim, setup.lp_model)

            elif setup.task == 'nr':
                # For NR set TG = G no train/valid split needed and get random subset of true and false edges for pred
                pos_e, neg_e = stt.random_edge_sample(nx.adj_matrix(G), setup.nr_edge_samp_frac, nx.is_directed(G))
                if len(pos_e) == 0:
                    logging.error('Sampling fraction {} on {} network returned 0 positive edges. Skipping evaluation...'
                                  .format(setup.nr_edge_samp_frac, setup.names[i]))
                    break
                traintest_split.set_splits(train_E=pos_e, train_E_false=neg_e, test_E=None, test_E_false=None,
                                           directed=nx.is_directed(G), nw_name=setup.names[i], TG=G)
                # Create an NR evaluator
                nee = NREvaluator(traintest_split, setup.embed_dim, setup.lp_model)
            else:
                # For NC no train test split needed, only train valid
                traintest_split.set_splits(train_E=G.edges, train_E_false=None, test_E=None, test_E_false=None,
                                           directed=nx.is_directed(G), nw_name=setup.names[i], TG=G)
                trainvalid_split.compute_splits(G, nw_name=setup.names[i],
                                                train_frac=setup.trainvalid_frac, split_alg=setup.split_alg,
                                                owa=setup.owa, fe_ratio=setup.fe_ratio, split_id=repeat,
                                                verbose=setup.verbose)
                # Create an NC evaluator
                nee = NCEvaluator(trainvalid_split, setup.embed_dim, setup.lp_model)

            edge_split_time.append(time.time() - split_time)

            # Evaluate baselines
            if setup.lp_baselines is not None and setup.task != 'nc':
                eval_baselines(setup, nee, i, scoresheet, repeat, nw_outpath)

            # Evaluate other NE methods
            if setup.methods_opne is not None or setup.methods_other is not None:
                eval_other(setup, nee, i, scoresheet, repeat, nw_outpath)

            # Update progress bar
            t.update(1)

    # Store the results
    if setup.scores is not None:
        if setup.scores == 'all':
            scoresheet.write_all(filename=os.path.join(outpath, 'eval_output.txt'))
        else:
            scoresheet.write_tabular(filename=os.path.join(outpath, 'eval_output.txt'), metric=setup.scores)
            scoresheet.write_tabular(filename=os.path.join(outpath, 'eval_output.txt'), metric='eval_time')
        scoresheet.write_pickle(os.path.join(outpath, 'eval.pkl'))

    # Close progress bar
    t.close()
    print('Average edge split times per dataset:')
    print(setup.names)
    print(np.array(edge_split_time).reshape(-1, repeats).mean(axis=1))
    logging.info('Evaluation end\n\n')


def preprocess(setup, nw_outpath, i):
    """
    Graph preprocessing routine.
    """
    print('Preprocessing graph...')

    # Load a graph
    if setup.directed:
        G = nx.read_edgelist(setup.inpaths[i], delimiter=setup.separators[i], comments=setup.comments[i],
                             create_using=nx.DiGraph, nodetype=int)
    else:
        G = nx.read_edgelist(setup.inpaths[i], delimiter=setup.separators[i], comments=setup.comments[i], nodetype=int)

    # Preprocess the graph
    if setup.task == 'lp' and setup.split_alg == 'random':
        G, ids = pp.prep_graph(G, relabel=setup.relabel, del_self_loops=setup.del_selfloops, maincc=False)
    else:
        G, ids = pp.prep_graph(G, relabel=setup.relabel, del_self_loops=setup.del_selfloops)

    # Save preprocessed graph to a file
    if setup.save_prep_nw:
        pp.save_graph(G, output_path=os.path.join(nw_outpath, 'prep_nw.edgelist'), delimiter=setup.delimiter,
                      write_stats=setup.write_stats, write_weights=False, write_dir=True)

    # Return the preprocessed graph
    return G


def eval_baselines(setup, nee, i, scoresheet, repeat, nw_outpath):
    """
    Experiment to test the baselines.
    """
    print('Evaluating baselines...')

    for method in setup.lp_baselines:
        try:
            # Evaluate baseline methods
            if setup.directed:
                print('Input {} network is directed. Running baseline for all neighbourhoods specified...'
                      .format(setup.names[i]))
                for neigh in setup.neighbourhood:
                    result = nee.evaluate_baseline(method=method, neighbourhood=neigh)
                    scoresheet.log_results(result)
                    # Plot the curves if needed
                    if setup.curves is not None:
                        result.plot(filename=os.path.join(nw_outpath, '{}_rep_{}'.format(result.method, repeat)),
                                    curve=setup.curves)

            else:
                print('Input {} network is undirected. Running standard baselines...'.format(setup.names[i]))
                result = nee.evaluate_baseline(method=method)
                scoresheet.log_results(result)
                # Plot the curves if needed
                if setup.curves is not None:
                    result.plot(filename=os.path.join(nw_outpath, '{}_rep_{}'.format(result.method, repeat)),
                                curve=setup.curves)
        except AttributeError as e:
            logging.exception('Exception occurred while evaluating method `{}` on `{}` network.'
                              .format(method, setup.names[i]))


def eval_other(setup, nee, i, scoresheet, repeat, nw_outpath):
    """
    Experiment to test other embedding methods not integrated in the library.
    """
    print('Evaluating Embedding methods...')

    if setup.methods_other is not None:
        # Evaluate non OpenNE method
        # -------------------------------
        for j in range(len(setup.methods_other)):
            try:
                # Evaluate the method
                results = nee.evaluate_cmd(method_name=setup.names_other[j], method_type=setup.embtype_other[j],
                                           command=setup.methods_other[j],
                                           edge_embedding_methods=setup.edge_embedding_methods,
                                           input_delim=setup.input_delim_other[j],
                                           output_delim=setup.output_delim_other[j],
                                           tune_params=setup.tune_params_other[j], maximize=setup.maximize,
                                           write_weights=setup.write_weights_other[j],
                                           write_dir=setup.write_dir_other[j], verbose=setup.verbose)

                # Log the list of results and generate plots if necessary
                for res in results:
                    scoresheet.log_results(res)
                    if setup.curves is not None:
                        res.plot(filename=os.path.join(nw_outpath, '{}_rep_{}'.format(res.method, repeat)),
                                 curve=setup.curves)
            except ValueError:
                logging.exception('Exception occurred while evaluating method `{}` on `{}` network.'
                                  .format(setup.names_other[j], setup.names[i]))
            except IOError:
                logging.exception('Exception occurred while evaluating method `{}` on `{}` network.'
                                  .format(setup.names_other[j], setup.names[i]))

    if setup.methods_opne is not None:
        # Evaluate methods from OpenNE
        # ----------------------------
        for j in range(len(setup.methods_opne)):
            try:
                # Evaluate the method
                if setup.directed:
                    command = setup.methods_opne[j] + \
                              " --graph-format edgelist --directed --input {} --output {} --representation-size {}"
                else:
                    command = setup.methods_opne[j] + \
                              " --graph-format edgelist --input {} --output {} --representation-size {}"
                results = nee.evaluate_cmd(method_name=setup.names_opne[j], method_type='ne', command=command,
                                           input_delim=' ', edge_embedding_methods=setup.edge_embedding_methods,
                                           output_delim=' ', tune_params=setup.tune_params_opne[j], maximize=setup.maximize,
                                           write_weights=False, write_dir=True, verbose=setup.verbose)

                # Log the list of results and generate plots if necessary
                for res in results:
                    scoresheet.log_results(res)
                    if setup.curves is not None:
                        res.plot(filename=os.path.join(nw_outpath, '{}_rep_{}'.format(res.method, repeat)),
                                 curve=setup.curves)
            except ValueError:
                logging.exception('Exception occurred while evaluating method `{}` on `{}` network.'
                                  .format(setup.names_other[j], setup.names[i]))
            except IOError:
                logging.exception('Exception occurred while evaluating method `{}` on `{}` network.'
                                  .format(setup.names_other[j], setup.names[i]))


if __name__ == "__main__":
    main()
