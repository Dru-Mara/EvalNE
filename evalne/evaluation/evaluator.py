#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# TODO: Use true labels and the preds to give statistics of where the method fails.

from __future__ import division

import itertools
import os
import re
import subprocess
import time
import logging

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from evalne.evaluation import edge_embeddings
from evalne.evaluation import score
from evalne.evaluation import split
from evalne.methods import similarity as sim
from evalne.utils import split_train_test as stt
from evalne.methods import katz


class LPEvaluator(object):
    """
    Class designed to simplify the evaluation of embedding methods for link prediction tasks.

    Parameters
    ----------
    traintest_split : EvalSplit()
        An object containing the train graph (a subgraph the full network spanning all the nodes) and a set of train
        true and false edges. Test edges are optional. If not provided only train results will be generated.
    trainvalid_split : EvalSplit()
        An object containing the validation graph (a subgraph the train network spanning all the nodes) and a set of
        train and valid true and false edges. If not provided a split with the same paremeters as the train one but
        with train_frac=0.9 will be computed.
    dim : int
        Embedding dimensionality.
    lp_model : Sklearn binary classifier
        The binary classifier to use for edge prediction.
    """

    def __init__(self, traintest_split, trainvalid_split=None, dim=128,
                 lp_model=LogisticRegressionCV(Cs=10, cv=5, penalty='l2', scoring='roc_auc', solver='lbfgs',
                                               max_iter=100)):
        # General evaluation parameters
        self.traintest_split = traintest_split
        self.trainvalid_split = trainvalid_split
        self.dim = dim
        self.edge_embed_method = None
        self.lp_model = lp_model

    def _init_trainvalid(self):
        if self.trainvalid_split is None or len(self.trainvalid_split.test_edges) == 0:
            logging.error('No test edges in trainvalid_split. Recomputing correct split...')
        self.trainvalid_split = split.EvalSplit()
        self.trainvalid_split.compute_splits(self.traintest_split.TG, nw_name=self.traintest_split.nw_name,
                                             train_frac=0.9, split_alg=self.traintest_split.split_alg,
                                             owa=self.traintest_split.owa, fe_ratio=self.traintest_split.fe_ratio,
                                             split_id=self.traintest_split.split_id, verbose=False)

    def evaluate_baseline(self, method, neighbourhood='in'):
        """
        Evaluates the baseline method requested. Evaluation output is returned as a Results object.
        Execution time is contain in the results object. If the train/test split object used to initialize the
        evaluator does not contain test edges, the results object will only contain train results.

        Parameters
        ----------
        method : basestring
            The names of any link prediction baseline from evalne.methods.similarity to evaluate.
        neighbourhood : basestring, optional
            A string indicating the 'in' or 'out' neighbourhood to be used for directed graphs.
            Default is 'in'.

        Returns
        -------
        results : Results
            Returns the evaluation results as a Results object.
        """
        # Measure execution time
        start = time.time()
        test_pred = None

        if 'katz' in method:
            m = method.split()
            if len(m) > 1:
                try:
                    exact = katz.Katz(self.traintest_split.TG, float(m[1]))
                except TypeError:
                    raise TypeError('Call to katz method incorrect, try: `katz 0.01`')
            else:
                exact = katz.Katz(self.traintest_split.TG)
            train_pred = exact.predict(self.traintest_split.train_edges)
            if len(self.traintest_split.test_edges) != 0:
                test_pred = exact.predict(self.traintest_split.test_edges)

        else:
            try:
                func = getattr(sim, str(method))
            except AttributeError:
                raise AttributeError('Method `{}` is not one of the available baselines!'.format(method))
            train_pred = func(self.traintest_split.TG, self.traintest_split.train_edges, neighbourhood)
            if len(self.traintest_split.test_edges) != 0:
                test_pred = func(self.traintest_split.TG, self.traintest_split.test_edges, neighbourhood)

        # Make predictions column vectors
        train_pred = np.array(train_pred)
        if test_pred is not None:
            test_pred = np.array(test_pred)

        # End of exec time measurement
        end = time.time() - start

        # Set some parameters for the results object
        params = {'neighbourhood': neighbourhood, 'eval_time': end}
        self.edge_embed_method = None

        # Compute the scores
        if nx.is_directed(self.traintest_split.TG):
            results = self.compute_results(data_split=self.traintest_split, method_name=method + '-' + neighbourhood,
                                           train_pred=train_pred, test_pred=test_pred, params=params)
        else:
            results = self.compute_results(data_split=self.traintest_split, method_name=method,
                                           train_pred=train_pred, test_pred=test_pred, params=params)

        return results

    def evaluate_cmd(self, method_name, method_type, command, edge_embedding_methods, input_delim, output_delim,
                     tune_params=None, maximize='auroc', write_weights=False, write_dir=False, verbose=True):
        r"""
        Evaluates an embedding method and tunes its parameters from the method's command line call string. This
        function can evaluate node embedding, edge embedding or end to end embedding methods.

        Parameters
        ----------
        method_name : basestring
            A string indicating the name of the method to be evaluated.
        method_type : basestring
            A string indicating the type of embedding method (i.e. ne, ee, e2e)
        command : basestring
            A string containing the call to the method as it would be written in the command line.
            For 'ne' methods placeholders (i.e. {}) need to be provided for the parameters: input network file,
            output file and embedding dimensionality, precisely IN THIS ORDER.
            For 'ee' methods with parameters: input network file, input train edgelist, input test edgelist, output
            train embeddings, output test embeddings and embedding dimensionality, 6 placeholders (i.e. {}) need to
            be provided, precisely IN THIS ORDER.
            For methods with parameters: input network file, input edgelist, output embeddings, and embedding
            dimensionality, 4 placeholders (i.e. {}) need to be provided, precisely IN THIS ORDER.
            For 'e2e' methods with parameters: input network file, input train edgelist, input test edgelist, output
            train predictions, output test predictions and embedding dimensionality, 6 placeholders (i.e. {}) need
            to be provided, precisely IN THIS ORDER.
            For methods with parameters: input network file, input edgelist, output predictions, and embedding
            dimensionality, 4 placeholders (i.e. {}) need to be provided, precisely IN THIS ORDER.
        edge_embedding_methods : array-like
            A list of methods used to compute edge embeddings from the node embeddings output by the NE models.
            The accepted values are the function names in evalne.evaluation.edge_embeddings.
            When evaluating 'ee' or 'e2e' methods, this parameter is ignored.
        input_delim : basestring
            The delimiter expected by the method as input (edgelist).
        output_delim : basestring
            The delimiter provided by the method in the output
        tune_params : basestring
            A string containing all the parameters to be tuned and their values.
        maximize : basestring
            The score to maximize while performing parameter tuning.
        write_weights : bool, optional
            If True the train graph passed to the embedding methods will be stored as weighted edgelist
            (e.g. triplets src, dst, weight) otherwise as normal edgelist. If the graph edges have no weight attribute
            and this parameter is set to True, a weight of 1 will be assigned to each edge. Default is False.
        write_dir : bool, optional
            This option is only relevant for undirected graphs. If False, the train graph will be stored with a single
            direction of the edges. If True, both directions of edges will be stored. Default is False.
        verbose : bool
            A parameter to control the amount of screen output.

        Returns
        -------
        results : list of Results
            Returns a list of Results objects, one per edge embedding method.

        """
        # Measure execution time
        start = time.time()

        # CHeck if a validation set needs to be initialized
        if self.trainvalid_split is None or len(self.trainvalid_split.test_edges) == 0:
            self._init_trainvalid()

        # Check the method type and raise an error if necessary
        if method_type not in ['ne', 'ee', 'e2e']:
            raise ValueError('Method type `{}` of method `{}` is unknown! Valid options are: `ne`, `ee`, `e2e`'
                             .format(method_type, method_name))

        # If the method evaluated does not require edge embeddings set this parameter to ['none']
        if method_type != 'ne':
            edge_embedding_methods = ['none']
            self.edge_embed_method = None

        # Check if tuning parameters is needed
        if tune_params is not None:
            print('Tuning parameters for {} ...'.format(method_name))

            # Variable to store the best results and parameters for each ee_method
            best_results = list()
            best_params = list()
            for i in range(len(edge_embedding_methods)):
                best_results.append(None)
                best_params.append(None)

            # Prepare the parameters
            sep = re.compile(r"--\w+")
            if sep.match(tune_params.strip()) is not None:
                params = tune_params.split('--')
                dash = ' --'
            else:
                params = tune_params.split('-')
                dash = ' -'
            params.pop(0)     # the first element is always nothing
            param_names = list()
            for i in range(len(params)):
                # Split the parameter name from the parameter values to be tested
                aux = (params[i].strip()).split()
                param_names.append(aux.pop(0))
                params[i] = aux

            # If there is only one parameter we treat it separately
            if len(param_names) == 1:
                for i in params[0]:
                    # Format the parameter combination
                    param_str = dash + param_names[0] + ' ' + i

                    # Create a command string with the new parameter
                    ext_command = command + param_str

                    try:
                        # Call the corresponding evaluation method
                        if method_type == 'ee' or method_type == 'e2e':
                            results = self._evaluate_ee_e2e_cmd(self.trainvalid_split, method_name, method_type,
                                                                ext_command, input_delim, output_delim, write_weights,
                                                                write_dir, verbose)
                        else:
                            results = self._evaluate_ne_cmd(self.trainvalid_split, method_name, ext_command,
                                                            edge_embedding_methods, input_delim, output_delim,
                                                            write_weights, write_dir, verbose)
                        results = list(results)

                        # Log the best results
                        for j in range(len(results)):
                            if best_results[j] is None:
                                best_results[j] = results[j]
                                best_params[j] = param_str
                            else:
                                func1 = getattr(results[j].test_scores, str(maximize))
                                func2 = getattr(best_results[j].test_scores, str(maximize))
                                if func1() > func2():
                                    best_results[j] = results[j]
                                    best_params[j] = param_str

                    except ValueError:
                        logging.exception('Exception occurred while evaluating param `{}` for method `{}` on `{}`.'
                                          .format(param_str, method_name, self.trainvalid_split.nw_name))

                    except IOError:
                        logging.exception('Exception occurred while evaluating param `{}` for method `{}` on `{}`.'
                                          .format(param_str, method_name, self.trainvalid_split.nw_name))
            else:
                # All parameter combinations
                combinations = list(itertools.product(*params))
                for comb in combinations:
                    # Format the parameter combination
                    param_str = ''
                    for i in range(len(comb)):
                        param_str += dash + param_names[i] + ' ' + comb[i]

                    # Update the command string with the parameter combination
                    ext_command = command + param_str

                    try:
                        # Call the corresponding evaluation method
                        if method_type == 'ee' or method_type == 'e2e':
                            results = self._evaluate_ee_e2e_cmd(self.trainvalid_split, method_name, method_type,
                                                                ext_command, input_delim, output_delim, write_weights,
                                                                write_dir, verbose)
                        else:
                            results = self._evaluate_ne_cmd(self.trainvalid_split, method_name, ext_command,
                                                            edge_embedding_methods, input_delim, output_delim,
                                                            write_weights, write_dir, verbose)
                        results = list(results)

                        # Log the best results
                        for i in range(len(results)):
                            if best_results[i] is None:
                                best_results[i] = results[i]
                                best_params[i] = param_str
                            else:
                                func1 = getattr(results[i].test_scores, str(maximize))
                                func2 = getattr(best_results[i].test_scores, str(maximize))
                                if func1() > func2():
                                    best_results[i] = results[i]
                                    best_params[i] = param_str

                    except ValueError:
                        logging.exception('Exception occurred while evaluating params `{}` for method `{}` on `{}`.'
                                          .format(param_str, method_name, self.trainvalid_split.nw_name))

                    except IOError:
                        logging.exception('Exception occurred while evaluating params `{}` for method `{}` on `{}`.'
                                          .format(param_str, method_name, self.trainvalid_split.nw_name))

            # We found best params for each ee method, log that info and corresponding score
            ee_scores = list()
            for i in range(len(edge_embedding_methods)):
                if best_results[i] is not None:
                    func = getattr(best_results[i].test_scores, str(maximize))
                    bestscore = func()
                else:
                    bestscore = 0.0
                ee_scores.append(bestscore)
                logging.info('Validation score for method `{}_{}` is: {}, corresponding best params were: `{}`'
                             .format(method_name, edge_embedding_methods[i], bestscore, best_params[i]))

            # We now select the ee that performs best in terms of maximize score
            best_ee_idx = np.argmax(ee_scores)
            if ee_scores[best_ee_idx] == 0.0:
                raise ValueError('All parameter combinations for method `{}` have failed! No results available.'
                                 .format(method_name))
            ext_command = command + best_params[best_ee_idx]

            # Call the corresponding evaluation method on the whole train data for the selected ee method
            if method_type == 'ee' or method_type == 'e2e':
                results = self._evaluate_ee_e2e_cmd(self.traintest_split, method_name, method_type, ext_command,
                                                    input_delim, output_delim, write_weights, write_dir, verbose)
            else:
                results = self._evaluate_ne_cmd(self.traintest_split, method_name, ext_command,
                                                [edge_embedding_methods[best_ee_idx]], input_delim, output_delim,
                                                write_weights, write_dir, verbose)

            # # We found best params for each ee method, now train model on whole train data to get actual results
            # results = list()
            # # For most ee method the best params will be the same, so we compute ne for distinct best params only
            # d = defaultdict(list)
            # for k, v in zip(best_params, edge_embedding_methods):
            #     d[k].append(v)
            # # If for any ee best params is none then all parameter combos failed for that ee method and we raise error
            # if None in d.keys():
            #     raise ValueError('All parameter combinations for method `{}` have failed! No results available.'
            #                      .format(method_name))
            # else:
            #     for params, ee_methods in d.items():
            #         ext_command = command + params
            #         logging.info('Best params for method `{}` using ee `{}` are `{}`'
            #                      .format(method_name, ee_methods, params))
            #         # Call the corresponding evaluation method
            #         if method_type == 'ee' or method_type == 'e2e':
            #             results.extend(self._evaluate_ee_e2e_cmd(self.traintest_split, method_name, method_type,
            #                                                     ext_command, input_delim, output_delim, write_weights,
            #                                                      write_dir, verbose))
            #         else:
            #           results.extend(self._evaluate_ne_cmd(self.traintest_split, method_name, ext_command, ee_methods,
            #                                                  input_delim, output_delim, write_weights, write_dir,
            #                                                  verbose))

        else:
            # No parameter tuning is needed
            # Call the corresponding evaluation method
            if method_type == 'ee' or method_type == 'e2e':
                results = self._evaluate_ee_e2e_cmd(self.traintest_split, method_name, method_type, command,
                                                    input_delim, output_delim, write_weights, write_dir, verbose)
            else:
                # We still have to tune the edge embedding method
                if len(edge_embedding_methods) > 1:
                    # For NE methods first compute the results on validation data
                    valid_results = self._evaluate_ne_cmd(self.trainvalid_split, method_name, command,
                                                          edge_embedding_methods, input_delim, output_delim,
                                                          write_weights, write_dir, verbose=False)

                    # Extract and log the validation scores
                    ee_scores = list()
                    for i in range(len(valid_results)):
                        func = getattr(valid_results[i].test_scores, str(maximize))
                        bestscore = func()
                        ee_scores.append(bestscore)
                        logging.info('Validation score for method `{}_{}` is: {}, no other tuned params.'
                                     .format(method_name, edge_embedding_methods[i], bestscore))

                    # We now select the ee that performs best in terms of maximize score
                    best_ee_idx = np.argmax(ee_scores)
                else:
                    # If we only have one ee method then that the one we compute results for, no need for validation
                    best_ee_idx = 0

                # Compute the results on the full train split
                results = self._evaluate_ne_cmd(self.traintest_split, method_name, command,
                                                [edge_embedding_methods[best_ee_idx]], input_delim, output_delim,
                                                write_weights, write_dir, verbose)

        # End of exec time measurement
        end = time.time() - start
        for res in results:
            res.params.update({'eval_time': end})

        # Return the evaluation results
        return results

    def _evaluate_ne_cmd(self, data_split, method_name, command, edge_embedding_methods, input_delim, output_delim,
                         write_weights, write_dir, verbose):
        """
        The actual implementation of the node embedding evaluation. Stores the train graph as an edgelist to a
        temporal file and provides it as input to the method evaluated. Performs the command line call and reads
        the output. Node embeddings are transformed to edge embeddings and predictions are run.

        Returns
        -------
        results : list
            A list of results, one for each edge embedding method set.
        """
        # Create temporal files with in/out data for method
        tmpedg = './edgelist.tmp'
        tmpemb = './emb.tmp'

        # Write the train data to a file
        data_split.save_tr_graph(tmpedg, delimiter=input_delim, write_stats=False,
                                 write_weights=write_weights, write_dir=write_dir)

        # Add the input, output and embedding dimensionality to the command
        command = command.format(tmpedg, tmpemb, self.dim)

        print('Running command...')
        print(command)

        try:
            # Call the method
            if not verbose:
                devnull = open(os.devnull, 'w')
                subprocess.call(command, shell=True, stdout=devnull, stderr=devnull)
            else:
                subprocess.call(command, shell=True)

            # Some methods append a .txt filetype to the outfile if its the case, read the txt
            if os.path.isfile('./emb.tmp.txt'):
                tmpemb = './emb.tmp.txt'

            # Autodetect header of output
            # Read num lines in output
            num_vectors = sum(1 for _ in open(tmpemb))
            emb_skiprows = num_vectors - len(data_split.TG.nodes)

            if emb_skiprows < 0:
                raise ValueError('Method {} does not provide a unique embedding for every graph node!'
                                 '\nExpected num. node embeddings: {} '
                                 '\nObtained num. node embeddings: {}'
                                 .format(method_name, len(data_split.TG.nodes), num_vectors))
            elif emb_skiprows > 0:
                logging.warning('Output of method {} contains {} more lines than expected. Will consider them part '
                                'of the header and ignore them... Expected num_lines {}, obtained lines {}.'
                                .format(method_name, emb_skiprows, len(data_split.TG.nodes), num_vectors))

            # Read the embeddings
            X = np.genfromtxt(tmpemb, delimiter=output_delim, dtype=float, skip_header=emb_skiprows, autostrip=True)

            if X.ndim == 1:
                raise ValueError('Error encountered while reading node embeddings for method {}. '
                                 'Please check the output delimiter for the method, this value is probably incorrect.'
                                 .format(method_name))

            if X.shape[1] == self.dim:
                # Assume embeddings given as matrix [X_0, X_1, ..., X_D] where rows correspond to sorted node id
                keys = map(str, sorted(data_split.TG.nodes))
                # keys = map(str, range(len(X)))
                X = dict(zip(keys, X))
            elif X.shape[1] == self.dim + 1:
                logging.warning('Output provided by method {} contains {} columns, {} expected!'
                                '\nAssuming first column to be the nodeID...'
                                .format(method_name, X.shape[1], self.dim))
                # Assume first col is node id and rest are embedding features [id, X_0, X_1, ..., X_D]
                keys = map(str, np.array(X[:, 0], dtype=int))
                X = dict(zip(keys, X[:, 1:]))
            else:
                raise ValueError('Incorrect node embedding dimensions for method {}!'
                                 '\nValues expected: {} or {} \nValue received: {}'
                                 .format(method_name, self.dim, self.dim + 1, X.shape[1]))

            # Evaluate the model
            results = list()
            for ee in edge_embedding_methods:
                results.append(self.evaluate_ne(data_split=data_split, X=X, method=method_name, edge_embed_method=ee))
            return results

        except IOError:
            raise IOError('Execution of method `{}` did not generate node embeddings file. \nPossible reasons: '
                          '1) method is not correctly installed or 2) wrong method call or parameters... '
                          '\nSetting verbose=True can provide more information.'.format(method_name))

        finally:
            # Delete the temporal files
            if os.path.isfile(tmpedg):
                os.remove(tmpedg)
            if os.path.isfile(tmpemb):
                os.remove(tmpemb)
            if os.path.isfile('./emb.tmp.txt'):
                os.remove('./emb.tmp.txt')

    def _evaluate_ee_e2e_cmd(self, data_split, method_name, method_type, command, input_delim, output_delim,
                             write_weights, write_dir, verbose):
        """
        The actual implementation of the edge embedding and end to end evaluation. Stores the train graph as an
        edgelist to a temporal file and provides it as input to the method evaluated together with the train and
        test edge sets. Performs the command line method call and reads the output edge embeddings/predictions.
        The method results are then computed according to the method type and returned.
        If no test edges are required, we still pass two dummy ones to the methods to prevent them from failing.

        Returns
        -------
        results : list
            A list with a single element, the result for the user-set edge embedding method.
            It returns a list for consistency with self._evaluate_ne_cmd()
        """
        # Create temporal files with in/out data for method
        tmpedg = './edgelist.tmp'
        tmp_tr_e = './tmp_tr_e.tmp'
        tmp_te_e = './tmp_te_e.tmp'
        tmp_tr_out = './tmp_tr_out.tmp'
        tmp_te_out = './tmp_te_out.tmp'

        # Check the amount of placeholders.
        # If 4 we assume: nw, tr_e, tr_out, dim
        # If 6 we assume: nw, tr_e, te_e, tr_out, te_out, dim
        placeholders = len(command.split('{}')) - 1
        if placeholders == 4:
            # Add input and output file paths and the embedding dimensionality to the command
            command = command.format(tmpedg, tmp_tr_e, tmp_tr_out, self.dim)
        elif placeholders == 6:
            # Add input and output file paths and the embedding dimensionality to the command
            command = command.format(tmpedg, tmp_tr_e, tmp_te_e, tmp_tr_out, tmp_te_out, self.dim)
        else:
            raise ValueError('Incorrect number of placeholders in {} command! Accepted values are 4 or 6.'
                             .format(method_name))

        # Write the train data to a file
        data_split.save_tr_graph(tmpedg, delimiter=input_delim, write_stats=False,
                                 write_weights=write_weights, write_dir=write_dir)

        # Write the train and test edgelists to files
        dummy_edges = 0
        if placeholders == 4:
            # Stack train and test edges if the method only takes one input file
            if len(data_split.test_edges) != 0:
                ebunch = np.vstack((data_split.train_edges, data_split.test_edges))
            else:
                ebunch = data_split.train_edges
            stt.store_edgelists(tmp_tr_e, tmp_te_e, ebunch, [])
        else:
            if len(data_split.test_edges) != 0:
                data_split.store_edgelists(tmp_tr_e, tmp_te_e)
            else:
                # If no test preds required we pass two dummy edges as test
                ebunch = data_split.train_edges
                stt.store_edgelists(tmp_tr_e, tmp_te_e, ebunch, [ebunch[0], ebunch[1]])
                dummy_edges = 2

        print('Running command...')
        print(command)

        try:
            # Call the method
            if not verbose:
                devnull = open(os.devnull, 'w')
                subprocess.call(command, shell=True, stdout=devnull, stderr=devnull)
            else:
                subprocess.call(command, shell=True)

            if placeholders == 4:
                # Autodetect and skip header if exists in prediction output.
                num_tr_out = sum(1 for _ in open(tmp_tr_out))
                skiprows = num_tr_out - (len(data_split.train_edges) + len(data_split.test_edges))

                if skiprows < 0:
                    raise ValueError('Method {} does not provide a unique embedding/prediction for every edge passed!'
                                     '\nExpected num. embeddings/predictions: {} '
                                     '\nObtained num. embeddings/predictions: {}'
                                     .format(method_name, len(data_split.train_edges) + len(data_split.test_edges),
                                             num_tr_out))
                elif skiprows > 0:
                    logging.warning('Found {} more lines in the output file than expected. Will consider these part '
                                    'of the header and ignore them... Expected num_lines {}, obtained num lines {}.'
                                    .format(skiprows, (len(data_split.train_edges) + len(data_split.test_edges)),
                                            num_tr_out))

                # Read the embeddings/predictions
                out = np.genfromtxt(tmp_tr_out, delimiter=output_delim, dtype=float, skip_header=skiprows,
                                    autostrip=True)

                # Check if the method is ee or e2e
                if method_type == 'ee':
                    if out.ndim == 2 and out.shape[1] == self.dim:
                        # Assume edge embeddings given as matrix [X_0, X_1, ..., X_D] in same order as edgelist
                        tr_out = out[0:len(data_split.train_edges), :]
                        te_out = out[len(data_split.train_edges):, :]
                    else:
                        raise ValueError('Incorrect edge embedding dimension for method {}!'
                                         '\nOutput expected: ({},{}) \nOutput received: {}'
                                         .format(method_name, len(data_split.train_edges) + len(data_split.test_edges),
                                                 self.dim, out.shape))
                else:
                    if out.ndim == 1:
                        # If output is a vector of predictions, assume is in the same order as the edgelist provided.
                        tr_out = out[0:len(data_split.train_edges)]
                        te_out = out[len(data_split.train_edges):]
                    else:
                        # If output is a matrix, assume last column has predictions in the same order as the edgelist.
                        logging.warning('Output provided by method {} is a matrix! '
                                        '\nPredictions assumed to be in the last column...'.format(method_name))
                        tr_out = out[0:len(data_split.train_edges), -1]
                        te_out = out[len(data_split.train_edges):, -1]

            else:
                # Autodetect and skip header if exists in output.
                num_tr_out = sum(1 for _ in open(tmp_tr_out))
                num_te_out = sum(1 for _ in open(tmp_te_out))
                tr_skiprows = num_tr_out - len(data_split.train_edges)
                te_skiprows = num_te_out - len(data_split.test_edges) - dummy_edges

                if tr_skiprows < 0 or te_skiprows < 0:
                    raise ValueError('Method {} does not provide a unique prediction/embedding for every edge passed!'
                                     '\nExpected num. train predictions/embeddings {}'
                                     '\nObtained num. train predictions/embeddings {}'
                                     '\nExpected num. test predictions/embeddings {}'
                                     '\nObtained num. test predictions/embeddings {}'
                                     .format(method_name, len(data_split.train_edges), num_tr_out,
                                             len(data_split.test_edges)+dummy_edges, num_te_out))
                elif tr_skiprows > 0:
                    logging.warning('Found {} more lines in the train file than expected. Will consider these part '
                                    'of the header and ignore them... Expected num_lines {}, obtained num lines {}.'
                                    .format(tr_skiprows, len(data_split.train_edges), num_tr_out))
                elif te_skiprows > 0:
                    logging.warning('Found {} more lines in the test file than expected. Will consider these part '
                                    'of the header and ignore them... Expected num_lines {}, obtained num lines {}.'
                                    .format(te_skiprows, len(data_split.test_edges)+dummy_edges, num_te_out))

                # Read the embeddings/predictions
                tr_out = np.genfromtxt(tmp_tr_out, delimiter=output_delim, dtype=float, skip_header=tr_skiprows,
                                       autostrip=True)
                te_out = np.genfromtxt(tmp_te_out, delimiter=output_delim, dtype=float, skip_header=te_skiprows,
                                       autostrip=True)

                # Check if the method is ee or e2e
                if method_type == 'ee':
                    # By default assume edge embeddings given as matrix [X_0, X_1, ..., X_D] in same order as edgelist
                    if tr_out.shape[0] != len(data_split.train_edges) or \
                            te_out.shape[0] != len(data_split.test_edges) + dummy_edges or \
                            tr_out.shape[1] != self.dim or te_out.shape[1] != self.dim:
                        raise ValueError('Incorrect edge embedding dimension for method {}!'
                                         '\nOutput expected train: ({},{}) \nOutput received train: {}'
                                         '\nOutput expected test: ({},{}) \nOutput received test: {}'
                                         .format(method_name, len(data_split.train_edges), self.dim, tr_out.shape,
                                                 len(data_split.test_edges)+dummy_edges, self.dim, te_out.shape))
                else:
                    # By default we assume the output is a vector of predictions in the same order as the edgelist.
                    # If output is a matrix, assume last column has predictions in the same order as the edgelist.
                    if tr_out.ndim == 2:
                        logging.warning('Output provided by method {} is a matrix!'
                                        '\nPredictions assumed to be in the last column...'.format(method_name))
                        tr_out = tr_out[:, -1]
                        if len(data_split.test_edges) > 1:
                            te_out = te_out[:, -1]

                    if len(data_split.test_edges) == 1:
                        if te_out.ndim == 1:
                            te_out = [float(te_out[:, -1])]
                        else:
                            te_out = [float(te_out)]

            # If no test edges were required make te_out none
            if len(data_split.test_edges) == 0:
                te_out = None

            # Check if the method is ee or e2e and call the corresponding function
            results = list()
            if method_type == 'ee':
                train_pred, test_pred = self.compute_pred(data_split=data_split, tr_edge_embeds=tr_out,
                                                          te_edge_embeds=te_out)
                results.append(self.compute_results(data_split=data_split, method_name=method_name,
                                                    train_pred=train_pred, test_pred=test_pred))
            else:
                results.append(self.compute_results(data_split=data_split, method_name=method_name,
                                                    train_pred=tr_out, test_pred=te_out))
            return results

        except IOError:
            raise IOError('Execution of method `{}` did not generate expected output file. \nPossible reasons: '
                          '1) method is not correctly installed or 2) wrong method call or parameters... '
                          '\nSetting verbose=True can provide more information.'.format(method_name))

        finally:
            # Delete the temporal files
            if os.path.isfile(tmpedg):
                os.remove(tmpedg)
            if os.path.isfile(tmp_tr_e):
                os.remove(tmp_tr_e)
            if os.path.isfile(tmp_te_e):
                os.remove(tmp_te_e)
            if os.path.isfile(tmp_tr_out):
                os.remove(tmp_tr_out)
            if os.path.isfile(tmp_te_out):
                os.remove(tmp_te_out)

    def evaluate_ne(self, data_split, X, method, edge_embed_method,
                    label_binarizer=LogisticRegression(solver='liblinear'), params=None):
        r"""
        Runs the complete pipeline, from node embeddings to edge embeddings and returns the prediction results.
        If data_split.test_edges is None, the train results will be returned as both train and test in Results.

        Parameters
        ----------
        data_split : EvalSplit
            An EvalSplit object that encapsulates the train/test or train/validation data.
        X : dict
            A dictionary where keys are nodes in the graph and values are the node embeddings.
            The keys are of type str and the values of type array.
        method : basestring
            A string indicating the name of the method to be evaluated.
        edge_embed_method : basestring
            A string indicating the method used to compute edge embeddings from node embeddings.
            The accepted values are any of the function names in evalne.evaluation.edge_embeddings.
        label_binarizer : string or Sklearn binary classifier, optional
            If the predictions returned by the model are not binary, this parameter indicates how these binary
            predictions should be computed in order to be able to provide metrics such as the confusion matrix.
            Any Sklear binary classifier can be used or the keyword 'median' which will used the prediction medians
            as binarization thresholds.
            Default is LogisticRegression(solver='liblinear')
        params : dict
            A dictionary of parameters : values to be added to the results class.

        Returns
        -------
        results : Results
            A results object
        """
        # Run the evaluation pipeline
        tr_edge_embeds, te_edge_embeds = self.compute_ee(data_split, X, edge_embed_method)
        train_pred, test_pred = self.compute_pred(data_split, tr_edge_embeds, te_edge_embeds)

        return self.compute_results(data_split=data_split, method_name=method, train_pred=train_pred,
                                    test_pred=test_pred, label_binarizer=label_binarizer, params=params)

    def compute_ee(self, data_split, X, edge_embed_method):
        r"""
        Computes edge embeddings using the given node embeddings dictionary and edge embedding method.
        If data_split.test_edges is None, te_edge_embeds will be None.

        Parameters
        ----------
        data_split : EvalSplit
            An EvalSplit object that encapsulates the train/test or train/validation data.
        X : dict
            A dictionary where keys are nodes in the graph and values are the node embeddings.
            The keys are of type str and the values of type array.
        edge_embed_method : basestring
            A string indicating the method used to compute edge embeddings from node embeddings.
            The accepted values are any of the function names in evalne.evaluation.edge_embeddings.

        Returns
        -------
        tr_edge_embeds : matrix
            A Numpy matrix containing the train edge embeddings.
        te_edge_embeds : matrix
            A Numpy matrix containing the test edge embeddings. Returns None if data_split.test_edges is None.
        """
        self.edge_embed_method = edge_embed_method

        try:
            func = getattr(edge_embeddings, str(edge_embed_method))
        except AttributeError:
            raise AttributeError('Edge embedding method `{}` is not a valid option.'.format(edge_embed_method))

        tr_edge_embeds = func(X, data_split.train_edges)
        if len(data_split.test_edges) != 0:
            te_edge_embeds = func(X, data_split.test_edges)
            return tr_edge_embeds, te_edge_embeds
        else:
            return tr_edge_embeds, None

    def compute_pred(self, data_split, tr_edge_embeds, te_edge_embeds=None):
        r"""
        Computes predictions from the given edge embeddings.
        Trains an LP model with the train edge embeddings and performs predictions for train and test edge embeddings.
        If te_edge_embeds is None test_pred will be None.

        Parameters
        ----------
        data_split : EvalSplit
            An EvalSplit object that encapsulates the train/test or train/validation data.
        tr_edge_embeds : matrix
            A Numpy matrix containing the train edge embeddings.
        te_edge_embeds : matrix, optional
            A Numpy matrix containing the test edge embeddings. Default is None.

        Returns
        -------
        train_pred : array
            The link predictions for the train data.
        test_pred : array
            The link predictions for the test data. Returns None if te_edge_embeds is None.
        """
        # Train the LP model
        self.lp_model.fit(tr_edge_embeds, data_split.train_labels)

        # Predict
        train_pred = self.lp_model.predict_proba(tr_edge_embeds)[:, 1]
        if te_edge_embeds is not None:
            test_pred = self.lp_model.predict_proba(te_edge_embeds)[:, 1]
            return train_pred, test_pred
        else:
            return train_pred, None

    def compute_results(self, data_split, method_name, train_pred, test_pred=None,
                        label_binarizer=LogisticRegression(solver='liblinear'), params=None):
        r"""
        Generates results from the given predictions and returns them. If test_pred is not provided, the Results
        object will only contain the train scores.

        Parameters
        ----------
        data_split : EvalSplit
            An EvalSplit object that encapsulates the train/test or train/validation data.
        method_name : basestring
            A string indicating the name of the method for which the results will be created.
        train_pred :
            The link predictions for the train data.
        test_pred : array_like, optional
            The link predictions for the test data. Default is None.
        label_binarizer : string or Sklearn binary classifier, optional
            If the predictions returned by the model are not binary, this parameter indicates how these binary
            predictions should be computed in order to be able to provide metrics such as the confusion matrix.
            Any Sklear binary classifier can be used or the keyword 'median' which will used the prediction medians
            as binarization thresholds.
            Default is LogisticRegression(solver='liblinear')
        params : dict, optional
            A dictionary of parameters : values to be added to the results class.
            Default is None.

        Returns
        -------
        results : Results
            Returns the evaluation results.
        """
        # Get global parameters
        if self.edge_embed_method is not None:
            parameters = {'dim': self.dim, 'edge_embed_method': self.edge_embed_method}
        else:
            parameters = {'dim': self.dim}

        # Get data related parameters
        parameters.update(self.traintest_split.get_parameters())

        # Obtain the evaluation parameters
        if params is not None:
            parameters.update(params)

        if test_pred is None:
            results = score.Results(method=method_name, params=parameters,
                                    train_pred=train_pred, train_labels=data_split.train_labels,
                                    test_pred=None, test_labels=None,
                                    label_binarizer=label_binarizer)
        else:
            results = score.Results(method=method_name, params=parameters,
                                    train_pred=train_pred, train_labels=data_split.train_labels,
                                    test_pred=train_pred, test_labels=data_split.train_labels,
                                    label_binarizer=label_binarizer)
        return results


class NREvaluator(LPEvaluator):
    """
    Class designed to simplify the evaluation of embedding methods for network reconstruction tasks.
    The train graphs is assumed to be the complete graph. Parameter tuning is performed on a validation graph which
    is also the complete graph.

    Parameters
    ----------
    traintest_split : EvalSplit()
        An object containing the train graph (in this case the full network) and a set of train true and false edges.
        These edges can be all edges in the graph or a subset.
    dim : int
        Embedding dimensionality
    lp_model : Sklearn binary classifier.
        The binary classifier to use for edge prediction.
    """

    def __init__(self, traintest_split, dim=128,
                 lp_model=LogisticRegressionCV(Cs=10, cv=5, penalty='l2', scoring='roc_auc', solver='lbfgs',
                                               max_iter=100)):
        # General evaluation parameters
        super(NREvaluator, self).__init__(traintest_split, dim=dim, lp_model=lp_model)

    def _check_split(self):
        if self.traintest_split.test_edges is not None:
            raise ValueError('For network reconstruction test edges need to be set to None!')

    def evaluate_cmd(self, method_name, method_type, command, edge_embedding_methods, input_delim, output_delim,
                     tune_params=None, maximize='auroc', write_weights=False, write_dir=False, verbose=True):
        r"""
        Evaluates an embedding method and tunes its parameters from the method's command line call string. This
        function can evaluate node embedding, edge embedding or end to end embedding methods.
        If model parameter tuning is required, models are tuned directly on the train data. The returned Results object
        will only contain train scores.

        Parameters
        ----------
        method_name : basestring
            A string indicating the name of the method to be evaluated.
        method_type : basestring
            A string indicating the type of embedding method (i.e. ne, ee, e2e)
        command : basestring
            A string containing the call to the method as it would be written in the command line.
            For 'ne' methods placeholders (i.e. {}) need to be provided for the parameters: input network file,
            output file and embedding dimensionality, precisely IN THIS ORDER.
            For 'ee' methods with parameters: input network file, input train edgelist, input test edgelist, output
            train embeddings, output test embeddings and embedding dimensionality, 6 placeholders (i.e. {}) need to
            be provided, precisely IN THIS ORDER.
            For methods with parameters: input network file, input edgelist, output embeddings, and embedding
            dimensionality, 4 placeholders (i.e. {}) need to be provided, precisely IN THIS ORDER.
            For 'e2e' methods with parameters: input network file, input train edgelist, input test edgelist, output
            train predictions, output test predictions and embedding dimensionality, 6 placeholders (i.e. {}) need
            to be provided, precisely IN THIS ORDER.
            For methods with parameters: input network file, input edgelist, output predictions, and embedding
            dimensionality, 4 placeholders (i.e. {}) need to be provided, precisely IN THIS ORDER.
        edge_embedding_methods : array-like
            A list of methods used to compute edge embeddings from the node embeddings output by the NE models.
            The accepted values are the function names in evalne.evaluation.edge_embeddings.
            When evaluating 'ee' or 'e2e' methods, this parameter is ignored.
        input_delim : basestring
            The delimiter expected by the method as input (edgelist).
        output_delim : basestring
            The delimiter provided by the method in the output
        tune_params : basestring
            A string containing all the parameters to be tuned and their values.
        maximize : basestring
            The score to maximize while performing parameter tuning.
        write_weights : bool, optional
            If True the train graph passed to the embedding methods will be stored as weighted edgelist
            (e.g. triplets src, dst, weight) otherwise as normal edgelist. If the graph edges have no weight attribute
            and this parameter is set to True, a weight of 1 will be assigned to each edge. Default is False.
        write_dir : bool, optional
            This option is only relevant for undirected graphs. If False, the train graph will be stored with a single
            direction of the edges. If True, both directions of edges will be stored. Default is False.
        verbose : bool
            A parameter to control the amount of screen output.

        Returns
        -------
        results : list of Results
            Returns a list of Results objects, one per edge embedding method.

        """
        # Measure execution time
        start = time.time()

        # Check the method type and raise an error if necessary
        if method_type not in ['ne', 'ee', 'e2e']:
            raise ValueError('Method type `{}` of method `{}` is unknown! Valid options are: `ne`, `ee`, `e2e`'
                             .format(method_type, method_name))

        # If the method evaluated does not require edge embeddings set this parameter to ['none']
        if method_type != 'ne':
            edge_embedding_methods = ['none']
            self.edge_embed_method = None

        # Check if tuning parameters is needed
        if tune_params is not None:
            print('Tuning parameters for {} ...'.format(method_name))

            # Variable to store the best results and parameters for each ee_method
            best_results = list()
            best_params = list()
            for i in range(len(edge_embedding_methods)):
                best_results.append(None)
                best_params.append(None)

            # Prepare the parameters
            sep = re.compile(r"--\w+")
            if sep.match(tune_params.strip()) is not None:
                params = tune_params.split('--')
                dash = ' --'
            else:
                params = tune_params.split('-')
                dash = ' -'
            params.pop(0)     # the first element is always nothing
            param_names = list()
            for i in range(len(params)):
                # Split the parameter name from the parameter values to be tested
                aux = (params[i].strip()).split()
                param_names.append(aux.pop(0))
                params[i] = aux

            # If there is only one parameter we treat it separately
            if len(param_names) == 1:
                for i in params[0]:
                    # Format the parameter combination
                    param_str = dash + param_names[0] + ' ' + i

                    # Create a command string with the new parameter
                    ext_command = command + param_str

                    try:
                        # Call the corresponding evaluation method
                        if method_type == 'ee' or method_type == 'e2e':
                            results = self._evaluate_ee_e2e_cmd(self.traintest_split, method_name, method_type,
                                                                ext_command, input_delim, output_delim, write_weights,
                                                                write_dir, verbose)
                        else:
                            results = self._evaluate_ne_cmd(self.traintest_split, method_name, ext_command,
                                                            edge_embedding_methods, input_delim, output_delim,
                                                            write_weights, write_dir, verbose)
                        results = list(results)

                        # Log the best results
                        for j in range(len(results)):
                            if best_results[j] is None:
                                best_results[j] = results[j]
                                best_params[j] = param_str
                            else:
                                # For NR we look at the train score only!
                                func1 = getattr(results[j].train_scores, str(maximize))
                                func2 = getattr(best_results[j].train_scores, str(maximize))
                                if func1() > func2():
                                    best_results[j] = results[j]
                                    best_params[j] = param_str

                    except ValueError:
                        logging.exception('Exception occurred while evaluating param `{}` for method `{}` on `{}`.'
                                          .format(param_str, method_name, self.traintest_split.nw_name))

                    except IOError:
                        logging.exception('Exception occurred while evaluating param `{}` for method `{}` on `{}`.'
                                          .format(param_str, method_name, self.traintest_split.nw_name))
            else:
                # All parameter combinations
                combinations = list(itertools.product(*params))
                for comb in combinations:
                    # Format the parameter combination
                    param_str = ''
                    for i in range(len(comb)):
                        param_str += dash + param_names[i] + ' ' + comb[i]

                    # Update the command string with the parameter combination
                    ext_command = command + param_str

                    try:
                        # Call the corresponding evaluation method
                        if method_type == 'ee' or method_type == 'e2e':
                            results = self._evaluate_ee_e2e_cmd(self.traintest_split, method_name, method_type,
                                                                ext_command, input_delim, output_delim, write_weights,
                                                                write_dir, verbose)
                        else:
                            results = self._evaluate_ne_cmd(self.traintest_split, method_name, ext_command,
                                                            edge_embedding_methods, input_delim, output_delim,
                                                            write_weights, write_dir, verbose)
                        results = list(results)

                        # Log the best results
                        for i in range(len(results)):
                            if best_results[i] is None:
                                best_results[i] = results[i]
                                best_params[i] = param_str
                            else:
                                # For NR we look at the train score only!
                                func1 = getattr(results[i].train_scores, str(maximize))
                                func2 = getattr(best_results[i].train_scores, str(maximize))
                                if func1() > func2():
                                    best_results[i] = results[i]
                                    best_params[i] = param_str

                    except ValueError:
                        logging.exception('Exception occurred while evaluating params `{}` for method `{}` on `{}`.'
                                          .format(param_str, method_name, self.traintest_split.nw_name))

                    except IOError:
                        logging.exception('Exception occurred while evaluating params `{}` for method `{}` on `{}`.'
                                          .format(param_str, method_name, self.traintest_split.nw_name))

            # We found best params for each ee method, log that info and corresponding score
            ee_scores = list()
            for i in range(len(edge_embedding_methods)):
                if best_results[i] is not None:
                    func = getattr(best_results[i].train_scores, str(maximize))
                    bestscore = func()
                else:
                    bestscore = 0.0
                ee_scores.append(bestscore)
                logging.info('Validation score for method `{}_{}` is: {}, corresponding best params were: `{}`'
                             .format(method_name, edge_embedding_methods[i], bestscore, best_params[i]))

            # We now select the ee that performs best in terms of maximize score
            best_ee_idx = np.argmax(ee_scores)
            if ee_scores[best_ee_idx] == 0.0:
                raise ValueError('All parameter combinations for method `{}` have failed! No results available.'
                                 .format(method_name))

            # Since we validated best params on train, we found our scores.
            results = [best_results[best_ee_idx]]

        else:
            # No parameter tuning is needed
            # Call the corresponding evaluation method
            if method_type == 'ee' or method_type == 'e2e':
                results = self._evaluate_ee_e2e_cmd(self.traintest_split, method_name, method_type, command,
                                                    input_delim, output_delim, write_weights, write_dir, verbose)
            else:
                # For NE methods we still have to tune the edge embedding method
                results = self._evaluate_ne_cmd(self.traintest_split, method_name, command,
                                                edge_embedding_methods, input_delim, output_delim,
                                                write_weights, write_dir, verbose)

                # Extract and log the validation scores
                ee_scores = list()
                for i in range(len(results)):
                    # For NR we only look at the train scores
                    func = getattr(results[i].train_scores, str(maximize))
                    bestscore = func()
                    ee_scores.append(bestscore)
                    logging.info('Validation score for method `{}_{}` is: {}, no other tuned params.'
                                 .format(method_name, edge_embedding_methods[i], bestscore))

                # We now select the ee that performs best in terms of maximize score
                best_ee_idx = np.argmax(ee_scores)
                results = [results[best_ee_idx]]

        # End of exec time measurement
        end = time.time() - start
        for res in results:
            res.params.update({'eval_time': end})

        # Return the evaluation results
        return results


class NCEvaluator(object):
    """
    Class designed to simplify the evaluation of embedding methods for node classification tasks.
    The train graphs is assumed to be the complete graph. Parameter tuning is performed on a validation graph.

    Parameters
    ----------
    traintest_split : EvalSplit()
        An object containing the train graph (in this case the full network) and a set of train true edges.
        The sets of edges from the object will be ignored.
    trainvalid_split : EvalSplit()
        An object containing the validation graph and a set of train and valid true and false edges. If this object is
        not privided, a validation split with the same parameters as the train/test one will be generated but with a
        90/10 split.
    dim : int
        Embedding dimensionality.
    lp_model : Sklearn binary classifier.
        The binary classifier to use for edge prediction.
    """

    def __init__(self, traintest_split, trainvalid_split=None, dim=128,
                 lp_model=LogisticRegressionCV(Cs=10, cv=5, penalty='l2', scoring='roc_auc', solver='lbfgs',
                                               max_iter=100)):
        # General evaluation parameters
        self.traintest_split = traintest_split
        self.trainvalid_split = self._init_trainvalid(trainvalid_split)
        self._check_split()
        self.dim = dim
        self.edge_embed_method = None
        self.lp_model = lp_model

    def _check_split(self):
        if self.traintest_split.test_edges is not None:
            raise ValueError('For node classification test edges need to be set to None!')

    def _init_trainvalid(self, tv):
        if tv is None or tv.test_edges is None:
            if tv.test_edges is None:
                logging.error('Not all edge sets initialized in trainvalid_split. Recomputing correct split...')
            tv = split.EvalSplit()
            tv.compute_splits(self.traintest_split.TG, nw_name=self.traintest_split.nw_name, train_frac=0.9,
                              split_alg=self.traintest_split.split_alg, owa=self.traintest_split.owa,
                              fe_ratio=self.traintest_split.fe_ratio, split_id=self.traintest_split.split_id,
                              verbose=False)
        return tv

    def evaluate_cmd(self, method_name, method_type, command, edge_embedding_methods, input_delim, output_delim,
                     tune_params=None, maximize='auroc', write_weights=False, write_dir=False, verbose=True):
        r"""
        Evaluates an embedding method and tunes its parameters from the method's command line call string. This
        function can evaluate node embedding, edge embedding or end to end embedding methods.
        If model parameter tuning is required and train/valid split is provided this will be used. Otherwise
        a new 90/10 train/valid split will be computed.

        Parameters
        ----------
        method_name : basestring
            A string indicating the name of the method to be evaluated.
        method_type : basestring
            A string indicating the type of embedding method (i.e. ne, ee, e2e)
        command : basestring
            A string containing the call to the method as it would be written in the command line.
            For 'ne' methods placeholders (i.e. {}) need to be provided for the parameters: input network file,
            output file and embedding dimensionality, precisely IN THIS ORDER.
            For 'ee' methods with parameters: input network file, input train edgelist, input test edgelist, output
            train embeddings, output test embeddings and embedding dimensionality, 6 placeholders (i.e. {}) need to
            be provided, precisely IN THIS ORDER.
            For methods with parameters: input network file, input edgelist, output embeddings, and embedding
            dimensionality, 4 placeholders (i.e. {}) need to be provided, precisely IN THIS ORDER.
            For 'e2e' methods with parameters: input network file, input train edgelist, input test edgelist, output
            train predictions, output test predictions and embedding dimensionality, 6 placeholders (i.e. {}) need
            to be provided, precisely IN THIS ORDER.
            For methods with parameters: input network file, input edgelist, output predictions, and embedding
            dimensionality, 4 placeholders (i.e. {}) need to be provided, precisely IN THIS ORDER.
        edge_embedding_methods : array-like
            A list of methods used to compute edge embeddings from the node embeddings output by the NE models.
            The accepted values are the function names in evalne.evaluation.edge_embeddings.
            When evaluating 'ee' or 'e2e' methods, this parameter is ignored.
        input_delim : basestring
            The delimiter expected by the method as input (edgelist).
        output_delim : basestring
            The delimiter provided by the method in the output
        tune_params : basestring
            A string containing all the parameters to be tuned and their values.
        maximize : basestring
            The score to maximize while performing parameter tuning.
        write_weights : bool, optional
            If True the train graph passed to the embedding methods will be stored as weighted edgelist
            (e.g. triplets src, dst, weight) otherwise as normal edgelist. If the graph edges have no weight attribute
            and this parameter is set to True, a weight of 1 will be assigned to each edge. Default is False.
        write_dir : bool, optional
            This option is only relevant for undirected graphs. If False, the train graph will be stored with a single
            direction of the edges. If True, both directions of edges will be stored. Default is False.
        verbose : bool
            A parameter to control the amount of screen output.

        Returns
        -------
        results : list of Results
            Returns a list of Results objects, one per edge embedding method.

        """
        # Measure execution time
        start = time.time()

        # Check the method type and raise an error if necessary
        if method_type not in ['ne', 'ee', 'e2e']:
            raise ValueError('Method type `{}` of method `{}` is unknown! Valid options are: `ne`, `ee`, `e2e`'
                             .format(method_type, method_name))

        # If the method evaluated does not require edge embeddings set this parameter to ['none']
        if method_type != 'ne':
            edge_embedding_methods = ['none']
            self.edge_embed_method = None

        # Check if tuning parameters is needed
        if tune_params is not None:
            print('Tuning parameters for {} ...'.format(method_name))

            # Variable to store the best results and parameters for each ee_method
            best_results = list()
            best_params = list()
            for i in range(len(edge_embedding_methods)):
                best_results.append(None)
                best_params.append(None)

            # Prepare the parameters
            sep = re.compile(r"--\w+")
            if sep.match(tune_params.strip()) is not None:
                params = tune_params.split('--')
                dash = ' --'
            else:
                params = tune_params.split('-')
                dash = ' -'
            params.pop(0)     # the first element is always nothing
            param_names = list()
            for i in range(len(params)):
                # Split the parameter name from the parameter values to be tested
                aux = (params[i].strip()).split()
                param_names.append(aux.pop(0))
                params[i] = aux

            # If there is only one parameter we treat it separately
            if len(param_names) == 1:
                for i in params[0]:
                    # Format the parameter combination
                    param_str = dash + param_names[0] + ' ' + i

                    # Create a command string with the new parameter
                    ext_command = command + param_str

                    try:
                        # Call the corresponding evaluation method
                        if method_type == 'ee' or method_type == 'e2e':
                            results = self._evaluate_ee_e2e_cmd(self.trainvalid_split, method_name, method_type,
                                                                ext_command, input_delim, output_delim, write_weights,
                                                                write_dir, verbose)
                        else:
                            results = self._evaluate_ne_cmd(self.trainvalid_split, method_name, ext_command,
                                                            edge_embedding_methods, input_delim, output_delim,
                                                            write_weights, write_dir, verbose)
                        results = list(results)

                        # Log the best results
                        for j in range(len(results)):
                            if best_results[j] is None:
                                best_results[j] = results[j]
                                best_params[j] = param_str
                            else:
                                func1 = getattr(results[j].test_scores, str(maximize))
                                func2 = getattr(best_results[j].test_scores, str(maximize))
                                if func1() > func2():
                                    best_results[j] = results[j]
                                    best_params[j] = param_str

                    except ValueError:
                        logging.exception('Exception occurred while evaluating param `{}` for method `{}` on `{}`.'
                                          .format(param_str, method_name, self.trainvalid_split.nw_name))

                    except IOError:
                        logging.exception('Exception occurred while evaluating param `{}` for method `{}` on `{}`.'
                                          .format(param_str, method_name, self.trainvalid_split.nw_name))
            else:
                # All parameter combinations
                combinations = list(itertools.product(*params))
                for comb in combinations:
                    # Format the parameter combination
                    param_str = ''
                    for i in range(len(comb)):
                        param_str += dash + param_names[i] + ' ' + comb[i]

                    # Update the command string with the parameter combination
                    ext_command = command + param_str

                    try:
                        # Call the corresponding evaluation method
                        if method_type == 'ee' or method_type == 'e2e':
                            results = self._evaluate_ee_e2e_cmd(self.trainvalid_split, method_name, method_type,
                                                                ext_command, input_delim, output_delim, write_weights,
                                                                write_dir, verbose)
                        else:
                            results = self._evaluate_ne_cmd(self.trainvalid_split, method_name, ext_command,
                                                            edge_embedding_methods, input_delim, output_delim,
                                                            write_weights, write_dir, verbose)
                        results = list(results)

                        # Log the best results
                        for i in range(len(results)):
                            if best_results[i] is None:
                                best_results[i] = results[i]
                                best_params[i] = param_str
                            else:
                                func1 = getattr(results[i].test_scores, str(maximize))
                                func2 = getattr(best_results[i].test_scores, str(maximize))
                                if func1() > func2():
                                    best_results[i] = results[i]
                                    best_params[i] = param_str

                    except ValueError:
                        logging.exception('Exception occurred while evaluating params `{}` for method `{}` on `{}`.'
                                          .format(param_str, method_name, self.trainvalid_split.nw_name))

                    except IOError:
                        logging.exception('Exception occurred while evaluating params `{}` for method `{}` on `{}`.'
                                          .format(param_str, method_name, self.trainvalid_split.nw_name))

            # We found best params for each ee method, log that info and corresponding score
            ee_scores = list()
            for i in range(len(edge_embedding_methods)):
                if best_results[i] is not None:
                    func = getattr(best_results[i].test_scores, str(maximize))
                    bestscore = func()
                else:
                    bestscore = 0.0
                ee_scores.append(bestscore)
                logging.info('Validation score for method `{}_{}` is: {}, corresponding best params were: `{}`'
                             .format(method_name, edge_embedding_methods[i], bestscore, best_params[i]))

            # We now select the ee that performs best in terms of maximize score
            best_ee_idx = np.argmax(ee_scores)
            if ee_scores[best_ee_idx] == 0.0:
                raise ValueError('All parameter combinations for method `{}` have failed! No results available.'
                                 .format(method_name))
            ext_command = command + best_params[best_ee_idx]

            # Call the corresponding evaluation method on the whole train data for the selected ee method
            if method_type == 'ee' or method_type == 'e2e':
                results = self._evaluate_ee_e2e_cmd(self.traintest_split, method_name, method_type, ext_command,
                                                    input_delim, output_delim, write_weights, write_dir, verbose)
            else:
                results = self._evaluate_ne_cmd(self.traintest_split, method_name, ext_command,
                                                [edge_embedding_methods[best_ee_idx]], input_delim, output_delim,
                                                write_weights, write_dir, verbose)

            # # We found best params for each ee method, now train model on whole train data to get actual results
            # results = list()
            # # For most ee method the best params will be the same, so we compute ne for distinct best params only
            # d = defaultdict(list)
            # for k, v in zip(best_params, edge_embedding_methods):
            #     d[k].append(v)
            # # If for any ee best params is none then all parameter combos failed for that ee method and we raise error
            # if None in d.keys():
            #     raise ValueError('All parameter combinations for method `{}` have failed! No results available.'
            #                      .format(method_name))
            # else:
            #     for params, ee_methods in d.items():
            #         ext_command = command + params
            #         logging.info('Best params for method `{}` using ee `{}` are `{}`'
            #                      .format(method_name, ee_methods, params))
            #         # Call the corresponding evaluation method
            #         if method_type == 'ee' or method_type == 'e2e':
            #             results.extend(self._evaluate_ee_e2e_cmd(self.traintest_split, method_name, method_type,
            #                                                     ext_command, input_delim, output_delim, write_weights,
            #                                                      write_dir, verbose))
            #         else:
            #           results.extend(self._evaluate_ne_cmd(self.traintest_split, method_name, ext_command, ee_methods,
            #                                                  input_delim, output_delim, write_weights, write_dir,
            #                                                  verbose))

        else:
            # No parameter tuning is needed
            # Call the corresponding evaluation method
            if method_type == 'ee' or method_type == 'e2e':
                results = self._evaluate_ee_e2e_cmd(self.traintest_split, method_name, method_type, command,
                                                    input_delim, output_delim, write_weights, write_dir, verbose)
            else:
                # We still have to tune the edge embedding method
                if len(edge_embedding_methods) > 1:
                    # For NE methods first compute the results on validation data
                    valid_results = self._evaluate_ne_cmd(self.trainvalid_split, method_name, command,
                                                          edge_embedding_methods, input_delim, output_delim,
                                                          write_weights, write_dir, verbose=False)

                    # Extract and log the validation scores
                    ee_scores = list()
                    for i in range(len(valid_results)):
                        func = getattr(valid_results[i].test_scores, str(maximize))
                        bestscore = func()
                        ee_scores.append(bestscore)
                        logging.info('Validation score for method `{}_{}` is: {}, no other tuned params.'
                                     .format(method_name, edge_embedding_methods[i], bestscore))

                    # We now select the ee that performs best in terms of maximize score
                    best_ee_idx = np.argmax(ee_scores)
                else:
                    # If we only have one ee method then that the one we compute results for, no need for validation
                    best_ee_idx = 0

                # Compute the results on the full train split
                results = self._evaluate_ne_cmd(self.traintest_split, method_name, command,
                                                [edge_embedding_methods[best_ee_idx]], input_delim, output_delim,
                                                write_weights, write_dir, verbose)

        # End of exec time measurement
        end = time.time() - start
        for res in results:
            res.params.update({'eval_time': end})

        # Return the evaluation results
        return results

    def _evaluate_ne_cmd(self, data_split, method_name, command, edge_embedding_methods, input_delim, output_delim,
                         write_weights, write_dir, verbose):
        """
        The actual implementation of the node embedding evaluation. Stores the train graph as an edgelist to a
        temporal file and provides it as input to the method evaluated. Performs the command line call and reads
        the output. Node embeddings are transformed to edge embeddings and predictions are run.

        Returns
        -------
        results : list
            A list of results, one for each edge embedding method set.
        """
        # Create temporal files with in/out data for method
        tmpedg = './edgelist.tmp'
        tmpemb = './emb.tmp'

        # Write the train data to a file
        data_split.save_tr_graph(tmpedg, delimiter=input_delim, write_stats=False,
                                 write_weights=write_weights, write_dir=write_dir)

        # Add the input, output and embedding dimensionality to the command
        command = command.format(tmpedg, tmpemb, self.dim)

        print('Running command...')
        print(command)

        try:
            # Call the method
            if not verbose:
                devnull = open(os.devnull, 'w')
                subprocess.call(command, shell=True, stdout=devnull, stderr=devnull)
            else:
                subprocess.call(command, shell=True)

            # Some methods append a .txt filetype to the outfile if its the case, read the txt
            if os.path.isfile('./emb.tmp.txt'):
                tmpemb = './emb.tmp.txt'

            # Autodetect header of output
            # Read num lines in output
            num_vectors = sum(1 for _ in open(tmpemb))
            emb_skiprows = num_vectors - len(data_split.TG.nodes)

            if emb_skiprows < 0:
                raise ValueError('Method {} does not provide a unique embedding for every graph node!'
                                 '\nExpected num. node embeddings: {} '
                                 '\nObtained num. node embeddings: {}'
                                 .format(method_name, len(data_split.TG.nodes), num_vectors))
            elif emb_skiprows > 0:
                logging.warning('Output of method {} contains {} more lines than expected. Will consider them part '
                                'of the header and ignore them... Expected num_lines {}, obtained lines {}.'
                                .format(method_name, emb_skiprows, len(data_split.TG.nodes), num_vectors))

            # Read the embeddings
            X = np.genfromtxt(tmpemb, delimiter=output_delim, dtype=float, skip_header=emb_skiprows, autostrip=True)

            if X.ndim == 1:
                raise ValueError('Error encountered while reading node embeddings for method {}. '
                                 'Please check the output delimiter for the method, this value is probably incorrect.'
                                 .format(method_name))

            if X.shape[1] == self.dim:
                # Assume embeddings given as matrix [X_0, X_1, ..., X_D] where rows correspond to sorted node id
                keys = map(str, sorted(data_split.TG.nodes))
                # keys = map(str, range(len(X)))
                X = dict(zip(keys, X))
            elif X.shape[1] == self.dim + 1:
                logging.warning('Output provided by method {} contains {} columns, {} expected!'
                                '\nAssuming first column to be the nodeID...'
                                .format(method_name, X.shape[1], self.dim))
                # Assume first col is node id and rest are embedding features [id, X_0, X_1, ..., X_D]
                keys = map(str, np.array(X[:, 0], dtype=int))
                X = dict(zip(keys, X[:, 1:]))
            else:
                raise ValueError('Incorrect node embedding dimensions for method {}!'
                                 '\nValues expected: {} or {} \nValue received: {}'
                                 .format(method_name, self.dim, self.dim + 1, X.shape[1]))

            # Evaluate the model
            results = list()
            for ee in edge_embedding_methods:
                results.append(self.evaluate_ne(data_split=data_split, X=X, method=method_name, edge_embed_method=ee))
            return results

        except IOError:
            raise IOError('Execution of method `{}` did not generate node embeddings file. \nPossible reasons: '
                          '1) method is not correctly installed or 2) wrong method call or parameters... '
                          '\nSetting verbose=True can provide more information.'.format(method_name))

        finally:
            # Delete the temporal files
            if os.path.isfile(tmpedg):
                os.remove(tmpedg)
            if os.path.isfile(tmpemb):
                os.remove(tmpemb)
            if os.path.isfile('./emb.tmp.txt'):
                os.remove('./emb.tmp.txt')