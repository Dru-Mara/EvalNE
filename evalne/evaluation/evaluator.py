#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# TODO: Implement NC as link prediction for node-pair embedding and end to end predictors.

from __future__ import division

import itertools
import logging
import os
import re
import time

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

from evalne.evaluation import edge_embeddings
from evalne.evaluation import score
from evalne.evaluation import split
from evalne.methods import katz
from evalne.methods import similarity as sim
from evalne.utils import preprocess as pp
from evalne.utils import split_train_test as stt
from evalne.utils import util


def _eval_katz(q, method, traintest_split):
    """
    Helper function that evaluates the Katz method and puts the train and test predictions in a queue object.

    Parameters
    ----------
    q : queue
        An object used to communicate the results of this function to the calling method.
    method : string
        A string indicating the name of the method to evaluate (katz) and the associated beta parameter value if any.
    traintest_split : a subclass of BaseEvalSplit
        A subclass of BaseEvalSplit containing the train graph (a subgraph of the full network that spans all nodes)
        and a set of train edges and non-edges. Test edges are optional. If not provided only train results will be
        generated.
    """
    m = method.split()
    test_pred = None
    if len(m) > 1:
        exact = katz.Katz(traintest_split.TG, float(m[1]))
    else:
        exact = katz.Katz(traintest_split.TG)
    train_pred = exact.predict(traintest_split.train_edges)
    if len(traintest_split.test_edges) != 0:
        test_pred = exact.predict(traintest_split.test_edges)
    q.put((train_pred, test_pred))


def _eval_sim(q, method, traintest_split, neighbourhood):
    """
    Helper function that evaluates a heuristic baseline and puts the train and test predictions in a queue object.

    Parameters
    ----------
    q : queue
        An object used to communicate the results of this function to the calling method.
    method : string
        A string indicating the name of the method to evaluate.
    traintest_split : a subclass of BaseEvalSplit
        A subclass of BaseEvalSplit containing the train graph (a subgraph of the full network that spans all nodes)
        and a set of train edges and non-edges. Test edges are optional. If not provided only train results will be
        generated.
    neighbourhood : string, optional
        A string indicating the 'in' or 'out' neighbourhood to be used for directed graphs. Default is 'in'.
    """
    func = getattr(sim, str(method))
    train_pred = func(traintest_split.TG, traintest_split.train_edges, neighbourhood)
    test_pred = None
    if len(traintest_split.test_edges) != 0:
        test_pred = func(traintest_split.TG, traintest_split.test_edges, neighbourhood)
    q.put((train_pred, test_pred))


class LPEvaluator(object):
    """
    Class designed to simplify the evaluation of embedding methods for link prediction tasks.

    Parameters
    ----------
    traintest_split : LPEvalSplit
        An object containing the train graph (a subgraph of the full network that spans all nodes) and a set of train
        edges and non-edges. Test edges are optional. If not provided only train results will be generated.
    trainvalid_split : LPEvalSplit, optional
        An object containing the validation graph (a subgraph of the training network that spans all nodes) and a set of
        train and valid edges and non-edges. If not provided a split with the same parameters as the train one, but
        with train_frac=0.9, will be computed. Default is None.
    dim : int, optional
        Embedding dimensionality. Default is 128.
    lp_model : Sklearn binary classifier, optional
        The binary classifier to use for prediction. Default is logistic regression with 5 fold cross validation:
        `LogisticRegressionCV(Cs=10, cv=5, penalty='l2', scoring='roc_auc', solver='lbfgs', max_iter=100))`

    Notes
    -----
    In link prediction the aim is to predict, given a set of node pairs, if they should be connected or not. This is
    generally solved as a binary classification task. For training the binary classifier, we sample a set of edges as
    well as a set of unconnected node pairs. We then compute the node-pair embeddings of this training data. We use
    the node-pair embeddings together with the corresponding labels (0 for non-edges and 1 for edges) to train the
    classifier. Finally, the performance is evaluated on the test data (the remaining edges not used in training plus
    another set of randomly selected non-edges).

    Examples
    --------
    Instantiating an LPEvaluator without a specific train/validation split (this split will be computed automatically if
    parameter tuning for any method is required):

    >>> from evalne.evaluation.evaluator import LPEvaluator
    >>> from evalne.evaluation.split import LPEvalSplit
    >>> from evalne.utils import preprocess as pp
    >>> # Load and preprocess a network
    >>> G = pp.load_graph('./evalne/tests/data/network.edgelist')
    >>> G, _ = pp.prep_graph(G)
    >>> # Create the required train/test split
    >>> traintest_split = LPEvalSplit()
    >>> _ = traintest_split.compute_splits(G)
    >>> # Initialize the LPEvaluator
    >>> nee = LPEvaluator(traintest_split)

    Instantiating an LPEvaluator with a specific train/validation split (allows the user to specify any parameters
    for the train/validation split):

    >>> from evalne.evaluation.evaluator import LPEvaluator
    >>> from evalne.evaluation.split import LPEvalSplit
    >>> from evalne.utils import preprocess as pp
    >>> # Load and preprocess a network
    >>> G = pp.load_graph('./evalne/tests/data/network.edgelist')
    >>> G, _ = pp.prep_graph(G)
    >>> # Create the required train/test split
    >>> traintest_split = LPEvalSplit()
    >>> _ = traintest_split.compute_splits(G)
    >>> # Create the train/validation split from the train data computed in the trintest_split
    >>> # The graph used to initialize this split must, thus, be the train graph from the traintest_split
    >>> trainvalid_split = EvalSplit()
    >>> _ = trainvalid_split.compute_splits(traintest_split.TG)
    >>> # Initialize the LPEvaluator
    >>> nee = LPEvaluator(traintest_split, trainvalid_split)

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
        """
        Initializes the train/validation EvalSplit.
        """
        if self.trainvalid_split is None or len(self.trainvalid_split.test_edges) == 0:
            logging.warning('No test edges in trainvalid_split. Recomputing correct split...')
        self.trainvalid_split = split.EvalSplit()
        self.trainvalid_split.compute_splits(self.traintest_split.TG, nw_name=self.traintest_split.nw_name,
                                             train_frac=0.9, split_alg=self.traintest_split.split_alg,
                                             owa=self.traintest_split.owa, fe_ratio=self.traintest_split.fe_ratio,
                                             split_id=self.traintest_split.split_id, verbose=False)

    @staticmethod
    def _log_best(best_results, best_params, results, params, maximize, tr_te='test'):
        """
        Keeps track of the best evaluation results and corresponding parameters individually for each node-pair
        embedding method being evaluated. Updates the best results if needed.

        Parameters
        ----------
        best_results : list
            A list of Results objects, one for each node-pair embedding method being evaluated, containing the best
            Results.
        best_params : list
            A list of strings, one for each node-pair embedding method being evaluated, containing the parameter names
            and their associated values used to compute the best Results.
        results : list
            A list of new Results objects, one for each node-pair embedding method being evaluated.
        params : string
            A string containing the parameter names and their associated values used to compute the new `results`.
        maximize : string
            The score to check while comparing results objects.
        tr_te : string
            A string indicating if the 'train' or 'test' results should be checked.

        Returns
        -------
        best_results : list
            A list of Results objects, one for each node-pair embedding method being evaluated, containing the best
            Results.
        best_params : list
            A list of strings, one for each node-pair embedding method being evaluated, containing the parameter names
            and their associated values used to compute the best Results.
        """
        # Log the best results
        for j in range(len(results)):
            if best_results[j] is None:
                best_results[j] = results[j]
                best_params[j] = params
            else:
                if tr_te == 'train':
                    func1 = getattr(results[j].train_scores, str(maximize))
                    func2 = getattr(best_results[j].train_scores, str(maximize))
                else:
                    func1 = getattr(results[j].test_scores, str(maximize))
                    func2 = getattr(best_results[j].test_scores, str(maximize))
                if func1() > func2():
                    best_results[j] = results[j]
                    best_params[j] = params

        return best_results, best_params

    def evaluate_baseline(self, method, neighbourhood='in', timeout=None):
        """
        Evaluates the baseline method requested. Evaluation output is returned as a Results object. For Katz
        neighbourhood=`in` and neighbourhood=`out` will return the same results corresponding to neighbourhood=`in`.
        Execution time is contained in the results object. If the train/test split object used to initialize the
        evaluator does not contain test edges, the results object will only contain train results.

        Parameters
        ----------
        method : string
            A string indicating the name of any baseline from evalne.methods to evaluate.
        neighbourhood : string, optional
            A string indicating the 'in' or 'out' neighbourhood to be used for directed graphs. Default is 'in'.
        timeout : float or None
            A float indicating the maximum amount of time (in seconds) the evaluation can run for. If None, the
            evaluation is allowed to continue until completion. Default is None.

        Returns
        -------
        results : Results
            The evaluation results as a Results object.

        Raises
        ------
        TimeoutExpired
            If the execution does not finish within the allocated time.
        TypeError
            If the Katz method call is incorrect.
        ValueError
            If the heuristic selected does not exist.

        See Also
        --------
        evalne.utils.util.run_function : The low level function used to run a baseline with given timeout.

        Examples
        --------
        Evaluating the common neighbours heuristic with default parameters. We assume an evaluator (nee) has already
        been instantiated (see class examples):

        >>> result = nee.evaluate_baseline(method='common_neighbours')
        >>> # Print the results
        >>> result.pretty_print()
        Method: common_neighbours
        Parameters:
        [('split_id', 0), ('dim', 128), ('eval_time', 0.06909489631652832), ('neighbourhood', 'in'),
        ('split_alg', 'spanning_tree'), ('fe_ratio', 1.0), ('owa', True), ('nw_name', 'test'),
        ('train_frac', 0.510061919504644)]
        Test scores:
        tn = 1124
        [...]

        Evaluating katz with beta=0.05 and timeout 60 seconds. We assume an evaluator (nee) has already
        been instantiated (see class examples):

        >>> result = nee.evaluate_baseline(method='katz 0.05', timeout=60)
        >>> # Print the results
        >>> result.pretty_print()
        Method: katz 0.05
        Parameters:
        [('split_id', 0), ('dim', 128), ('eval_time', 0.11670708656311035), ('neighbourhood', 'in'),
        ('split_alg', 'spanning_tree'), ('fe_ratio', 1.0), ('owa', True), ('nw_name', 'test'),
        ('train_frac', 0.510061919504644)]
        Test scores:
        tn = 1266
        [...]

        """
        # Measure execution time
        start = time.time()

        if 'katz' in method.lower():
            try:
                train_pred, test_pred = util.run_function(timeout, _eval_katz, *[method, self.traintest_split])
            except TypeError:
                raise TypeError('Call to katz method is incorrect. Check method parameters.')
            except util.TimeoutExpired as e:
                raise util.TimeoutExpired('Method `{}` timed out after {} seconds'.format(method, time.time()-start))

        else:
            try:
                train_pred, test_pred = util.run_function(timeout, _eval_sim,
                                                          *[method, self.traintest_split, neighbourhood])
            except AttributeError:
                raise AttributeError('Method `{}` is not one of the available baselines!'.format(method))
            except util.TimeoutExpired as e:
                raise util.TimeoutExpired('Method `{}` timed out after {} seconds'.format(method, time.time()-start))

        # Make predictions column vectors
        train_pred = np.array(train_pred)
        if test_pred is not None:
            test_pred = np.array(test_pred)

        # End of exec time measurement
        end = time.time() - start

        # Set some parameters for the results object
        params = {'neighbourhood': neighbourhood, 'eval_time': end}
        self.edge_embed_method = None

        if 'all_baselines' in method:
            # This method returns node-pair embeddings so we need to compute the predictions
            train_pred, test_pred = self.compute_pred(data_split=self.traintest_split, tr_edge_embeds=train_pred,
                                                      te_edge_embeds=test_pred)

        # Compute the scores
        if nx.is_directed(self.traintest_split.TG):
            results = self.compute_results(data_split=self.traintest_split, method_name=method + '-' + neighbourhood,
                                           train_pred=train_pred, test_pred=test_pred, params=params)
        else:
            results = self.compute_results(data_split=self.traintest_split, method_name=method,
                                           train_pred=train_pred, test_pred=test_pred, params=params)

        return results

    def evaluate_cmd(self, method_name, method_type, command, edge_embedding_methods, input_delim, output_delim,
                     tune_params=None, maximize='auroc', write_weights=False, write_dir=False, timeout=None,
                     verbose=True):
        """
        Evaluates an embedding method and tunes its parameters from the method's command line call string. This
        function can evaluate node embedding, node-pair embedding or end to end predictors.

        Parameters
        ----------
        method_name : string
            A string indicating the name of the method to be evaluated.
        method_type : string
            A string indicating the type of embedding method (i.e. ne, ee, e2e).
            NE methods are expected to return embeddings, one per graph node, as either dict or matrix sorted by nodeID.
            EE methods are expected to return node-pair emb. as [num_edges x embed_dim] matrix in same order as input.
            E2E methods are expected to return predictions as a vector in the same order as the input edgelist.
        command : string
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
            A list of methods used to compute node-pair embeddings from the node embeddings output by NE models.
            The accepted values are the function names in evalne.evaluation.edge_embeddings.
            When evaluating 'ee' or 'e2e' methods, this parameter is ignored.
        input_delim : string
            The delimiter expected by the method as input (edgelist).
        output_delim : string
            The delimiter provided by the method in the output.
        tune_params : string, optional
            A string containing all the parameters to be tuned and their values. Default is None.
        maximize : string, optional
            The score to maximize while performing parameter tuning. Default is 'auroc'.
        write_weights : bool, optional
            If True the train graph passed to the embedding methods will be stored as weighted edgelist
            (e.g. triplets src, dst, weight) otherwise as normal edgelist. If the graph edges have no weight attribute
            and this parameter is set to True, a weight of 1 will be assigned to each edge. Default is False.
        write_dir : bool, optional
            This option is only relevant for undirected graphs. If False, the train graph will be stored with a single
            direction of the edges. If True, both directions of edges will be stored. Default is False.
        timeout : float or None, optional
            A float indicating the maximum amount of time (in seconds) the evaluation can run for. If None, the
            evaluation is allowed to continue until completion. Default is None.
        verbose : bool, optional
            A parameter to control the amount of screen output. Default is True.

        Returns
        -------
        results : Results
            Returns the evaluation results as a Results object.

        Raises
        ------
        TimeoutExpired
            If the execution does not finish within the allocated time.
        IOError
            If the method call does not succeed.
        ValueError
            If the method type is unknown.
            If for a method all parameter combinations fail to provide results.

        See Also
        --------
        evalne.utils.util.run : The low level function used to run a cmd call with given timeout.

        Examples
        --------
        Evaluating the OpenNE implementation of node2vec without parameter tuning and with 'average' and 'hadamard' as
        node-pair embedding operators. We assume the method is installed in a virtual environment and that an evaluator
        (nee) has already been instantiated (see class examples):

        >>> # Prepare the cmd command for running the method. If running on a python console full paths are required
        >>> cmd = '../OpenNE-master/venv/bin/python -m openne --method node2vec '\
        ...       '--graph-format edgelist --input {} --output {} --representation-size {}'
        >>> # Call the evaluation
        >>> result = nee.evaluate_cmd(method_name='Node2vec', method_type='ne', command=cmd,
        ...                          edge_embedding_methods=['average', 'hadamard'], input_delim=' ', output_delim=' ')
        Running command...
        [...]
        >>> # Print the results
        >>> result.pretty_print()
        Method: Node2vec
        Parameters:
        [('split_id', 0), ('dim', 128), ('owa', True), ('nw_name', 'test'), ('train_frac', 0.51),
        ('split_alg', 'spanning_tree'), ('eval_time', 24.329686164855957), ('edge_embed_method', 'average'),
        ('fe_ratio', 1.0)]
        Test scores:
        tn = 913
        [...]

        Evaluating the metapath2vec c++ implementation with parameter tuning and with 'average' node-pair embedding
        operator. We assume the method is installed and that an evaluator (nee) has already been instantiated
        (see class examples):

        >>> # Prepare the cmd command for running the method. If running on a python console full paths are required
        >>> cmd = '../../methods/metapath2vec/metapath2vec -min-count 1 -iter 20 '\
        ...       '-samples 100 -train {} -output {} -size {}'
        >>> # Call the evaluation
        >>> result = nee.evaluate_cmd(method_name='Metapath2vec', method_type='ne', command=cmd,
        ...                          edge_embedding_methods=['average'], input_delim=' ', output_delim=' ')
        Running command...
        [...]
        >>> # Print the results
        >>> result.pretty_print()
        Method: Metapath2vec
        Parameters:
        [('split_id', 0), ('dim', 128), ('owa', True), ('nw_name', 'test'), ('train_frac', 0.51),
        ('split_alg', 'spanning_tree'), ('eval_time', 1.9907279014587402), ('edge_embed_method', 'average'),
        ('fe_ratio', 1.0)]
        Test scores:
        tn = 919
        [...]

        """
        # Measure execution time
        start = time.time()
        if timeout is None:
            timeout = 31536000

        # Check if a validation set needs to be initialized
        if self.trainvalid_split is None or len(self.trainvalid_split.test_edges) == 0:
            self._init_trainvalid()

        # Check the method type and raise an error if necessary
        if method_type not in ['ne', 'ee', 'e2e']:
            raise ValueError('Method type `{}` of method `{}` is unknown! Valid options are: `ne`, `ee`, `e2e`'
                             .format(method_type, method_name))

        # If the method evaluated does not require node-pair embeddings set this parameter to ['none']
        if method_type != 'ne':
            edge_embedding_methods = ['none']
            self.edge_embed_method = None

        # Check if tuning parameters is needed
        if tune_params is not None:
            print('Tuning parameters for {} ...'.format(method_name))

            # Variable to store the best results and parameters for each ee_method
            best_results = list()
            best_params = list()
            for j in range(len(edge_embedding_methods)):
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
                                                                write_dir, timeout-(time.time()-start), verbose)
                        else:
                            results = self._evaluate_ne_cmd(self.trainvalid_split, method_name, ext_command,
                                                            edge_embedding_methods, input_delim, output_delim,
                                                            write_weights, write_dir, timeout-(time.time()-start),
                                                            verbose)
                        results = list(results)

                        # Log the best results
                        best_results, best_params = self._log_best(best_results, best_params, results, param_str,
                                                                   maximize)

                    except (ValueError, IOError, util.TimeoutExpired) as e:
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
                                                                write_dir, timeout-(time.time()-start), verbose)
                        else:
                            results = self._evaluate_ne_cmd(self.trainvalid_split, method_name, ext_command,
                                                            edge_embedding_methods, input_delim, output_delim,
                                                            write_weights, write_dir, timeout-(time.time()-start),
                                                            verbose)
                        results = list(results)

                        # Log the best results
                        best_results, best_params = self._log_best(best_results, best_params, results, param_str,
                                                                   maximize)

                    except (ValueError, IOError, util.TimeoutExpired) as e:
                        logging.exception('Exception occurred while evaluating param `{}` for method `{}` on `{}`.'
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
            best_ee_idx = int(np.argmax(ee_scores))
            if ee_scores[best_ee_idx] == 0.0:
                raise ValueError('All parameter combinations for method `{}` have failed! No results available.'
                                 .format(method_name))
            ext_command = command + best_params[best_ee_idx]

            # Call the corresponding evaluation method on the whole train data for the selected ee method
            if method_type == 'ee' or method_type == 'e2e':
                results = self._evaluate_ee_e2e_cmd(self.traintest_split, method_name, method_type, ext_command,
                                                    input_delim, output_delim, write_weights, write_dir,
                                                    timeout-(time.time()-start), verbose)
            else:
                results = self._evaluate_ne_cmd(self.traintest_split, method_name, ext_command,
                                                [edge_embedding_methods[best_ee_idx]], input_delim, output_delim,
                                                write_weights, write_dir, timeout-(time.time()-start), verbose)

            # # A more efficient approach that does not recompute the embeddings for each ee method
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
                                                    input_delim, output_delim, write_weights, write_dir,
                                                    timeout - (time.time() - start), verbose)
            else:
                # We still have to tune the node-pair embedding method
                if len(edge_embedding_methods) > 1:
                    # For NE methods first compute the results on validation data
                    valid_results = self._evaluate_ne_cmd(self.trainvalid_split, method_name, command,
                                                          edge_embedding_methods, input_delim, output_delim,
                                                          write_weights, write_dir, timeout-(time.time()-start),
                                                          verbose=False)

                    # Extract and log the validation scores
                    ee_scores = list()
                    for i in range(len(valid_results)):
                        func = getattr(valid_results[i].test_scores, str(maximize))
                        bestscore = func()
                        ee_scores.append(bestscore)
                        logging.info('Validation score for method `{}_{}` is: {}, no other tuned params.'
                                     .format(method_name, edge_embedding_methods[i], bestscore))

                    # We now select the ee that performs best in terms of maximize score
                    best_ee_idx = int(np.argmax(ee_scores))
                else:
                    # If we only have one ee method then that the one we compute results for, no need for validation
                    best_ee_idx = 0

                # Compute the results on the full train split
                results = self._evaluate_ne_cmd(self.traintest_split, method_name, command,
                                                [edge_embedding_methods[best_ee_idx]], input_delim, output_delim,
                                                write_weights, write_dir, timeout, verbose)

        # End of exec time measurement
        end = time.time() - start
        res = results[0]
        res.params.update({'eval_time': end})

        # Return the evaluation results
        return res

    def _evaluate_ne_cmd(self, data_split, method_name, command, edge_embedding_methods, input_delim, output_delim,
                         write_weights, write_dir, timeout, verbose):
        """
        The actual implementation of the node embedding evaluation. Stores the train graph as an edgelist to a
        temporal file and provides it as input to the method evaluated. Performs the command line call and reads
        the output. Node embeddings are transformed to node-pair embeddings and predictions are run.

        Returns
        -------
        results : list
            A list of results, one for each node-pair embedding method set.
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
            util.run(command, timeout, verbose)

            # Some methods append a .txt filetype to the outfile if its the case, read the txt
            if os.path.isfile('./emb.tmp.txt'):
                tmpemb = './emb.tmp.txt'

            # Read embeddings from output file
            X = pp.read_node_embeddings(tmpemb, data_split.TG.nodes, self.dim, output_delim, method_name)

            # Evaluate the model
            results = list()
            for ee in edge_embedding_methods:
                results.append(self.evaluate_ne(data_split=data_split, X=X, method=method_name, edge_embed_method=ee))
            return results

        except (IOError, OSError):
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
                             write_weights, write_dir, timeout, verbose):
        """
        The actual implementation of the node-pair embedding and end to end evaluation. Stores the train graph as an
        edgelist to a temporal file and provides it as input to the method evaluated together with the train and
        test edge sets. Performs the command line method call and reads the output node-pair embeddings/predictions.
        The method results are then computed according to the method type and returned.
        If no test edges are required, we still pass two dummy ones to the methods to prevent them from failing.

        Returns
        -------
        results : list
            A list with a single element, the result for the user defined node-pair embedding method.
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
            raise ValueError('Incorrect number of placeholders in `{}` command! Accepted values are 4 or 6.'
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
            util.run(command, timeout, verbose)

            if placeholders == 4:
                # Check if the method is ee or e2e
                if method_type == 'ee':
                    Y = pp.read_edge_embeddings(tmp_tr_out, (len(data_split.train_edges) + len(data_split.test_edges)),
                                                self.dim, output_delim, method_name)
                    tr_out = Y[0:len(data_split.train_edges), :]
                    te_out = Y[len(data_split.train_edges):, :]
                else:
                    Y = pp.read_predictions(tmp_tr_out, (len(data_split.train_edges) + len(data_split.test_edges)),
                                            output_delim, method_name)
                    tr_out = Y[0:len(data_split.train_edges)]
                    te_out = Y[len(data_split.train_edges):]

            else:
                # Check if the method is ee or e2e
                if method_type == 'ee':
                    tr_out = pp.read_edge_embeddings(tmp_tr_out, len(data_split.train_edges), self.dim, output_delim,
                                                     method_name)
                    te_out = pp.read_edge_embeddings(tmp_te_out, len(data_split.test_edges) + dummy_edges, self.dim,
                                                     output_delim, method_name)
                else:
                    tr_out = pp.read_predictions(tmp_tr_out, len(data_split.train_edges), output_delim,
                                                 method_name)
                    te_out = pp.read_predictions(tmp_te_out, len(data_split.test_edges) + dummy_edges,
                                                 output_delim, method_name)

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

        except (IOError, OSError):
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
        """
        Runs the complete pipeline, from node embeddings to node-pair embeddings and returns the prediction results.
        If data_split.test_edges is None, the Results object will only contain train Scores.

        Parameters
        ----------
        data_split : a subclass of BaseEvalSplit
            A subclass of BaseEvalSplit object that encapsulates the train/test or train/validation data.
        X : dict
            A dictionary where keys are nodes in the graph and values are the node embeddings.
            The keys are of type string and the values of type array.
        method : string
            A string indicating the name of the method to be evaluated.
        edge_embed_method : string
            A string indicating the method used to compute node-pair embeddings from node embeddings.
            The accepted values are any of the function names in evalne.evaluation.edge_embeddings.
        label_binarizer : string or Sklearn binary classifier, optional
            If the predictions returned by the model are not binary, this parameter indicates how these binary
            predictions should be computed in order to be able to provide metrics such as the confusion matrix.
            Any Sklear binary classifier can be used or the keyword 'median' which will used the prediction medians
            as binarization thresholds. Default is `LogisticRegression(solver='liblinear')`.
        params : dict, optional
            A dictionary of parameters and values to be added to the results class. Default is None.

        Returns
        -------
        results : Results
            A results object.
        """
        # Run the evaluation pipeline
        tr_edge_embeds, te_edge_embeds = self.compute_ee(data_split, X, edge_embed_method)
        train_pred, test_pred = self.compute_pred(data_split, tr_edge_embeds, te_edge_embeds)

        return self.compute_results(data_split=data_split, method_name=method, train_pred=train_pred,
                                    test_pred=test_pred, label_binarizer=label_binarizer, params=params)

    def compute_ee(self, data_split, X, edge_embed_method):
        """
        Computes node-pair embeddings using the given node embeddings dictionary and node-pair embedding method.
        If data_split.test_edges is None, te_edge_embeds will be None.

        Parameters
        ----------
        data_split : a subclass of BaseEvalSplit
            A subclass of BaseEvalSplit object that encapsulates the train/test or train/validation data.
        X : dict
            A dictionary where keys are nodes in the graph and values are the node embeddings.
            The keys are of type string and the values of type array.
        edge_embed_method : string
            A string indicating the method used to compute node-pair embeddings from node embeddings.
            The accepted values are any of the function names in evalne.evaluation.edge_embeddings.

        Returns
        -------
        tr_edge_embeds : matrix
            A Numpy matrix containing the train node-pair embeddings.
        te_edge_embeds : matrix
            A Numpy matrix containing the test node-pair embeddings. Returns None if data_split.test_edges is None.

        Raises
        ------
        AttributeError
            If the node-pair embedding operator selected is not valid.
        """
        self.edge_embed_method = edge_embed_method

        try:
            func = getattr(edge_embeddings, str(edge_embed_method))
        except AttributeError:
            raise AttributeError('Node-pair embedding method `{}` is not a valid option.'.format(edge_embed_method))

        tr_edge_embeds = func(X, data_split.train_edges)
        if len(data_split.test_edges) != 0:
            te_edge_embeds = func(X, data_split.test_edges)
            return tr_edge_embeds, te_edge_embeds
        else:
            return tr_edge_embeds, None

    def compute_pred(self, data_split, tr_edge_embeds, te_edge_embeds=None):
        """
        Computes predictions from the given node-pair embeddings. Trains an LP model with the train node-pair
        embeddings and performs predictions for train and test node-pair embeddings. If te_edge_embeds is None
        test_pred will be None.

        Parameters
        ----------
        data_split : a subclass of BaseEvalSplit
            A subclass of BaseEvalSplit object that encapsulates the train/test or train/validation data.
        tr_edge_embeds : matrix
            A Numpy matrix containing the train node-pair embeddings.
        te_edge_embeds : matrix, optional
            A Numpy matrix containing the test node-pair embeddings. Default is None.

        Returns
        -------
        train_pred : array
            The predictions for the train data.
        test_pred : array
            The predictions for the test data. Returns None if te_edge_embeds is None.
        """
        # Train the LP model
        self.lp_model.fit(tr_edge_embeds, data_split.train_labels)

        # Predict
        try:
            train_pred = self.lp_model.predict_proba(tr_edge_embeds)[:, 1]
            test_pred = None
            if te_edge_embeds is not None:
                test_pred = self.lp_model.predict_proba(te_edge_embeds)[:, 1]
        except AttributeError:
            logging.warning('Selected classifier does not have a `predict_proba` method... trying to call `predict`')
            train_pred = self.lp_model.predict(tr_edge_embeds)
            test_pred = None
            if te_edge_embeds is not None:
                test_pred = self.lp_model.predict(te_edge_embeds)

        # Return the predictions
        return train_pred, test_pred

    def compute_results(self, data_split, method_name, train_pred, test_pred=None,
                        label_binarizer=LogisticRegression(solver='liblinear'), params=None):
        """
        Generates results from the given predictions and returns them. If test_pred is not provided, the Results
        object will only contain the train scores.

        Parameters
        ----------
        data_split : a subclass of BaseEvalSplit
            A subclass of BaseEvalSplit object that encapsulates the train/test or train/validation data.
        method_name : string
            A string indicating the name of the method for which the results will be created.
        train_pred :
            The predictions for the train data.
        test_pred : array_like, optional
            The predictions for the test data. Default is None.
        label_binarizer : string or Sklearn binary classifier, optional
            If the predictions returned by the model are not binary, this parameter indicates how these binary
            predictions should be computed in order to be able to provide metrics such as the confusion matrix.
            Any Sklear binary classifier can be used or the keyword 'median' which will used the prediction medians
            as binarization thresholds.
            Default is LogisticRegression(solver='liblinear')
        params : dict, optional
            A dictionary of parameters and values to be added to the results class. Default is None.

        Returns
        -------
        results : Results
            The evaluation results.
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
                                    test_pred=test_pred, test_labels=data_split.test_labels,
                                    label_binarizer=label_binarizer)
        return results


class NREvaluator(LPEvaluator):
    """
    Class designed to simplify the evaluation of embedding methods for network reconstruction tasks.
    The train graph is assumed to be the entire network. Parameter tuning is performed directly on this complete graph.

    Parameters
    ----------
    traintest_split : NREvalSplit
        An object containing the train graph (in this case the full network) and a set of train edges and non-edges.
        These edges can be all edges in the graph or a subset.
    dim : int, optional
        Embedding dimensionality. Default is 128.
    lp_model : Sklearn binary classifier, optional
        The binary classifier to use for prediction. Default is logistic regression with 5 fold cross validation:
        `LogisticRegressionCV(Cs=10, cv=5, penalty='l2', scoring='roc_auc', solver='lbfgs', max_iter=100))`

    Notes
    -----
    In network reconstruction the aim is to asses how well an embedding method captures the structure of a given graph.
    The embedding methods are trained on a complete input graph. Hyperparameter tuning is performed directly on this
    graph (overfitting is, in this case, expected and desired). The embeddings obtained are used to perform link
    predictions and their quality is evaluated. Checking the link predictions for all node pairs is generally
    unfeasible, therefore a subset of all node pairs in the input graph are selected for evaluation.

    Examples
    --------
    Instantiating an NREvaluator with default parameters (for this task train/validation splits are not necessary):

    >>> from evalne.evaluation.evaluator import NREvaluator
    >>> from evalne.evaluation.split import NREvalSplit
    >>> from evalne.utils import preprocess as pp
    >>> # Load and preprocess a network
    >>> G = pp.load_graph('./evalne/tests/data/network.edgelist')
    >>> G, _ = pp.prep_graph(G)
    >>> # Create the required train/test split
    >>> traintest_split = NREvalSplit()
    >>> _ = traintest_split.compute_splits(G)
    >>> # Initialize the NREvaluator
    >>> nee = NREvaluator(traintest_split)

    Instantiating an NREvaluator where we randomly select 10% of all node pairs in the network for evaluation:

    >>> from evalne.evaluation.evaluator import NREvaluator
    >>> from evalne.evaluation.split import NREvalSplit
    >>> from evalne.utils import preprocess as pp
    >>> # Load and preprocess a network
    >>> G = pp.load_graph('./evalne/tests/data/network.edgelist')
    >>> G, _ = pp.prep_graph(G)
    >>> # Create the required train/test split and sample 0.1, i.e. 10% of all nodes
    >>> traintest_split = NREvalSplit()
    >>> _ = traintest_split.compute_splits(G, samp_frac=0.1)
    >>> # Initialize the NREvaluator
    >>> nee = NREvaluator(traintest_split)

    """

    def __init__(self, traintest_split, dim=128,
                 lp_model=LogisticRegressionCV(Cs=10, cv=5, penalty='l2', scoring='roc_auc', solver='lbfgs',
                                               max_iter=100)):
        # General evaluation parameters
        super(NREvaluator, self).__init__(traintest_split, dim=dim, lp_model=lp_model)

    def _check_split(self):
        """
        Checks that only train edges are provided in the traintest_split object and raises an error if not. For network
        reconstruction the entire evaluation is performed on the train edges only.

        Raises
        ------
        ValueError
            If the traintest_split used to initialize the class contains any test edges.
        """
        if self.traintest_split.test_edges is not None:
            raise ValueError('For network reconstruction test edges need to be set to None!')

    def evaluate_cmd(self, method_name, method_type, command, edge_embedding_methods, input_delim, output_delim,
                     tune_params=None, maximize='auroc', write_weights=False, write_dir=False, timeout=None,
                     verbose=True):
        """
        Evaluates an embedding method and tunes its parameters from the method's command line call string. This
        function can evaluate node embedding, node-pair embedding or end to end predictors. If model parameter tuning
        is required, models are tuned directly on the train data. The returned Results object will only contain
        train scores.

        Parameters
        ----------
        method_name : string
            A string indicating the name of the method to be evaluated.
        method_type : string
            A string indicating the type of embedding method (i.e. ne, ee, e2e).
            NE methods are expected to return embeddings, one per graph node, as either dict or matrix sorted by nodeID.
            EE methods are expected to return node-pair emb. as [num_edges x embed_dim] matrix in same order as input.
            E2E methods are expected to return predictions as a vector in the same order as the input edgelist.
        command : string
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
            A list of methods used to compute node-pair embeddings from the node embeddings output by NE models.
            The accepted values are the function names in evalne.evaluation.edge_embeddings.
            When evaluating 'ee' or 'e2e' methods, this parameter is ignored.
        input_delim : string
            The delimiter expected by the method as input (edgelist).
        output_delim : string
            The delimiter provided by the method in the output.
        tune_params : string, optional
            A string containing all the parameters to be tuned and their values. Default is None.
        maximize : string, optional
            The score to maximize while performing parameter tuning. Default is 'auroc'.
        write_weights : bool, optional
            If True the train graph passed to the embedding methods will be stored as weighted edgelist
            (e.g. triplets src, dst, weight) otherwise as normal edgelist. If the graph edges have no weight attribute
            and this parameter is set to True, a weight of 1 will be assigned to each edge. Default is False.
        write_dir : bool, optional
            This option is only relevant for undirected graphs. If False, the train graph will be stored with a single
            direction of the edges. If True, both directions of edges will be stored. Default is False.
        timeout : float or None, optional
            A float indicating the maximum amount of time (in seconds) the evaluation can run for. If None, the
            evaluation is allowed to continue until completion. Default is None.
        verbose : bool, optional
            A parameter to control the amount of screen output. Default is True.

        Returns
        -------
        results : Results
            The evaluation results as a Results object.

        Raises
        ------
        TimeoutExpired
            If the execution does not finish within the allocated time.
        IOError
            If the method call does not succeed.
        ValueError
            If the method type is unknown.
            If for a method all parameter combinations fail to provide results.

        See Also
        --------
        evalne.utils.util.run : The low level function used to run a cmd call with given timeout.

        Examples
        --------
        Evaluating the OpenNE implementation of node2vec without parameter tuning and with 'average' and 'hadamard' as
        node-pair embedding operators. We assume the method is installed in a virtual environment and that an evaluator
        (nee) has already been instantiated (see class examples):

        >>> # Prepare the cmd command for running the method. If running on a python console full paths are required
        >>> cmd = '../OpenNE-master/venv/bin/python -m openne --method node2vec '\
        ...       '--graph-format edgelist --input {} --output {} --representation-size {}'
        >>> # Call the evaluation
        >>> result = nee.evaluate_cmd(method_name='Node2vec', method_type='ne', command=cmd,
        ...                          edge_embedding_methods=['average', 'hadamard'], input_delim=' ', output_delim=' ')
        Running command...
        [...]
        >>> # Print the results
        >>> result.pretty_print()
        Method: Node2vec
        Parameters:
        [('split_id', 0), ('dim', 128), ('eval_time', 21.773473024368286), ('nw_name', 'test'),
        ('split_alg', 'random_edge_sample'), ('train_frac', 1), ('edge_embed_method', 'hadamard'), ('samp_frac', 0.01)]
        Train scores:
        tn = 2444
        [...]

        Evaluating the metapath2vec c++ implementation with parameter tuning and with 'average' node-pair embedding
        operator. We assume the method is installed and that an evaluator (nee) has already been instantiated
        (see class examples):

        >>> # Prepare the cmd command for running the method. If running on a python console full paths are required
        >>> cmd = '../../methods/metapath2vec/metapath2vec -min-count 1 -iter 20 '\
        ...       '-samples 100 -train {} -output {} -size {}'
        >>> # Call the evaluation
        >>> result = nee.evaluate_cmd(method_name='Metapath2vec', method_type='ne', command=cmd,
        ...                          edge_embedding_methods=['average'], input_delim=' ', output_delim=' ')
        Running command...
        [...]
        >>> # Print the results
        >>> result.pretty_print()
        Method: Metapath2vec
        Parameters:
        Method: Metapath2vec
        Parameters:
        [('split_id', 0), ('dim', 128), ('eval_time', 1.948814868927002), ('nw_name', 'test'),
        ('split_alg', 'random_edge_sample'), ('train_frac', 1), ('edge_embed_method', 'average'), ('samp_frac', 0.01)]
        Train scores:
        tn = 2444
        [...]

        """
        # Measure execution time
        start = time.time()
        if timeout is None:
            timeout = 31536000

        # Check the method type and raise an error if necessary
        if method_type not in ['ne', 'ee', 'e2e']:
            raise ValueError('Method type `{}` of method `{}` is unknown! Valid options are: `ne`, `ee`, `e2e`'
                             .format(method_type, method_name))

        # If the method evaluated does not require node-pair embeddings set this parameter to ['none']
        if method_type != 'ne':
            edge_embedding_methods = ['none']
            self.edge_embed_method = None

        # Check if tuning parameters is needed
        if tune_params is not None:
            print('Tuning parameters for {} ...'.format(method_name))

            # Variable to store the best results and parameters for each ee_method
            best_results = list()
            best_params = list()
            for j in range(len(edge_embedding_methods)):
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
                                                                write_dir, timeout-(time.time()-start), verbose)
                        else:
                            results = self._evaluate_ne_cmd(self.traintest_split, method_name, ext_command,
                                                            edge_embedding_methods, input_delim, output_delim,
                                                            write_weights, write_dir, timeout-(time.time()-start),
                                                            verbose)
                        results = list(results)

                        # Log the best results
                        best_results, best_params = self._log_best(best_results, best_params, results, param_str,
                                                                   maximize, 'train')

                    except (ValueError, IOError, util.TimeoutExpired) as e:
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
                                                                write_dir, timeout-(time.time()-start), verbose)
                        else:
                            results = self._evaluate_ne_cmd(self.traintest_split, method_name, ext_command,
                                                            edge_embedding_methods, input_delim, output_delim,
                                                            write_weights, write_dir, timeout-(time.time()-start),
                                                            verbose)
                        results = list(results)

                        # Log the best results
                        best_results, best_params = self._log_best(best_results, best_params, results, param_str,
                                                                   maximize, 'train')

                    except (ValueError, IOError, util.TimeoutExpired) as e:
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
            best_ee_idx = int(np.argmax(ee_scores))
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
                                                    input_delim, output_delim, write_weights, write_dir,
                                                    timeout-(time.time()-start), verbose)
            else:
                # For NE methods we still have to tune the node-pair embedding method
                results = self._evaluate_ne_cmd(self.traintest_split, method_name, command,
                                                edge_embedding_methods, input_delim, output_delim,
                                                write_weights, write_dir, timeout-(time.time()-start), verbose)

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
                best_ee_idx = int(np.argmax(ee_scores))
                results = [results[best_ee_idx]]

        # End of exec time measurement
        end = time.time() - start
        res = results[0]
        res.params.update({'eval_time': end})

        # Return the evaluation results
        return res


class SPEvaluator(LPEvaluator):
    """
    Class designed to simplify the evaluation of embedding methods for sign prediction tasks.
    The train and validation graphs are assumed to be weighted and contain positive and negative edges. This is a simple
    extension of an LP evaluation which overrides the baseline implementation to work for sign prediction.

    Parameters
    ----------
    traintest_split : SPEvalSplit
        An object containing the train graph (a subgraph of the full network that spans all nodes) and a set of train
        positive and negative edges. Test edges are optional. If not provided only train results will be generated.
    trainvalid_split : SPEvalSplit, optional
        An object containing the validation graph (a subgraph of the training network that spans all nodes) and a set of
        positive and negative edges. If not provided a split with the same parameters as the train one, but
        with train_frac=0.9, will be computed. Default is None.
    dim : int, optional
        Embedding dimensionality. Default is 128.
    lp_model : Sklearn binary classifier, optional
        The binary classifier to use for prediction. Default is logistic regression with 5 fold cross validation:
        `LogisticRegressionCV(Cs=10, cv=5, penalty='l2', scoring='roc_auc', solver='lbfgs', max_iter=100))`.

    Notes
    -----
    In sign prediction the aim is to predict the sign (positive or negative) of given edges. The existence of the edges
    is assumed (i.e. we do not predict the sign of unconnected node pairs). Therefore, sign prediction is also a binary
    classification task similar to link prediction where, instead of predicting the existence of edges or not, we
    predict the signs for edges we know exist. Unlike for link prediction, in this case we do not need to perform
    negative sampling, since we already have both classes (the positively and the negatively connected node pairs).

    Examples
    --------
    Instantiating an SPEvaluator without a specific train/validation split (this split will be computed automatically if
    parameter tuning for any method is required):

    >>> from evalne.evaluation.evaluator import SPEvaluator
    >>> from evalne.evaluation.split import SPEvalSplit
    >>> from evalne.utils import preprocess as pp
    >>> # Load and preprocess a network
    >>> G = pp.load_graph('./evalne/tests/data/sig_network.edgelist')
    >>> G, _ = pp.prep_graph(G)
    >>> # Create the required train/test split
    >>> traintest_split = SPEvalSplit()
    >>> _ = traintest_split.compute_splits(G)
    >>> # Check that the train test parameters are indeed the correct ones
    >>> traintest_split.get_parameters()
    {'split_id': 0, 'nw_name': 'test', 'split_alg': 'spanning_tree', 'train_frac': 0.4980}
    >>> # Initialize the SPEvaluator
    >>> nee = SPEvaluator(traintest_split)

    Instantiating an SPEvaluator with a specific train/validation split (allows the user to specify any parameters
    for the train/validation split). Use 'fast' as the algorithm to split train and test edges and set train fraction
    to 0.8 for both train and validation splits:

    >>> from evalne.evaluation.evaluator import SPEvaluator
    >>> from evalne.evaluation.split import SPEvalSplit
    >>> from evalne.utils import preprocess as pp
    >>> # Load and preprocess a network
    >>> G = pp.load_graph('./evalne/tests/data/sig_network.edgelist')
    >>> G, _ = pp.prep_graph(G)
    >>> # Create the required train/test split
    >>> traintest_split = SPEvalSplit()
    >>> _ = traintest_split.compute_splits(G, train_frac=0.8, split_alg='fast')
    >>> # Check that the train test parameters are indeed the correct ones
    >>> traintest_split.get_parameters()
    {'split_id': 0, 'nw_name': 'test', 'split_alg': 'fast', 'train_frac': 0.8125}
    >>> # Create the train/validation split from the train data computed in the trintest_split
    >>> # The graph used to initialize this split must, thus, be the train graph from the traintest_split
    >>> trainvalid_split = SPEvalSplit()
    >>> _ = trainvalid_split.compute_splits(traintest_split.TG, train_frac=0.8, split_alg='fast')
    >>> # Initialize the SPEvaluator
    >>> nee = SPEvaluator(traintest_split, trainvalid_split)

    """

    def __init__(self, traintest_split, trainvalid_split=None, dim=128,
                 lp_model=LogisticRegressionCV(Cs=10, cv=5, penalty='l2', scoring='roc_auc', solver='lbfgs',
                                               max_iter=100)):
        # General evaluation parameters
        super(SPEvaluator, self).__init__(traintest_split, trainvalid_split, dim=dim, lp_model=lp_model)

    def _init_trainvalid(self):
        """
        Initializes the train/validation SPEvalSplit.
        """
        if self.trainvalid_split is None or len(self.trainvalid_split.test_edges) == 0:
            logging.warning('No test edges in trainvalid_split. Recomputing correct split...')
        self.trainvalid_split = split.SPEvalSplit()
        self.trainvalid_split.compute_splits(self.traintest_split.TG, nw_name=self.traintest_split.nw_name,
                                             train_frac=0.9, split_alg=self.traintest_split.split_alg,
                                             split_id=self.traintest_split.split_id, verbose=False)

    def evaluate_baseline(self, method, neighbourhood='in', timeout=None):
        """
        Evaluates the baseline method requested. Evaluation output is returned as a Results object. To evaluate the
        baselines on sign prediction we remove all negative edges from the train graph in traintest_split. For Katz
        neighbourhood=`in` and neighbourhood=`out` will return the same results corresponding to neighbourhood=`in`.
        Execution time is contained in the results object. If the train/test split object used to initialize the
        evaluator does not contain test edges, the results object will only contain train results.

        Parameters
        ----------
        method : string
            A string indicating the name of any baseline from evalne.methods to evaluate.
        neighbourhood : string, optional
            A string indicating the 'in' or 'out' neighbourhood to be used for directed graphs. Default is 'in'.
        timeout : float or None
            A float indicating the maximum amount of time (in seconds) the evaluation can run for. If None, the
            evaluation is allowed to continue until completion. Default is None.

        Returns
        -------
        results : Results
            The evaluation results as a Results object.

        Raises
        ------
        TimeoutExpired
            If the execution does not finish within the allocated time.
        TypeError
            If the Katz method call is incorrect.
        ValueError
            If the heuristic selected does not exist.

        See Also
        --------
        evalne.utils.util.run_function : The low level function used to run a baseline with given timeout.

        Examples
        --------
        Evaluating the common neighbours heuristic with default parameters. We assume an evaluator (nee) has already
        been instantiated (see class examples):

        >>> result = nee.evaluate_baseline(method='common_neighbours')
        >>> # Print the results
        >>> result.pretty_print()
        Method: common_neighbours
        Parameters:
        [('split_id', 0), ('dim', 128), ('neighbourhood', 'in'), ('split_alg', 'fast'),
        ('eval_time', 0.04459214210510254), ('nw_name', 'test'), ('train_frac', 0.8125)]
        Test scores:
        tn = 71
        [...]

        Evaluating katz with beta=0.05 and timeout 60 seconds. We assume an evaluator (nee) has already
        been instantiated (see class examples):

        >>> result = nee.evaluate_baseline(method='katz 0.05', timeout=60)
        >>> # Print the results
        >>> result.pretty_print()
        Method: katz 0.05
        Parameters:
        [('split_id', 0), ('dim', 128), ('neighbourhood', 'in'), ('split_alg', 'fast'),
        ('eval_time', 0.1246330738067627), ('nw_name', 'test'), ('train_frac', 0.8125)]
        Test scores:
        tn = 120
        [...]

        """
        # Measure execution time
        start = time.time()

        # Process the input graph to consider only edges with +1 connections as edges.
        neg_tr_e = self.traintest_split.train_edges[np.where(self.traintest_split.train_labels == -1)[0], :]
        TG = self.traintest_split.TG.copy()
        TG.remove_edges_from(neg_tr_e)

        if 'katz' in method:
            try:
                train_pred, test_pred = util.run_function(timeout, _eval_katz, *[method, self.traintest_split])
            except TypeError:
                raise TypeError('Call to katz method incorrect, try: `katz 0.01`')
            except util.TimeoutExpired as e:
                raise util.TimeoutExpired('{} after {} seconds'.format(e, time.time() - start))

        else:
            try:
                train_pred, test_pred = util.run_function(timeout, _eval_sim,
                                                          *[method, self.traintest_split, neighbourhood])
            except AttributeError:
                raise AttributeError('Method `{}` is not one of the available baselines!'.format(method))
            except util.TimeoutExpired as e:
                raise util.TimeoutExpired('{} after {} seconds'.format(e, time.time()-start))

        # Make predictions column vectors
        train_pred = np.array(train_pred)
        if test_pred is not None:
            test_pred = np.array(test_pred)

        # End of exec time measurement
        end = time.time() - start

        # Set some parameters for the results object
        params = {'neighbourhood': neighbourhood, 'eval_time': end}
        self.edge_embed_method = None

        if 'all_baselines' in method:
            # This method returns node-pair embeddings so we need to compute the predictions
            train_pred, test_pred = self.compute_pred(data_split=self.traintest_split, tr_edge_embeds=train_pred,
                                                      te_edge_embeds=test_pred)

        # Compute the scores
        if nx.is_directed(self.traintest_split.TG):
            results = self.compute_results(data_split=self.traintest_split, method_name=method + '-' + neighbourhood,
                                           train_pred=train_pred, test_pred=test_pred, params=params)
        else:
            results = self.compute_results(data_split=self.traintest_split, method_name=method,
                                           train_pred=train_pred, test_pred=test_pred, params=params)

        return results


class NCEvaluator(object):
    """
    Class designed to simplify the evaluation of embedding methods for node classification tasks.
    The input graphs is assumed to be the entire network. Parameter tuning is performed directly on this complete graph
    using a train/valid node split of specified size.

    Parameters
    ----------
    G : nx.Graph
        The full graph for which to run the evaluation.
    labels : ndarray
        A numpy array containing nodeIDs as first columns and labels as second column.
    nw_name : string
        A string indicating the name of the network. For result logging purposes.
    num_shuffles : int
        The number of times to repeat the evaluation with different train and test node sets.
    traintest_fracs : array-like
        The fraction of all nodes to use for training.
    trainvalid_frac : float
        The fraction of all training nodes to use for actual model training (the rest are used for validation).
    dim : int, optional
        Embedding dimensionality. Default is 128.
    nc_model : Sklearn binary classifier, optional
        The classifier to use for prediction. Default is logistic regression with 3 fold cross validation:
        `LogisticRegressionCV(Cs=10, cv=3, penalty='l2', multi_class='ovr')`

    Notes
    -----
    In node multi-label classification the aim is to predict the label associated with each graph node. We start the
    evaluation of this task by computing the embeddings for each node in the graph. Then, we train a classifier with
    with a subset of these embeddings (the training nodes) and their corresponding labels. Performance is evaluate on a
    holdout set. For robustness, the performance is generally averaged over multiple executions over different shuffles
    of the data (different train and test sets). The `num_shuffles` attribute controls the number of shuffles that will
    be generated.

    Examples
    --------
    Instantiating an NCEvaluator with default parameters:

    >>> from evalne.evaluation.evaluator import NCEvaluator
    >>> from evalne.utils import preprocess as pp
    >>> import numpy as np
    >>> # Load and preprocess a network
    >>> G = pp.load_graph('./evalne/tests/data/network.edgelist')
    >>> G, _ = pp.prep_graph(G)
    >>> # Generate some random node labels
    >>> labels = np.random.choice([1,2,3,4,5], size=len(G.nodes))
    >>> # Create pairs of (nodeID, label) and make them a column vector
    >>> nl_pairs = np.vstack((range(len(G.nodes)), labels)).T
    >>> # For NC we do not need to create a train test edge split, we can initialize the evaluator directly
    >>> nee = NCEvaluator(G, labels=nl_pairs, nw_name='test_network', num_shuffles=5, traintest_fracs=[0.8, 0.5],
    ...                  trainvalid_frac=0.5)

    """

    def __init__(self, G, labels, nw_name, num_shuffles, traintest_fracs, trainvalid_frac, dim=128,
                 nc_model=LogisticRegressionCV(Cs=10, cv=3, penalty='l2', multi_class='ovr')):
        # General evaluation parameters
        self.G = G
        self.labels = labels[np.argsort(labels[:, 0]), :]
        self.nw_name = nw_name
        self.traintest_fracs = traintest_fracs
        self.trainvalid_frac = trainvalid_frac
        self.shuffles = self._init_shuffles(num_shuffles)
        self.dim = dim
        self.nc_model = nc_model
        # Run some simple input checks
        self._check_labels()

    def _init_shuffles(self, num_shuffles):
        """
        Creates the required amount of indexing vectors and shuffles them randomly.

        Parameters
        ----------
        num_shuffles : int
            The number of randomly shuffled indexing vectors.
        """
        shuffles = list()
        for i in range(num_shuffles):
            sh = range(len(self.labels))
            np.random.shuffle(sh)
            shuffles.append(sh)
        return shuffles

    def _check_labels(self):
        """
        Checks that labels for each node in the input graph are provided and raises an error if not.

        Raises
        ------
        ValueError
            If there is a mismatch between the number of labels and nodes in the graph.
        """
        if len(set(self.labels[:, 0]) - set(self.G.nodes())) != 0:
            raise ValueError('Mismatch between node labels and node IDs of G')

    def _log_best(self, best_results, best_params, best_X, results, params, X, maximize, tr_te):
        """
        Keeps track of the best evaluation results and corresponding parameters. Updates the best results if needed.

        Parameters
        ----------
        best_results : list
            A list of Results objects containing the best Results.
        best_params : list of string
            A list of strings containing the parameter names and their associated values used to compute the best
            Results.
        best_X : list
            A list of the embedding dictionaries corresponding to the best results.
        results : list
            A list of new Results objects.
        params : string
            A string containing the parameter names and their associated values used to compute the new `results`.
        X : dict
            A dictionary where keys are nodes in the graph and values are the node embeddings. Is the dictionary used
            to obtain the new `results`.
        maximize : string
            The score to check while comparing results objects.
        tr_te : string
            A string indicating if the 'train' or 'test' results should be checked.

        Returns
        -------
        best_results : list
            A list of Results objects containing the best Results.
        best_params : list
            A list of strings containing the parameter names and their associated values used to compute the best
            Results.
        """
        for i in range(len(self.shuffles)):
            # Log the best results
            if best_results[i] is None:
                best_results[i] = results[i]
                best_params[i] = params
                best_X[i] = X
            else:
                if tr_te == 'train':
                    func1 = getattr(results[i].train_scores, str(maximize))
                    func2 = getattr(best_results[i].train_scores, str(maximize))
                else:
                    func1 = getattr(results[i].test_scores, str(maximize))
                    func2 = getattr(best_results[i].test_scores, str(maximize))
                if func1() > func2():
                    best_results[i] = results[i]
                    best_params[i] = params
                    best_X[i] = X

        return best_results, best_params, best_X

    def evaluate_cmd(self, method_name, command, input_delim, output_delim, tune_params=None,
                     maximize='f1_micro', write_weights=False, write_dir=False, timeout=None, verbose=True):
        """
        Evaluates an embedding method and tunes its parameters from the method's command line call string. Currently,
        this function can only evaluate node embedding methods.

        Parameters
        ----------
        method_name : string
            A string indicating the name of the method to be evaluated.
        command : string
            A string containing the call to the method as it would be written in the command line.
            For 'ne' methods placeholders (i.e. {}) need to be provided for the parameters: input network file,
            output file and embedding dimensionality, precisely IN THIS ORDER.
        input_delim : string
            The delimiter expected by the method as input (edgelist).
        output_delim : string
            The delimiter provided by the method in the output.
        tune_params : string, optional
            A string containing all the parameters to be tuned and their values. Default is None.
        maximize : string, optional
            The score to maximize while performing parameter tuning. Default is 'f1_micro'.
        write_weights : bool, optional
            If True the train graph passed to the embedding methods will be stored as weighted edgelist
            (e.g. triplets src, dst, weight) otherwise as normal edgelist. If the graph edges have no weight attribute
            and this parameter is set to True, a weight of 1 will be assigned to each edge. Default is False.
        write_dir : bool, optional
            This option is only relevant for undirected graphs. If False, the train graph will be stored with a single
            direction of the edges. If True, both directions of edges will be stored. Default is False.
        timeout : float or None
            A float indicating the maximum amount of time (in seconds) the evaluation can run for. If None, the
            evaluation is allowed to continue until completion. Default is None.
        verbose : bool, optional
            A parameter to control the amount of screen output. Default is True.

        Returns
        -------
        results : list of Results
            Returns the evaluation results as a list of Results objects (one for each traintest_frac requested and
            each shuffle). The length of the list returned will thus be `num_shuffles` * `len(traintest_fracs)`.

        Raises
        ------
        TimeoutExpired
            If the execution does not finish within the allocated time.
        IOError
            If the method call does not succeed.

        See Also
        --------
        evalne.utils.util.run : The low level function used to run a cmd call with given timeout.

        Examples
        --------
        Evaluating the OpenNE implementation of node2vec without parameter tuning. We assume the method is installed in
        a virtual environment and that an evaluator (nee) has already been instantiated (see class examples):

        >>> # Prepare the cmd command for running the method. If running on a python console full paths are required
        >>> cmd = '../OpenNE-master/venv/bin/python -m openne --method node2vec '\
        ...       '--graph-format edgelist --input {} --output {} --representation-size {}'
        >>> # Call the evaluation
        >>> result = nee.evaluate_cmd(method_name='Node2vec', command=cmd, input_delim=' ', output_delim=' ')
        Running command...
        [...]
        >>> # Check the results of the first data shuffle of traintest_frac=0.8
        >>> result[0].pretty_print()
        Method: Node2vec_0.8
        Parameters:
        [('dim', 128), ('nw_name', 'test_network'), ('eval_time', 33.22737193107605)]
        Test scores:
        f1_micro = 0.177304964539
        f1_macro = 0.0975922953451
        f1_weighted = 0.107965347267
        >>> # Check the results of the first data shuffle of traintest_frac=0.5
        >>> result[5].pretty_print()
        Method: Node2vec_0.5
        Parameters:
        [('dim', 128), ('nw_name', 'test_network'), ('eval_time', 33.22737193107605)]
        Test scores:
        f1_micro = 0.173295454545
        f1_macro = 0.0590799031477
        f1_weighted = 0.0511913933524

        Evaluating the metapath2vec c++ implementation without parameter tuning. We assume the method is installed and
        that an evaluator (nee) has already been instantiated (see class examples):

        >>> # Prepare the cmd command for running the method. If running on a python console full paths are required
        >>> cmd = '../../methods/metapath2vec/metapath2vec -min-count 1 -iter 20 '\
        ...       '-samples 100 -train {} -output {} -size {}'
        >>> # Call the evaluation
        >>> result = nee.evaluate_cmd(method_name='Metapath2vec', command=cmd, input_delim=' ', output_delim=' ')
        Running command...
        [...]
        >>> # Check the results of the second data shuffle of traintest_frac=0.8
        >>> result.pretty_print()
        Method: Metapath2vec_0.8
        Parameters:
        [('dim', 128), ('nw_name', 'test_network'), ('eval_time', 23.914228916168213)]
        Test scores:
        f1_micro = 0.205673758865
        f1_macro = 0.0711656441718
        f1_weighted = 0.0807553409041
        >>> # Check the results of the second data shuffle of traintest_frac=0.5
        >>> result.pretty_print()
        Method: Metapath2vec_0.5
        Parameters:
        [('dim', 128), ('nw_name', 'test_network'), ('eval_time', 23.914228916168213)]
        Test scores:
        f1_micro = 0.215909090909
        f1_macro = 0.0710280373832
        f1_weighted = 0.0766779949023

        """
        # Measure execution time
        start = time.time()
        if timeout is None:
            timeout = 31536000

        # Make sure the LRCV model maximizes what we want
        self.nc_model.scoring = maximize

        # Check the method type and raise an error if necessary
        # if method_type != 'ne':
        #    raise ValueError('Node classification not supported for method type `{}`.'.format(method_type))
        # if method_type in ['ee', 'e2e']:
        #     1) consider each node label as a new graph node with ID: len(TG.nodes())+np.unique(self.labels))
        #     2) add edges to the set of train edges between nodes and their `label_nodes`
        #     train_edges = np.vstack(self.traintest_split.train_edges,
        #                             np.array([labels[:,0], labels[:,1]+len(TG.nodes())]).T)
        #     3) Train with this data and predict only edges between nodes to `label_nodes`

        # Check if tuning parameters is needed
        if tune_params is not None:
            print('Tuning parameters for {} ...'.format(method_name))

            # Variable to store the best results and parameters for each ee_method
            num_sh = len(self.shuffles)
            best_results = [None] * num_sh
            best_params = [None] * num_sh
            best_X = [None] * num_sh

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
                        X = self._compute_emb_cmd(method_name, ext_command, input_delim, output_delim,
                                                  write_weights, write_dir, timeout-(time.time()-start), verbose)

                        # Compute results for all shuffles
                        results = self._evaluate_ne(X, method_name, [self.trainvalid_frac], self.shuffles,
                                                    train_only=True)

                        # Log the best results per shuffle
                        best_results, best_params, best_X = self._log_best(best_results, best_params, best_X,
                                                                           results, param_str, X, maximize, 'train')

                    except (ValueError, IOError, util.TimeoutExpired) as e:
                        logging.exception('Exception occurred while evaluating param `{}` for method `{}`.'
                                          .format(param_str, method_name))

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
                        X = self._compute_emb_cmd(method_name, ext_command, input_delim, output_delim,
                                                  write_weights, write_dir, timeout-(time.time()-start), verbose)

                        # Compute results for all shuffles
                        results = self._evaluate_ne(X, method_name, [self.trainvalid_frac], self.shuffles,
                                                    train_only=True)

                        # Log the best results per shuffle
                        best_results, best_params, best_X = self._log_best(best_results, best_params, best_X,
                                                                           results, param_str, X, maximize, 'train')

                    except (ValueError, IOError, util.TimeoutExpired) as e:
                        logging.exception('Exception occurred while evaluating param `{}` for method `{}`.'
                                          .format(param_str, method_name))

            results = list()
            # We found best params log that info and corresponding score
            for j in range(len(best_results)):
                if best_results[j] is None:
                    logging.error('NC shuffle {}: All param combinations for `{}` have failed! No results available.'
                                  .format(j, method_name))
                else:
                    # We report as validation scores the best results on the tr/valid split
                    bestscore = getattr(best_results[j].train_scores, str(maximize))
                    logging.info('NC shuffle {}: Validation score for `{}` is: {}, corresponding best params were: `{}`'
                                 .format(j, method_name, bestscore(), best_params[j]))
                    # Compute the best results for each shuffle using the best embeddings of full train/test split
                    results.extend(self._evaluate_ne(best_X[j], method_name, self.traintest_fracs, [self.shuffles[j]]))

        else:
            # No parameter tuning is needed
            # Compute the results on the full train split
            X = self._compute_emb_cmd(method_name, command, input_delim, output_delim,
                                      write_weights, write_dir, timeout-(time.time()-start), verbose)

            results = self.evaluate_ne(X, method_name)

        # End of exec time measurement
        end = time.time() - start
        for res in results:
            res.params.update({'eval_time': end})

        # Return the evaluation results
        return results

    def _compute_emb_cmd(self, method_name, command, input_delim, output_delim, write_weights,
                         write_dir, timeout, verbose):
        """
        Method that performs the cmd call and reads the embeddings. Stores the train graph as an edgelist to a temporal
        file and provides it as input to the method evaluated. Performs the command line call and reads the output.

        Returns
        -------
        X : dict
            A dictionary where keys are nodes in the graph and values are the node embeddings.
            The keys are of type string and the values of type array.
        """
        # Create temporal files with in/out data for method
        tmpedg = './edgelist.tmp'
        tmpemb = './emb.tmp'

        # Write the graph to a file
        pp.save_graph(self.G, output_path=tmpedg, delimiter=input_delim, write_stats=False,
                      write_weights=write_weights, write_dir=write_dir)

        # Add the input, output and embedding dimensionality to the command
        command = command.format(tmpedg, tmpemb, self.dim)

        print('Running command...')
        print(command)

        try:
            # Call the method
            util.run(command, timeout, verbose)

            # Some methods append a .txt filetype to the outfile if its the case, read the txt
            if os.path.isfile('./emb.tmp.txt'):
                tmpemb = './emb.tmp.txt'

            # Read embeddings from output file
            X = pp.read_node_embeddings(tmpemb, list(self.G.nodes()), self.dim, output_delim, method_name)

            # Evaluate the model
            return X

        except (IOError, OSError):
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

    def evaluate_ne(self, X, method_name, params=None):
        """
        Runs the NC evaluation pipeline. For each 'node_frac' trains a nc_model and uses it to compute predictions
        which are then returned as a results object. If data_split.test_edges is None, the Results object will only
        contain train Scores.

        Parameters
        ----------
        X : dict
            A dictionary where keys are nodes in the graph and values are the node embeddings.
            The keys are of type string and the values of type array.
        method_name : string
            A string indicating the name of the method to be evaluated.
        params : dict, optional
            A dictionary of parameters and values to be added to the results class. Default is None.

        Returns
        -------
        results : list
            Returns a list of Results objects one per each train/test fraction and each node shuffle.
        """
        return self._evaluate_ne(X, method_name, self.traintest_fracs, self.shuffles, params)

    def _evaluate_ne(self, X, method_name, node_fracs, shuffles, params=None, train_only=False):
        """
        The actual implementation of the node classification evaluation. For each 'node_frac' trains a nc_model and
        uses it to compute predictions which are then returned as a results object. If data_split.test_edges is None,
        the Results object will only contain train Scores.

        Returns
        -------
        results : list
            Returns a list of Results objects, one per each train/test fraction and each node shuffle.
        """

        # Initialize node frac if needed
        if node_fracs is None:
            node_fracs = [0.5]

        # Get the embeddings and sort them
        keys = map(int, X.keys())
        X = np.array(X.values())
        X = X[np.argsort(keys), :]

        results = list()
        for frac in node_fracs:
            for sh in shuffles:
                # Compute the train size
                train_size = int(len(sh) * frac)

                # Compute train data
                X_train = X[sh[:train_size], :]
                y_train = self.labels[sh[:train_size], 1]

                # Compute test data
                X_test = None
                y_test = None
                if not train_only:
                    X_test = X[sh[train_size:], :]
                    y_test = self.labels[sh[train_size:], 1]

                # Compute predictions
                train_pred, test_pred = self.compute_pred(X_train, y_train, X_test)

                # Compute results
                results.append(self.compute_results(method_name=method_name+'_'+str(frac), train_pred=train_pred,
                                                    train_labels=y_train, test_pred=test_pred, test_labels=y_test,
                                                    params=params))
        # Return the results
        return results

    def compute_pred(self, X_train, y_train, X_test=None):
        """
        Computes predictions from the given embeddings. Trains a NC model with the train node-pair embeddings and
        performs predictions for train and test embeddings. If te_edge_embeds is None test_pred will be None.

        Parameters
        ----------
        X_train : ndarray
            An array containing the train embeddings.
        y_train : ndarray
            An array containing the train labels.
        X_test : ndarray, optional
            An array containing the test embeddings.

        Returns
        -------
        train_pred : ndarray
            The label predictions for the train data.
        test_pred : ndarray
            The label predictions for the test data. Returns None if X_test is None.
        """
        # Fit the NC model
        self.nc_model.fit(X_train, y_train)

        # Predict
        train_pred = self.nc_model.predict(X_train)
        test_pred = None
        if X_test is not None:
            test_pred = self.nc_model.predict(X_test)

        # Return the predictions
        return train_pred, test_pred

    def compute_results(self, method_name, train_pred, train_labels, test_pred=None, test_labels=None, params=None):
        """
        Generates results from the given predictions and returns them. If test_pred is not provided, the Results
        object will only contain the train scores.

        Parameters
        ----------
        method_name : string
            A string indicating the name of the method for which the results will be created.
        train_pred : ndarray
            The predictions for the train data.
        test_pred : ndarray, optional
            The predictions for the test data. Default is None.
        params : dict, optional
            A dictionary of parameters and values to be added to the results class. Default is None.

        Returns
        -------
        results : Results
            The evaluation results.
        """
        # Get global parameters
        parameters = {'dim': self.dim, 'nw_name': self.nw_name}

        # Obtain the evaluation parameters
        if params is not None:
            parameters.update(params)

        results = score.NCResults(method=method_name, params=parameters,
                                  train_pred=train_pred, train_labels=train_labels,
                                  test_pred=test_pred, test_labels=test_labels)
        # Return the results
        return results
