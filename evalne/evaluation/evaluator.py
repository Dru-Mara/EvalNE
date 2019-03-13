#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# TODO: Use true labels and the preds to give statistics of where the method fails.

# TODO: Allow the user to select the parameters used for generating the validation data!
# TODO: After optimizing parameters, the execution on whole train graph is not efficient, several embds with same params

# TODO: EvalSetup checks (baselines exist, neigh param is correct, ee method exist, scores and report are correct)

from __future__ import division

import itertools
import os
import re
import subprocess
import warnings

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression

from evalne.evaluation import edge_embeddings
from evalne.evaluation import score
from evalne.evaluation import split
from evalne.methods import similarity as sim
from evalne.preprocessing import split_train_test as stt


class EvalSetup(object):
    r"""
    This class is a wrapper that parses the config file and provides the options as properties of the class.
    Also performs simple input checks

    Parameters
    ----------
    configpath : basestring
        The path of the configuration file.
    """

    def __init__(self, configpath):
        # Import config parser
        try:
            from ConfigParser import ConfigParser
        except ImportError:
            from configparser import ConfigParser

        # Read the configuration file
        config = ConfigParser()
        config.read(configpath)
        self._config = config

        # Check input paths
        self._check_inpaths()
        # Check methods
        self._check_methods('opne')
        self._check_methods('other')
        self._checkparams()
        self._check_edges()

    def _check_edges(self):
        if self.__getattribute__('train_frac') == 0:
            raise ValueError('The number of train edges can not be 0!')

        if (self.__getattribute__('num_fe_train') is not None and self.__getattribute__('num_fe_train') <= 0) or \
                (self.__getattribute__('num_fe_test') is not None and self.__getattribute__('num_fe_test') <= 0):
            raise ValueError('The number of train and test false edges has to be positive and grater than 0!')

    def _check_inpaths(self):
        l = len(self.__getattribute__('names'))
        for k in self._config.options('NETWORKS'):
            if len(self.__getattribute__(k)) != l:
                raise ValueError('Parameter `{}` in `NETWORKS` section does not have the required num. parameters ({})'
                                 .format(k, self.__getattribute__(k)))
        # Check if the input file exist
        for path in self.__getattribute__('inpaths'):
            if not os.path.exists(path):
                raise ValueError('Input network path {} does not exist'.format(path))

    def _check_methods(self, library):
        names = self.__getattribute__('names_' + library)
        methods = self.__getattribute__('methods_' + library)
        if names is not None and methods is not None and len(names) != len(methods):
            raise ValueError('Mismatch in the number of `NAMES` and `METHODS` to run in section `{} METHODS`'
                             .format(library))

    def _checkparams(self):
        directed = self.__getattribute__('directed')
        if self.__getattribute__('scores') == self.__getattribute__('maximize'):
            if sum(directed) == len(directed) or sum(directed) == 0:
                pass
            else:
                raise ValueError('Mix of directed and undirected networks found! Tabular output not supported.'
                                 'Please change the SCORES parameter to `all` or provide all dir./undir. networks')

    def getlist(self, section, option, dtype):
        r"""
        Returns option as a list of specified type, split by any kind of white space.

        Parameters
        ----------
        section : basestring
            The config file section name.
        option : basestring
            The config file option name.
        dtype : primitive type
            The type to which the output should be cast.

        Returns
        -------
        list : list
            A list of elements cast to the specified primitive type.
        """
        res = self._config.get(section, option).split()
        if len(res) == 0 or res[0] == '' or res[0] == 'None':
            return None
        else:
            return list(map(dtype, res))

    def getboollist(self, section, option):
        r"""
        Returns option as a list of booleans split by any kind of white space.
        Elements such as 'True', 'true', '1', 'yes', 'on' are considered True.
        Elements such as 'False', 'false', '0', 'no', 'off' are considered False.

        Parameters
        ----------
        section : basestring
            The config file section name.
        option : basestring
            The config file option name.

        Returns
        -------
        list : list
            A list of booleans.
        """
        res = self._config.get(section, option).split()
        if len(res) == 0 or res[0] == '' or res[0] == 'None':
            return None
        else:
            r = list()
            for elem in res:
                if elem in ['True', 'true', '1', 'yes', 'on']:
                    r.append(True)
                elif elem in ['False', 'false', '0', 'no', 'off']:
                    r.append(False)
            return r

    def getlinelist(self, section, option):
        r"""
        Returns option as a list of string, split specifically by a newline.

        Parameters
        ----------
        section : basestring
            The config file section name.
        option : basestring
            The config file option name.

        Returns
        -------
        list : list
            A list of strings.
        """
        res = self._config.get(section, option).split('\n')
        if len(res) == 0 or res[0] == '' or res[0] == 'None':
            return None
        else:
            return list(res)

    def getseplist(self, section, option):
        r"""
        Processes an options which contains a list of separators.
        Transforms \s, \t and \n to white space, tab and new line respectively

        Parameters
        ----------
        section : basestring
            The config file section name.
        option : basestring
            The config file option name.

        Returns
        -------
        list : list
            A list of strings.
        """
        separators = self.getlist(section, option, str)
        res = list()
        for sep in separators:
            s = sep.strip('\'')
            if s == '\\t':
                s = '\t'
            elif s == '\\s':
                s = ' '
            elif s == '\\n':
                s = '\n'
            res.append(s)
        return list(res)

    def gettuneparams(self, library):
        r"""
        Processes the tune parameters option. Generates a list of Nones the size of the number of methods.
        The list is filled in order with each line found in the TUNE_PARAMS option.

        Parameters
        ----------
        library : basestring
            This parameter indicates if the TUNE_PARAMETERS option processed if from OPNE METHODS of OTHER METHODS.

        Returns
        -------
        tune_params : list
            A list of string containing the parameters that need to be tuned.
        """
        methods = self.__getattribute__('methods_' + library)
        if library == 'opne':
            tune_params = self.getlinelist('OPENNE METHODS', 'tune_params_opne')
        elif library == 'other':
            tune_params = self.getlinelist('OTHER METHODS', 'tune_params_other')
        else:
            raise ValueError('Attribute name {}, does not exist'.format(library))
        if tune_params is None:
            tune_params = list()
        for i in range(len(methods) - len(tune_params)):
            tune_params.append(None)
        return tune_params

    @property
    def edge_embedding_methods(self):
        return self.getlist('GENERAL', 'edge_embedding_methods', str)

    @property
    def lp_model(self):
        return LogisticRegression(solver='liblinear')
        # return LogisticRegressionCV(Cs=10, cv=10, penalty='l2', scoring='roc_auc', solver='lbfgs', max_iter=200)

    @property
    def exp_repeats(self):
        return self._config.getint('GENERAL', 'exp_repeats')

    @property
    def embed_dim(self):
        return self._config.getint('GENERAL', 'embed_dim')

    @property
    def verbose(self):
        return self._config.getboolean('GENERAL', 'verbose')

    @property
    def seed(self):
        val = self._config.get('GENERAL', 'seed')
        if val == '' or val == 'None':
            return None
        else:
            return int(val)

    @property
    def names(self):
        return self.getlist('NETWORKS', 'names', str)

    @property
    def inpaths(self):
        return self.getlinelist('NETWORKS', 'inpaths')

    @property
    def outpaths(self):
        return self.getlinelist('NETWORKS', 'outpaths')

    @property
    def directed(self):
        return self.getboollist('NETWORKS', 'directed')

    @property
    def separators(self):
        return self.getseplist('NETWORKS', 'separators')

    @property
    def comments(self):
        return self.getseplist('NETWORKS', 'comments')

    @property
    def relabel(self):
        return self._config.getboolean('PREPROCESSING', 'relabel')

    @property
    def del_selfloops(self):
        return self._config.getboolean('PREPROCESSING', 'del_selfloops')

    @property
    def prep_nw_name(self):
        return self._config.get('PREPROCESSING', 'prep_nw_name')

    @property
    def write_stats(self):
        return self._config.getboolean('PREPROCESSING', 'write_stats')

    @property
    def delimiter(self):
        return self._config.get('PREPROCESSING', 'delimiter').strip('\'')

    @property
    def train_frac(self):
        return self._config.getfloat('TRAINTEST', 'train_frac')

    @property
    def fast_split(self):
        return self._config.getboolean('TRAINTEST', 'fast_split')

    @property
    def owa(self):
        return self._config.getboolean('TRAINTEST', 'owa')

    @property
    def num_fe_train(self):
        val = self._config.get('TRAINTEST', 'num_fe_train')
        if val == '' or val == 'None':
            return None
        else:
            return int(val)

    @property
    def num_fe_test(self):
        val = self._config.get('TRAINTEST', 'num_fe_test')
        if val == '' or val == 'None':
            return None
        else:
            return int(val)

    @property
    def traintest_path(self):
        return self._config.get('TRAINTEST', 'traintest_path')

    @property
    def lp_baselines(self):
        return self.getlinelist('BASELINES', 'lp_baselines')

    @property
    def neighbourhood(self):
        return self.getlist('BASELINES', 'neighbourhood', str)

    @property
    def names_opne(self):
        return self.getlist('OPENNE METHODS', 'names_opne', str)

    @property
    def methods_opne(self):
        return self.getlinelist('OPENNE METHODS', 'methods_opne')

    @property
    def tune_params_opne(self):
        return self.gettuneparams('opne')

    @property
    def names_other(self):
        return self.getlist('OTHER METHODS', 'names_other', str)

    @property
    def embtype_other(self):
        return self.getlist('OTHER METHODS', 'embtype_other', str)

    @property
    def write_weights_other(self):
        return self.getboollist('OTHER METHODS', 'write_weights_other')

    @property
    def write_dir_other(self):
        return self.getboollist('OTHER METHODS', 'write_dir_other')

    @property
    def methods_other(self):
        return self.getlinelist('OTHER METHODS', 'methods_other')

    @property
    def tune_params_other(self):
        return self.gettuneparams('other')

    @property
    def output_format_other(self):
        return self.getlinelist('OTHER METHODS', 'output_format_other')

    @property
    def input_delim_other(self):
        return self.getseplist('OTHER METHODS', 'input_delim_other')

    @property
    def output_delim_other(self):
        return self.getseplist('OTHER METHODS', 'output_delim_other')

    @property
    def maximize(self):
        return self._config.get('REPORT', 'maximize')

    @property
    def scores(self):
        return self._config.get('REPORT', 'scores')

    @property
    def curves(self):
        return self._config.get('REPORT', 'curves')

    @property
    def precatk_vals(self):
        return self.getlist('REPORT', 'precatk_vals', int)

    # @property
    # def report_train_scores(self):
    #    return self._config.getboolean('REPORT', 'report_train_scores')


class Evaluator(object):
    r"""
    Provides functions to evaluate network embedding methods on LP tasks.
    Can evaluate methods from their node embeddings, edge embeddings or edge predictions / similarity scores.

    Parameters
    ----------
    dim : int, optional
        The number of dimensions of the embedding. Default is 128.
    lp_model : object, optional
        The link prediction method to be used. Default is logistic regression.
    """

    def __init__(self, dim=128, lp_model=LogisticRegression(solver='liblinear')):
        # General evaluation parameters
        self.dim = dim
        self.edge_embed_method = None
        self.lp_model = lp_model
        # Train and validation data split objects
        self.traintest_split = split.EvalSplit()
        # Results
        self._results = list()

    def get_results(self):
        r"""
        Returns the scoresheet list
        """
        return self._results

    def reset_results(self):
        r"""
        Resets the scoresheet list
        """
        self._results = list()

    def evaluate_baseline(self, methods, neighbourhood='in'):
        """
        Evaluates the baseline methods. Results are stored as scoresheets.

        Parameters
        ----------
        methods : list
            A list of strings which are the names of any link prediction baseline from evalne.methods.similarity.
        neighbourhood : basestring, optional
            A string indicating the 'in' or 'out' neighbourhood to be used for directed graphs.
            Default is 'in'.
        """
        params = {'neighbourhood': neighbourhood}
        self.edge_embed_method = None

        for method in methods:
            try:
                func = getattr(sim, str(method))
                train_pred = func(self.traintest_split.TG, self.traintest_split.train_edges, neighbourhood)
                test_pred = func(self.traintest_split.TG, self.traintest_split.test_edges, neighbourhood)
            except AttributeError as e:
                raise AttributeError('Method is not one of the available baselines! {}'.format(e.message))

            # Make predictions column vectors
            train_pred = np.array(train_pred)
            test_pred = np.array(test_pred)

            # Add the scores
            if nx.is_directed(self.traintest_split.TG):
                results = self.compute_results(data_split=self.traintest_split,
                                               method_name=method + '-' + neighbourhood,
                                               train_pred=train_pred, test_pred=test_pred, params=params)
                self._results.append(results)
            else:
                results = self.compute_results(data_split=self.traintest_split,
                                               method_name=method,
                                               train_pred=train_pred, test_pred=test_pred, params=params)
                self._results.append(results)

    def evaluate_cmd(self, method_name, method_type, command, edge_embedding_methods, input_delim, output_delim,
                     tune_params=None, maximize='auroc', write_weights=False, write_dir=False, verbose=True):
        r"""
        Evaluates an embedding method and tunes its parameters from the method's command line call string. This
        function can evaluate node embedding, edge embedding or end to end embedding methods.
        If model parameter tuning is required, this method automatically generates train/validation splits
        with the same parameters as the train/test splits.

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
        """
        # If the method evaluated does not require edge embeddings set this parameter to ['none']
        if method_type != 'ne':
            edge_embedding_methods = ['none']
            self.edge_embed_method = None

        # Check if tuning parameters is needed
        if tune_params is not None:
            if verbose:
                print('Tuning parameters for {} ...'.format(method_name))

            # Variable to store the best results and parameters for each ee_method
            best_results = list()
            best_params = list()
            for i in range(len(edge_embedding_methods)):
                best_results.append(None)
                best_params.append('')

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
                aux = (params[i].strip()).split()   # Splits the parameter name from the parameter values to be tested
                param_names.append(aux.pop(0))
                params[i] = aux

            # Prepare validation data
            valid_split = split.EvalSplit()
            valid_split.compute_splits(self.traintest_split.TG, train_frac=0.9,
                                       fast_split=self.traintest_split.fast_split,
                                       owa=self.traintest_split.owa,
                                       num_fe_train=self.traintest_split.num_fe_train,
                                       num_fe_test=self.traintest_split.num_fe_test,
                                       split_id=self.traintest_split.split_id, verbose=verbose)

            # If there is only one parameter we treat it separately
            if len(param_names) == 1:
                for i in params[0]:
                    # Format the parameter combination
                    param_str = dash + param_names[0] + ' ' + i

                    # Create a command string with the new parameter
                    ext_command = command + param_str

                    # Call the corresponding evaluation method
                    if method_type == 'ne':
                        results = self._evaluate_ne_cmd(valid_split, method_name, ext_command, edge_embedding_methods,
                                                        input_delim, output_delim, write_weights, write_dir, verbose)
                    elif method_type == 'ee' or method_type == 'e2e':
                        results = self._evaluate_ee_e2e_cmd(valid_split, method_name, method_type, ext_command,
                                                            input_delim, output_delim, write_weights, write_dir,
                                                            verbose)
                    else:
                        raise ValueError('Method type {} unknown!'.format(method_type))
                    results = list(results)

                    # Log the best results
                    for j in range(len(results)):
                        if best_results[j] is None:
                            best_results[j] = results[j]
                            best_params[j] = param_str
                        else:
                            func1 = getattr(results[j].test_scores, str(maximize))
                            func2 = getattr(best_results[j].test_scores, str(maximize))
                            if func1() < func2():
                                best_results[j] = results[j]
                                best_params[j] = param_str
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

                    # Call the corresponding evaluation method
                    if method_type == 'ne':
                        results = self._evaluate_ne_cmd(valid_split, method_name, ext_command, edge_embedding_methods,
                                                        input_delim, output_delim, write_weights, write_dir, verbose)
                    elif method_type == 'ee' or method_type == 'e2e':
                        results = self._evaluate_ee_e2e_cmd(valid_split, method_name, method_type, ext_command,
                                                            input_delim, output_delim, write_weights, write_dir,
                                                            verbose)
                    else:
                        raise ValueError('Method type {} unknown!'.format(method_type))
                    results = list(results)

                    # Log the best results
                    for i in range(len(results)):
                        if best_results[i] is None:
                            best_results[i] = results[i]
                            best_params[i] = param_str
                        else:
                            func1 = getattr(results[i].test_scores, str(maximize))
                            func2 = getattr(best_results[i].test_scores, str(maximize))
                            if func1() < func2():
                                best_results[i] = results[i]
                                best_params[i] = param_str

            # We have found the best parameters, train the model again on the whole train data to get actual results
            results = list()
            for i in range(len(edge_embedding_methods)):
                ext_command = command + best_params[i]

                # Call the corresponding evaluation method
                if method_type == 'ne':
                    results.extend(self._evaluate_ne_cmd(self.traintest_split, method_name, ext_command,
                                                         [edge_embedding_methods[i]], input_delim, output_delim,
                                                         write_weights, write_dir, verbose))
                elif method_type == 'ee' or method_type == 'e2e':
                    results.extend(self._evaluate_ee_e2e_cmd(self.traintest_split, method_name, method_type,
                                                             ext_command, input_delim, output_delim, write_weights,
                                                             write_dir, verbose))
                else:
                    raise ValueError('Method type {} unknown!'.format(method_type))

            # zip(edge_embedding_methods, best_params)
            # data = collections.defaultdict(list)
            # for best in set(best_params):
            #     ext_command = command + best
            #     results.extend(self._evaluate_ne_cmd(self.traintest_split, method_name, ext_command,
            #                                          [edge_embedding_methods[i]], input_delim, emb_delim, verbose))

            # Store the evaluation results
            self._results.extend(results)
        else:
            # No parameter tuning is needed
            # Call the corresponding evaluation method
            if method_type == 'ne':
                results = self._evaluate_ne_cmd(self.traintest_split, method_name, command, edge_embedding_methods,
                                                input_delim, output_delim, write_weights, write_dir, verbose)
            elif method_type == 'ee' or method_type == 'e2e':
                results = self._evaluate_ee_e2e_cmd(self.traintest_split, method_name, method_type, command,
                                                    input_delim, output_delim, write_weights, write_dir, verbose)
            else:
                raise ValueError('Method type {} unknown!'.format(method_type))

            # Store the evaluation results
            self._results.extend(results)

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
        if verbose:
            print('Running command...')
            print(command)

        try:
            # Call the method
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
                warnings.warn('Found {} more lines in the file than expected. Will consider these lines part of the '
                              'header and ignore them... Expected num_lines {}, obtained lines {}.'
                              .format(emb_skiprows, len(data_split.TG.nodes), num_vectors), Warning)

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
                warnings.warn('Output provided by method {} contains {} columns, {} expected!'
                              '\nAssuming first column to be the nodeID...'
                              .format(method_name, X.shape[1], self.dim), Warning)
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

        except IOError as e:
            # log the error
            print('I/O error({}): {} while evaluating method {}'.format(e.errno, e.strerror, method_name))
            raise

        except ValueError:
            # log the error
            print('Error while evaluating method {}'.format(method_name))
            raise

        finally:
            pass
            # Delete the temporal files
            os.remove('./edgelist.tmp')
            if os.path.isfile('./emb.tmp'):
                os.remove('./emb.tmp')
            if os.path.isfile('./emb.tmp.txt'):
                os.remove('./emb.tmp.txt')

    def _evaluate_ee_e2e_cmd(self, data_split, method_name, method_type, command, input_delim, output_delim,
                             write_weights, write_dir, verbose):
        """
        The actual implementation of the edge embedding and end to end evaluation. Stores the train graph as an
        edgelist to a temporal file and provides it as input to the method evaluated together with the train and
        test edge sets. Performs the command line method call and reads the output edge embeddings/predictions.
        The method results are then computed according to the method type and returned.

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
        if placeholders == 4:
            # Stack train and test edges if the method only takes one input file
            ebunch = np.vstack((list(data_split.train_edges), list(data_split.test_edges)))
            stt.store_edgelists(tmp_tr_e, tmp_te_e, ebunch, [])
        else:
            data_split.store_edgelists(tmp_tr_e, tmp_te_e)

        if verbose:
            print('Running command...')
            print(command)

        try:
            # Call the method
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
                    warnings.warn('Found {} more lines in the output file than expected. Will consider these lines part'
                                  ' of the header and ignore them... Expected num_lines {}, obtained num lines {}.'
                                  .format(skiprows, num_tr_out,
                                          (len(data_split.train_edges) + len(data_split.test_edges))), Warning)

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
                        warnings.warn('Output provided by method {} is a matrix! '
                                      '\nPredictions assumed to be in the last column...'.format(method_name), Warning)
                        tr_out = out[0:len(data_split.train_edges), -1]
                        te_out = out[len(data_split.train_edges):, -1]

            else:
                # Autodetect and skip header if exists in output.
                num_tr_out = sum(1 for _ in open(tmp_tr_out))
                num_te_out = sum(1 for _ in open(tmp_te_out))
                tr_skiprows = num_tr_out - len(data_split.train_edges)
                te_skiprows = num_te_out - len(data_split.test_edges)

                if tr_skiprows < 0 or te_skiprows < 0:
                    raise ValueError('Method {} does not provide a unique prediction/embedding for every edge passed!'
                                     '\nExpected num. train predictions/embeddings {}'
                                     '\nObtained num. train predictions/embeddings {}'
                                     '\nExpected num. test predictions/embeddings {}'
                                     '\nObtained num. test predictions/embeddings {}'
                                     .format(method_name, len(data_split.train_edges), num_tr_out,
                                             len(data_split.test_edges), num_te_out))
                elif tr_skiprows > 0:
                    warnings.warn('Found {} more lines in the train file than expected. Will consider these lines part '
                                  'of the header and ignore them... Expected num_lines {}, obtained num lines {}.'
                                  .format(tr_skiprows, num_tr_out, len(data_split.train_edges)), Warning)
                elif te_skiprows > 0:
                    warnings.warn('Found {} more lines in the test file than expected. Will consider these lines part '
                                  'of the header and ignore them... Expected num_lines {}, obtained num lines {}.'
                                  .format(te_skiprows, num_te_out, len(data_split.test_edges)), Warning)

                # Read the embeddings/predictions
                tr_out = np.genfromtxt(tmp_tr_out, delimiter=output_delim, dtype=float, skip_header=tr_skiprows,
                                       autostrip=True)
                te_out = np.genfromtxt(tmp_te_out, delimiter=output_delim, dtype=float, skip_header=te_skiprows,
                                       autostrip=True)

                # Check if the method is ee or e2e
                if method_type == 'ee':
                    # By default assume edge embeddings given as matrix [X_0, X_1, ..., X_D] in same order as edgelist
                    if tr_out.ndim != 2 or te_out.ndim != 2 or \
                            tr_out.shape[1] != self.dim or te_out.shape[1] != self.dim:
                        raise ValueError('Incorrect edge embedding dimension for method {}!'
                                         '\nOutput expected train: ({},{}) \nOutput received train: {}'
                                         '\nOutput expected test: ({},{}) \nOutput received test: {}'
                                         .format(method_name, len(data_split.train_edges), self.dim, tr_out.shape,
                                                 len(data_split.test_edges), self.dim, te_out.shape))
                else:
                    # By default we assume the output is a vector of predictions in the same order as the edgelist.
                    # If output is a matrix, assume last column has predictions in the same order as the edgelist.
                    if tr_out.ndim == 2:
                        warnings.warn('Output provided by method {} is a matrix!'
                                      '\nPredictions assumed to be in the last column...'.format(method_name), Warning)
                        tr_out = tr_out[:, -1]
                        te_out = te_out[:, -1]

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

        except IOError as e:
            # log the error
            print('I/O error({}): {} while evaluating method {}'.format(e.errno, e.strerror, method_name))
            raise

        except ValueError:
            # log the error
            print('Error while evaluating method {}'.format(method_name))
            raise

        finally:
            # Delete the temporal files
            os.remove(tmpedg)
            os.remove(tmp_tr_e)
            os.remove(tmp_te_e)
            os.remove(tmp_tr_out)
            if os.path.isfile(tmp_te_out):
                os.remove(tmp_te_out)

    def evaluate_ne(self, data_split, X, method, edge_embed_method,
                    label_binarizer=LogisticRegression(solver='liblinear'), params=None):
        r"""
        Runs the complete pipeline, from node embeddings to edge embeddings and returns the prediction results.

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
            A Numpy matrix containing the test edge embeddings.
        """
        self.edge_embed_method = edge_embed_method

        try:
            func = getattr(edge_embeddings, str(edge_embed_method))
            tr_edge_embeds = func(X, data_split.train_edges)
            te_edge_embeds = func(X, data_split.test_edges)
        except AttributeError as e:
            raise AttributeError('Option is not one of the available edge embedding methods! {}'.format(e.message))

        return tr_edge_embeds, te_edge_embeds

    def compute_pred(self, data_split, tr_edge_embeds, te_edge_embeds):
        r"""
        Computes predictions from the given edge embeddings.
        Trains an LP model with the train edge embeddings and performs predictions for train and test edge embeddings.

        Parameters
        ----------
        data_split : EvalSplit
            An EvalSplit object that encapsulates the train/test or train/validation data.
        tr_edge_embeds : matrix
            A Numpy matrix containing the train edge embeddings.
        te_edge_embeds : matrix
            A Numpy matrix containing the test edge embeddings.

        Returns
        -------
        train_pred : array
            The link predictions for the train data.
        test_pred : array
            The link predictions for the test data.
        """
        # Train the LP model
        self.lp_model.fit(tr_edge_embeds, data_split.train_labels)

        # Predict
        train_pred = self.lp_model.predict_proba(tr_edge_embeds)[:, 1]
        test_pred = self.lp_model.predict_proba(te_edge_embeds)[:, 1]

        return train_pred, test_pred

    def compute_results(self, data_split, method_name, train_pred, test_pred,
                        label_binarizer=LogisticRegression(solver='liblinear'), params=None):
        r"""
        Generates results from the given predictions and returns them.

        Parameters
        ----------
        data_split : EvalSplit
            An EvalSplit object that encapsulates the train/test or train/validation data.
        method_name : basestring
            A string indicating the name of the method for which the results will be created.
        train_pred :
            The link predictions for the train data.
        test_pred :
            The link predictions for the test data.
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

        results = score.Results(method=method_name, params=parameters,
                                train_pred=train_pred, train_labels=data_split.train_labels,
                                test_pred=test_pred, test_labels=data_split.test_labels,
                                label_binarizer=label_binarizer)

        return results
