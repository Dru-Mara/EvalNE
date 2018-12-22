#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# TODO: Use true labels and the preds to give statistics of where the method fails.

# TODO: dataset name should be passed to the method
# TODO: actually use naive/fast train test edge split
# TODO: Allow the user to select the parameters used for generating the validation data!!!
# TODO: After optimizing parameters, the execution on whole train graph is not efficient, several embds with same params

# TODO: check that the baselines read from the ini exist in the library
# TODO: check that the neighbourhood value is correct
# TODO: check that the ee methods in the ini exist in the library
# TODO:check if the scores and report parameters are correct

from __future__ import division

import itertools
import os
import re
import subprocess

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression

from evalne.evaluation import edge_embeddings
from evalne.evaluation import score
from evalne.evaluation import split
from evalne.methods import similarity as sim


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
        if len(names) != len(methods):
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

    def get_parameters(self):
        """
        Returns the evaluation parameters. Includes the train test split parameters.

        Returns
        -------
        parameters : dict
            The evaluation parameters as a dictionary of parameter : value
        """
        # Get global parameters
        params = {'dim': self.dim, 'edge_embed_method': self.edge_embed_method}

        # Get data related parameters
        params.update(self.traintest_split.get_parameters())

        return params

    def evaluate_baseline(self, methods=None, neighbourhood='in'):
        """
        Evaluates the baseline methods. Results are stored as scoresheets.

        Parameters
        ----------
        methods : list
            A list of strings which are the names of any link prediction baseline from evalne.methods.similarity.
        neighbourhood : basestring
            A string indicating the 'in' or 'out' neighbourhood to be used for directed graphs.
        """
        params = {'neighbourhood': neighbourhood}

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

    def evaluate_ne_cmd(self, method_name, command, edge_embedding_methods, input_delim, emb_delim, tune_params=None,
                        maximize='auroc', verbose=True):
        r"""
        Evaluates node embedding methods and tunes their parameters from the method's command line call string.
        This method generates automatically train/validation splits with the same parameters as the train/test splits.

        Parameters
        ----------
        method_name : basestring
            A string indicating the name of the method to be evaluated.
        command : basestring
            A string containing the call to the method as it would be written in the command line.
            For the values associated with the input file, output file and embedding dimensionality placeholders
            (i.e. {}) need to be provided precisely IN THIS ORDER.
        edge_embedding_methods : array-like
            A list of methods used to compute edge embeddings from the node embeddings output by the NE models.
            The accepted values are the function names in evalne.evaluation.edge_embeddings.
        input_delim : basestring
            The delimiter expected by the method as input (edgelist).
        emb_delim : basestring
            The delimiter provided by the method in the output (node embeddings)
        tune_params : basestring
            A string containing all the parameters to be tuned and their values.
        maximize : basestring
            The score to maximize while performing parameter tuning.
        verbose : bool
            A parameter to control the amount of screen output.
        """
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
                aux = (params[i].strip()).split()
                param_names.append(aux.pop(0))
                params[i] = aux

            # Prepare validation data
            valid_split = split.EvalSplit()
            valid_split.compute_splits(self.traintest_split.TG, train_frac=self.traintest_split.train_frac,
                                       fast_split=self.traintest_split.fast_split,
                                       owa=self.traintest_split.owa,
                                       num_fe_train=self.traintest_split.num_fe_train,
                                       num_fe_test=self.traintest_split.num_fe_test,
                                       seed=self.traintest_split.seed, verbose=verbose)

            # If there is only one parameter we treat it separately
            if len(param_names) == 1:
                for i in params[0]:
                    # Format the parameter combination
                    param_str = dash + param_names[0] + ' ' + i

                    # Create a command string with the new parameter
                    ext_command = command + param_str

                    results = self._evaluate_ne_cmd(valid_split, method_name, ext_command, edge_embedding_methods,
                                                    input_delim, emb_delim, verbose)

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

                    results = self._evaluate_ne_cmd(valid_split, method_name, ext_command, edge_embedding_methods,
                                                    input_delim, emb_delim, verbose)

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
                results.extend(self._evaluate_ne_cmd(self.traintest_split, method_name, ext_command,
                                                     [edge_embedding_methods[i]], input_delim, emb_delim, verbose))

            # zip(edge_embedding_methods, best_params)
            # data = collections.defaultdict(list)
            # for best in set(best_params):
            #     ext_command = command + best
            #     results.extend(self._evaluate_ne_cmd(self.traintest_split, method_name, ext_command,
            #                                          [edge_embedding_methods[i]], input_delim, emb_delim, verbose))

            self._results.extend(results)
        else:
            # No parameter tuning is needed
            results = self._evaluate_ne_cmd(self.traintest_split, method_name, command, edge_embedding_methods,
                                            input_delim, emb_delim, verbose)
            self._results.extend(results)

    def _evaluate_ne_cmd(self, data_split, method_name, command, edge_embedding_methods, input_delim, emb_delim,
                         verbose):
        """
        The actual implementation of the evaluation. Stores the train graph as an edgelist to a temporal file
        and provides it as input to the method evaluated. Performs teh command line call and reads the output.
        Node embeddings are transformed to edge embeddings and predistions are run.

        Returns
        -------
        results : list
            A list of results, one for each edge embedding method set.
        """
        # Create temporal files with in/out data for method
        tmpedg = './edgelist.tmp'
        tmpemb = './emb.tmp'

        # Write the train data to a file
        data_split.save_tr_graph(tmpedg, delimiter=input_delim)

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

            # Autodetect header of emb output
            # Read num lines in output
            num_vectors = sum(1 for _ in open(tmpemb))
            emb_skiprows = num_vectors - len(data_split.TG.nodes)

            if emb_skiprows < 0:
                print('ERROR: The method does not povide a unique embedding for every node in the graph')
                print('Expected node embeddings {}'.format(len(data_split.TG.nodes)))
                print('Obtained node embeddings {}'.format(num_vectors))

            # Read the embeddings
            X = np.genfromtxt(tmpemb, delimiter=emb_delim, dtype=float, skip_header=emb_skiprows, autostrip=True)

            if X.shape[1] == self.dim:
                # Assume embeddings given as matrix [X_0, X_1, ..., X_D] where row number is node id
                keys = map(str, range(len(X)))
                X = dict(zip(keys, X))
            elif X.shape[1] == self.dim + 1:
                # Assume first col is node id and rest are embedding features [id, X_0, X_1, ..., X_D]
                keys = map(str, np.array(X[:, 0], dtype=int))
                X = dict(zip(keys, X[:, 1:]))
            else:
                print('ERROR: Incorrect embedding dimensions!')
                exit(-1)

            # Evaluate the model
            results = list()
            for ee in edge_embedding_methods:
                results.append(self.evaluate_ne(data_split, X, method_name, ee))
            return results

        except IOError as e:
            print('I/O error({0}): {1}'.format(e.errno, e.strerror))

        except ValueError as e:
            print('ValueError {}'.format(e.message))
            print('ERROR: Could not parse embeddings!')

        except:
            print('Unexpected error')

        finally:
            # Delete the temporal files
            os.remove('./edgelist.tmp')
            os.remove('./emb.tmp')
            if os.path.isfile('./emb.tmp.txt'):
                os.remove('./emb.tmp.txt')

    def evaluate_ne(self, data_split, X, method, edge_embed_method, params=None):
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

        return self.compute_results(data_split, method, train_pred, test_pred, params)

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
        train_pred = self.lp_model.predict(tr_edge_embeds)
        test_pred = self.lp_model.predict(te_edge_embeds)

        return train_pred, test_pred

    def compute_results(self, data_split, method_name, train_pred, test_pred, params=None):
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
        params : dict
            A dictionary of parameters : values to be added to the results class.

        Returns
        -------
        results : Results
            Returns the evaluation results.
        """
        # Obtain the evaluation parameters
        parameters = self.get_parameters()
        if params is not None:
            parameters.update(params)

        results = score.Results(method=method_name, params=parameters,
                                train_pred=train_pred, train_labels=data_split.train_labels,
                                test_pred=test_pred, test_labels=data_split.test_labels)

        return results
