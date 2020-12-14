#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# This file contains methods and classes for reading and parsing configuration files. These files describe the entire
# evaluation pipeline in a set of variables called options organized in sections.

from __future__ import division

import os

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from evalne.utils import util


class EvalSetup(object):
    """
    Class that acts as a wrapper for the EvalNE .ini configuration files. Options (or variables) in the .ini files are
    exposed as class properties and basic input checks are performed.

    Parameters
    ----------
    configpath : string
        The path of the .ini configuration file.
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

        # Check input paremeters
        self._check_task()
        self._check_networks()
        self._check_edgesplit()
        self._check_methods('opne')
        self._check_methods('other')
        self._check_report()

    def _check_task(self):
        """
        Checks that all necessary options for a specific task are provided in the config file.

        Raises
        ------
        ValueError
            If any of the required options for the given task are not specified.
        """
        task = self.__getattribute__('task')
        if task not in ['lp', 'nc', 'nr', 'sp']:
            raise ValueError('Incorrect value for `TASK`! Accepted values are: `lp`, `nc`, `nr` or `sp`.')
        if self.__getattribute__('task') == 'lp' and self.__getattribute__('lp_num_edge_splits') is None:
            raise ValueError('For LP tasks `LP_NUM_EDGE_SPLITS` must be specified!')
        if self.__getattribute__('task') == 'sp' and self.__getattribute__('lp_num_edge_splits') is None:
            raise ValueError('For SP tasks `LP_NUM_EDGE_SPLITS` must be specified!')
        if self.__getattribute__('task') == 'nr' and self.__getattribute__('nr_edge_samp_frac') is None:
            raise ValueError('For NR tasks `NR_EDGE_SAMP_FRAC` must be specified!')
        if self.__getattribute__('task') == 'nc':
            if self.__getattribute__('nc_num_node_splits') is None or self.__getattribute__('nc_node_fracs') is None:
                raise ValueError('For NC tasks `NC_NUM_NODE_SPLITS` and `NC_NODE_FRACS` must be specified!')
            if all(x == 'ne' for x in self.__getattribute__('embtype_other')):
                pass
            else:
                raise ValueError('For NC tasks all methods must be of type node embedding (`EMBTYPE_OTHER = ne`)!')

    def _check_networks(self):
        """
        Checks config file options related to the networks (names, paths, labels, etc.).

        Raises
        ------
        ValueError
            If the input paths do not exist or if entries for any network are missing.
        """
        numnws = len(self.__getattribute__('names'))
        if self.__getattribute__('task') == 'nc' and self.__getattribute__('labelpaths') is None:
            raise ValueError('For NC tasks `LABELPATHS` must be specified for each network!')
        for k in self._config.options('NETWORKS'):
            if self.__getattribute__('task') == 'nc':
                if k != 'directed' and len(self.__getattribute__(k)) != numnws:
                    raise ValueError('Option `{}` in `NETWORKS` section does not have the required num. entries ({})!'
                                     .format(k, self.__getattribute__(k)))
            else:
                if k != 'directed' and k != 'labelpaths' and len(self.__getattribute__(k)) != numnws:
                    raise ValueError('Option `{}` in `NETWORKS` section does not have the required num. entries ({})!'
                                     .format(k, self.__getattribute__(k)))
        # Check if the input file exist
        for path in self.__getattribute__('inpaths'):
            if not os.path.exists(path):
                raise ValueError('Input network path `{}` does not exist!'.format(path))

    def _check_edgesplit(self):
        """
        Checks config file options related to the fraction of train and test edges and non-edges to generate.

        Raises
        ------
        ValueError
            If the entry values are out of their expected ranges or not specified.
        """
        if self.__getattribute__('traintest_frac') is None or self.__getattribute__('trainvalid_frac') is None:
            raise ValueError('Both train/test and train/validation fractions are required!')
        if self.__getattribute__('traintest_frac') == 0.0:
            raise ValueError('The train/test fraction (i.e. `TRAINTEST_FRAC`) can not be 0!')
        if self.__getattribute__('trainvalid_frac') == 0.0:
            raise ValueError('The train/valid fraction (i.e. `TRAINVALID_FRAC`) can not be 0!')
        if self.__getattribute__('fe_ratio') == 0.0:
            raise ValueError('The ratio of false edges (i.e. `FE_RATIO`) can not be 0!')

    def _check_methods(self, library):
        """
        Checks config file options related to the method calls and method names.

        Parameters
        ----------
        library : string
            A string indicating if the openne or other methods should be checked. Accepted values are: 'opne', 'other'.

        Raises
        ------
        ValueError
            In the number of methods calls and method names does not coincide.
        """
        names = self.__getattribute__('names_' + library)
        methods = self.__getattribute__('methods_' + library)
        if names is not None and methods is not None and len(names) != len(methods):
            raise ValueError('Mismatch in the number of `NAMES` and `METHODS` to run in section `{} METHODS`!'
                             .format(library.upper()))

    def _check_report(self):
        """
        Checks config file options related to results reporting. The performance metrics available depend on the task
        being evaluated.

        Raises
        ------
        ValueError
            If the wrong performance metric for a given task is required.
        """
        # Check if the maximize attribute is a correct one
        if self.__getattribute__('task') == 'nc':
            if self.__getattribute__('maximize') not in ['f1_micro', 'f1_macro', 'f1_weighted']:
                raise ValueError('The metric specified in `REPORT.MAXIMIZE` does not exist!')
            # Check if the scores attribute is a correct one
            if self.__getattribute__('scores') not in ['', 'f1_micro', 'f1_macro', 'f1_weighted', 'all']:
                raise ValueError('The metric specified in `REPORT.SCORES` does not exist!')
        else:
            if self.__getattribute__('maximize') not in ['auroc', 'f_score', 'precision', 'recall',
                                                         'accuracy', 'fallout', 'miss']:
                raise ValueError('The metric specified in `REPORT.MAXIMIZE` does not exist!')
            # Check if the scores attribute is a correct one
            if self.__getattribute__('scores') not in ['', 'auroc', 'f_score', 'precision', 'recall', 'accuracy',
                                                       'fallout', 'miss', 'all']:
                raise ValueError('The metric specified in `REPORT.SCORES` does not exist!')
            # Check if the curves attribute is a correct one
            if self.__getattribute__('curves') not in ['', 'roc', 'pr', 'all']:
                raise ValueError('The value of `REPORT.CURVES` is incorrect!')

    def getlist(self, section, option, dtype):
        """
        Reads a string option and returns it as a list of elements of the specified type. The input string is split
        by any kind of white space separator.

        Parameters
        ----------
        section : string
            A config file section name.
        option : string
            A config file option name.
        dtype : primitive type
            The desired type of the elements in the output list.

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
        """
        Reads a string option and returns it as a list of booleans. The input string is split by any kind of white
        space separator. Elements such as 'True', 'true', '1', 'yes', 'on' are mapped to True. Elements such as
        'False', 'false', '0', 'no', 'off' are mapped to False.

        Parameters
        ----------
        section : string
            A config file section name.
        option : string
            A config file option name.

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
        """
        Reads a string option and returns it as a list of strings split by new lines only.

        Parameters
        ----------
        section : string
            A config file section name.
        option : string
            A config file option name.

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
        """
        Reads a string option containing several separators ('\\s', '\\t' and '\\n' ) and returns it as a list of
        proper string separators (white space, tab or new line).

        Parameters
        ----------
        section : string
            A config file section name.
        option : string
            A config file option name.

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
        """
        Reads a 'TUNE_PARAMS' option that contain parameters and their associated values (e.g. 'TUNE_PARAMS').
        The method returns the option as a list of strings split by new lines. The list if filled with None if needed
        so the length is the same as the number of methods being evaluated.

        Parameters
        ----------
        library : string
            A string indicating if the openne or other 'TUNE_PARAMS' should be checked. Accepted values are: 'opne',
            'other'.

        Returns
        -------
        tune_params : list
            A list of string or None containing parameters and their values.
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
    def task(self):
        """Returns a string indicating the task to evaluate i.e. link prediction (LP), sign prediction (SP), network
        reconstruction (NR) or node classification (NC). Possible values: {'lp', 'sp', 'nr', 'nc'}"""
        return self._config.get('GENERAL', 'task')

    @property
    def lp_num_edge_splits(self):
        """Returns an int indicating the number of repetitions for experiment with different train/test edge splits.
        Required if task is 'lp', 'sp' or 'nr'. For 'nc' this value must be 1."""
        return self._config.getint('GENERAL', 'lp_num_edge_splits')

    @property
    def nc_num_node_splits(self):
        """Returns an int indicating the number of repetitions for NC experiments with different train/test node splits.
        Required if task is 'nc'."""
        return self._config.getint('GENERAL', 'nc_num_node_splits')

    @property
    def nc_node_fracs(self):
        """Returns a list of float indicating the fractions of train labels to use when evaluating NC. Required if task
        is 'nc'."""
        return self.getlist('GENERAL', 'nc_node_fracs', float)

    @property
    def nr_edge_samp_frac(self):
        """Returns a float indicating the fraction of all possible node pairs to sample and compute precision@k for
        when evaluating NR. Required if task is 'nr'."""
        aux = self._config.getfloat('GENERAL', 'nr_edge_samp_frac')
        if aux > 1.0:
            return aux/100
        else:
            return aux

    @property
    def edge_embedding_methods(self):
        """Returns a list of strings indicating the node-pair operators to use. Possible values: {'average', 'hadamard',
        'weighted_l1', 'weighted_l2'}"""
        return self.getlist('GENERAL', 'edge_embedding_methods', str)

    @property
    def lp_model(self):
        """Returns an sklearn binary classifier used to predict links from node-pair embeddings."""
        model = self._config.get('GENERAL', 'lp_model')
        if model == 'LogisticRegression':
            return LogisticRegression(solver='liblinear')
        elif model == 'LogisticRegressionCV':
            return LogisticRegressionCV(Cs=10, cv=5, penalty='l2', scoring='roc_auc', solver='lbfgs', max_iter=100)
        elif model == 'DecisionTreeClassifier':
            return DecisionTreeClassifier()
        elif model == 'SVM':
            parameters = {'C': [0.1, 1, 10, 100, 1000]}
            return GridSearchCV(LinearSVC(), parameters, cv=5)
        else:
            return util.auto_import(model)

    @property
    def embed_dim(self):
        """Returns an int indicating the dimensions of the embedding."""
        return self._config.getint('GENERAL', 'embed_dim')

    @property
    def timeout(self):
        """Returns a float indicating the maximum execution time in seconds (or None) for each method including
        hyperparameter tuning."""
        res = self._config.get('GENERAL', 'timeout')
        if res == '' or res == 'None' or res == 'NONE':
            return None
        else:
            return float(res)

    @property
    def verbose(self):
        """Returns a bool indicating the verbosity level of the execution."""
        return self._config.getboolean('GENERAL', 'verbose')

    @property
    def seed(self):
        """Returns and int or None indicating the random seed to use in the experiments. Possible values: {'', 'None',
        any_int}"""
        val = self._config.get('GENERAL', 'seed')
        if val == '' or val == 'None':
            return None
        else:
            return int(val)

    @property
    def names(self):
        """Returns a list of strings indicating the names of the networks to be evaluated."""
        return self.getlist('NETWORKS', 'names', str)

    @property
    def inpaths(self):
        """Returns a list of strings indicating the paths to files containing the networks. A check is performed to
        ensure the paths exist."""
        return self.getlinelist('NETWORKS', 'inpaths')

    @property
    def directed(self):
        """Returns a bool indicating if all the networks are directed or not."""
        return self._config.getboolean('NETWORKS', 'directed')

    @property
    def separators(self):
        """Returns a list of strings indicating the separators used in the network files."""
        return self.getseplist('NETWORKS', 'separators')

    @property
    def comments(self):
        """Returns a list of strings, the characters denoting comments in the network files."""
        return self.getseplist('NETWORKS', 'comments')

    @property
    def labelpaths(self):
        """Returns a list of string indicating the paths where the node label files can be found. Required if task is
        'nc'"""
        return self.getlinelist('NETWORKS', 'labelpaths')

    @property
    def relabel(self):
        """Returns a bool, relabel or not the network nodes to 0...N (required for methods such as PRUNE)"""
        return self._config.getboolean('PREPROCESSING', 'relabel')

    @property
    def del_selfloops(self):
        """Returns a bool, delete or not self loops in the network."""
        return self._config.getboolean('PREPROCESSING', 'del_selfloops')

    @property
    def save_prep_nw(self):
        """Returns a bool if the preprocessed graph should be stored or not."""
        return self._config.getboolean('PREPROCESSING', 'save_prep_nw')

    @property
    def write_stats(self):
        """Returns a bool, write or not common graph statistics as header in the preprocessed network file."""
        return self._config.getboolean('PREPROCESSING', 'write_stats')

    @property
    def delimiter(self):
        """Returns a string indicating the delimiter to be used when writing the preprocessed graphs to a files."""
        return self._config.get('PREPROCESSING', 'delimiter').strip('\'')

    @property
    def traintest_frac(self):
        """Returns a float indicating the fraction of total edges to use for training and validation. The rest should
        be used for testing."""
        return self._config.getfloat('EDGESPLIT', 'traintest_frac')

    @property
    def trainvalid_frac(self):
        """Returns a float indicating the fraction of train-validation edges to use for training. The rest should be
        used for validation."""
        return self._config.getfloat('EDGESPLIT', 'trainvalid_frac')

    @property
    def split_alg(self):
        """Returns a string indicating the algorithm to use for splitting edges in train/test, train/validation sets.
        Possible values: {'spanning_tree', 'random', 'naive', 'fast', 'timestamp'}."""
        return self._config.get('EDGESPLIT', 'split_alg')

    @property
    def owa(self):
        """Returns a bool, indicating if the open world (True) or the closed world assumption (False) for non-edges
        should be used."""
        return self._config.getboolean('EDGESPLIT', 'owa')

    @property
    def fe_ratio(self):
        """Returns a float indicating the ratio of non-edges to edges for tr & te. The num_fe = fe_ratio * num_edges."""
        return self._config.getfloat('EDGESPLIT', 'fe_ratio')

    @property
    def lp_baselines(self):
        """Returns a list of strings indicating the link prediction heuristics to evaluate. Possible values: {'',
        'random_prediction', 'common_neighbours', 'jaccard_coefficient', 'adamic_adar_index', 'preferential_attachment',
        'resource_allocation_index', 'cosine_similarity', 'lhn_index', 'topological_overlap', 'katz', 'all_baselines'}
        """
        return self.getlinelist('BASELINES', 'lp_baselines')

    @property
    def neighbourhood(self):
        """Returns a list of string indicating, for directed graphs, if the in or the out neighbourhood should be used.
        Possible values: {'', 'in', 'out'}"""
        return self.getlist('BASELINES', 'neighbourhood', str)

    @property
    def names_opne(self):
        """Returns a list of strings indicating the names of methods from OpenNE to be evaluated. In the same order as
        METHODS_OPNE."""
        return self.getlist('OPENNE METHODS', 'names_opne', str)

    @property
    def methods_opne(self):
        """Returns a list of strings indicating the command line calls to perform in order to evaluate each method."""
        return self.getlinelist('OPENNE METHODS', 'methods_opne')

    @property
    def tune_params_opne(self):
        """Returns a list of strings indicating the parameters of methods from OpenNE to be tuned by the library and
        values to try."""
        return self.gettuneparams('opne')

    @property
    def names_other(self):
        """Returns a list of strings indicating the names of any other methods not from OpenNE to be evaluated. In the
        same order as METHODS_OTHER."""
        return self.getlist('OTHER METHODS', 'names_other', str)

    @property
    def embtype_other(self):
        """Returns a list of strings indicating the method's output type: node embeddings (ne), edge embeddings (ee) or
         node similarities (e2e). Possible values: {'ne', 'ee', 'e2e'}."""
        return self.getlist('OTHER METHODS', 'embtype_other', str)

    @property
    def write_weights_other(self):
        """Returns a list of bool indicating if training graphs should be given as input to methods weighted (True) or
        unweighted (False)."""
        return self.getboollist('OTHER METHODS', 'write_weights_other')

    @property
    def write_dir_other(self):
        """Returns a list of bool indicating if training graphs should be given as input to methods with both edge dir.
        (True) or one (False)."""
        return self.getboollist('OTHER METHODS', 'write_dir_other')

    @property
    def methods_other(self):
        """Returns a list of strings indicating the command line calls to perform in order to evaluate each method."""
        return self.getlinelist('OTHER METHODS', 'methods_other')

    @property
    def tune_params_other(self):
        """Returns a list of strings indicating the parameters to be tuned by the library."""
        return self.gettuneparams('other')

    @property
    def output_format_other(self):
        """Returns """
        return self.getlinelist('OTHER METHODS', 'output_format_other')

    @property
    def input_delim_other(self):
        """Returns a list of strings indicating the input delimiters expected the by each methods."""
        return self.getseplist('OTHER METHODS', 'input_delim_other')

    @property
    def output_delim_other(self):
        """Returns a list of strings indicating the delimiter used by each method in the output file (when writing node
        embeddings, edge embeddings or predictions)."""
        return self.getseplist('OTHER METHODS', 'output_delim_other')

    @property
    def maximize(self):
        """Returns a string indicating the score to maximize when performing model validation. Possible values for LP,
        SP and NR: {'auroc', 'f_score', 'precision', 'recall', 'accuracy', 'fallout', 'miss'}. Possible values for NC:
        {'f1_micro', 'f1_macro', 'f1_weighted'}"""
        return self._config.get('REPORT', 'maximize')

    @property
    def scores(self):
        """Returns a string indicating the score to be reported in the output file. Possible values: {'',
        '%(maximize)s', 'all'}"""
        return self._config.get('REPORT', 'scores')

    @property
    def curves(self):
        """Returns a string indicating the curves to provide as output."""
        return self._config.get('REPORT', 'curves')

    @property
    def precatk_vals(self):
        """Returns a list of int indicating the values of k for which to provide the precision at k."""
        return self.getlist('REPORT', 'precatk_vals', int)
