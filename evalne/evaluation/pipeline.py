#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# The manager module contains functions and classes for reading, parsing and using a configuration file to
# run a complete evaluation of network embedding methods.

from __future__ import division

import os

from evalne.utils import util
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV


class EvalSetup(object):
    r"""
    This class is a wrapper that parses the config file and provides the options as properties of the class.
    Also performs basic input checks.

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

        self._check_inpaths()
        self._check_methods('opne')
        self._check_methods('other')
        self._checkparams()
        self._check_edges()
        self._check_task()

    def _check_task(self):
        task = self.__getattribute__('task')
        if task not in ['lp', 'nc', 'nr']:
            raise ValueError('Incorrect value for `TASK`. Options are: `lp`, `nc` or `nr`.')
        if self.__getattribute__('task') == 'lp' and self.__getattribute__('lp_num_edge_splits') is None:
            raise ValueError('Parameter `LP_NUM_EDGE_SPLITS` needs to be defined.')
        if self.__getattribute__('task') == 'nr' and self.__getattribute__('nr_edge_samp_frac') is None:
            raise ValueError('Parameter `NR_EDGE_SAMP_FRAC` needs to be defined.')
        if self.__getattribute__('task') == 'nc':
            if self.__getattribute__('nc_num_node_splits') is None or self.__getattribute__('nc_node_fracs') is None:
                raise ValueError('Parameters `NC_NUM_NODE_SPLITS` and `NC_NODE_FRACS` need to be defined.')
            if all(x == 'ne' for x in self.__getattribute__('embtype_other')):
                pass
            else:
                raise ValueError('TASK = `nc` is currently only supported for node embedding methods.')

    def _check_edges(self):
        if self.__getattribute__('traintest_frac') is None or self.__getattribute__('trainvalid_frac') is None:
            raise ValueError('Train/test and train/validation fractions are required!')
        if self.__getattribute__('traintest_frac') == 0.0:
            raise ValueError('The train/test fraction, `TRAINTEST_FRAC`, can not be 0!')
        if self.__getattribute__('trainvalid_frac') == 0.0:
            raise ValueError('The train/valid fraction, `TRAINVALID_FRAC`, can not be 0!')
        if self.__getattribute__('fe_ratio') == 0.0:
            raise ValueError('The ratio of false edges, `FE_RATIO`, can not be 0!')

    def _check_inpaths(self):
        numnws = len(self.__getattribute__('names'))
        if self.__getattribute__('task') == 'nc' and self.__getattribute__('labelpaths') is None:
            raise ValueError('LABELPATHS for each network are required for node classification!')
        for k in self._config.options('NETWORKS'):
            if self.__getattribute__('task') == 'nc':
                if k != 'directed' and len(self.__getattribute__(k)) != numnws:
                    raise ValueError('Parameter `{}` in `NETWORKS` section does not have the required num. entries ({})'
                                     .format(k, self.__getattribute__(k)))
            else:
                if k != 'directed' and k != 'labelpaths' and len(self.__getattribute__(k)) != numnws:
                    raise ValueError('Parameter `{}` in `NETWORKS` section does not have the required num. entries ({})'
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
        # Check if the maximize attribute is a correct one
        if self.__getattribute__('task') == 'nc':
            if self.__getattribute__('maximize') not in ['f1_micro', 'f1_macro', 'f1_weighted']:
                raise ValueError('The selected metric in `REPORT.MAXIMIZE` does not exist!')
            # Check if the scores attribute is a correct one
            if self.__getattribute__('scores') not in ['', 'f1_micro', 'f1_macro', 'f1_weighted', 'all']:
                raise ValueError('The selected metric in `REPORT.SCORES` does not exist!')
        else:
            if self.__getattribute__('maximize') not in ['auroc', 'f_score', 'precision', 'recall',
                                                         'accuracy', 'fallout', 'miss']:
                raise ValueError('The selected metric in `REPORT.MAXIMIZE` does not exist!')
            # Check if the scores attribute is a correct one
            if self.__getattribute__('scores') not in ['', 'auroc', 'f_score', 'precision', 'recall', 'accuracy',
                                                       'fallout', 'miss', 'all']:
                raise ValueError('The selected metric in `REPORT.SCORES` does not exist!')
            # Check if the curves attribute is a correct one
            if self.__getattribute__('curves') not in ['', 'roc', 'pr', 'all']:
                raise ValueError('The value of `REPORT.CURVES` is incorrect!')

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
    def task(self):
        return self._config.get('GENERAL', 'task')

    @property
    def lp_num_edge_splits(self):
        return self._config.getint('GENERAL', 'lp_num_edge_splits')

    @property
    def nc_num_node_splits(self):
        return self._config.getint('GENERAL', 'nc_num_node_splits')

    @property
    def nc_node_fracs(self):
        return self.getlist('GENERAL', 'nc_node_fracs', float)

    @property
    def nr_edge_samp_frac(self):
        aux = self._config.getfloat('GENERAL', 'nr_edge_samp_frac')
        if aux > 1.0:
            return aux/100
        else:
            return aux

    @property
    def edge_embedding_methods(self):
        return self.getlist('GENERAL', 'edge_embedding_methods', str)

    @property
    def lp_model(self):
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
        return self._config.getint('GENERAL', 'embed_dim')

    @property
    def timeout(self):
        res = self._config.get('GENERAL', 'timeout')
        if res == '' or res == 'None' or res == 'NONE':
            return None
        else:
            return int(res)

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
    def directed(self):
        return self._config.getboolean('NETWORKS', 'directed')

    @property
    def separators(self):
        return self.getseplist('NETWORKS', 'separators')

    @property
    def comments(self):
        return self.getseplist('NETWORKS', 'comments')

    @property
    def labelpaths(self):
        return self.getlinelist('NETWORKS', 'labelpaths')

    @property
    def relabel(self):
        return self._config.getboolean('PREPROCESSING', 'relabel')

    @property
    def del_selfloops(self):
        return self._config.getboolean('PREPROCESSING', 'del_selfloops')

    @property
    def save_prep_nw(self):
        return self._config.getboolean('PREPROCESSING', 'save_prep_nw')

    @property
    def write_stats(self):
        return self._config.getboolean('PREPROCESSING', 'write_stats')

    @property
    def delimiter(self):
        return self._config.get('PREPROCESSING', 'delimiter').strip('\'')

    @property
    def traintest_frac(self):
        return self._config.getfloat('EDGESPLIT', 'traintest_frac')

    @property
    def trainvalid_frac(self):
        return self._config.getfloat('EDGESPLIT', 'trainvalid_frac')

    @property
    def split_alg(self):
        return self._config.get('EDGESPLIT', 'split_alg')

    @property
    def owa(self):
        return self._config.getboolean('EDGESPLIT', 'owa')

    @property
    def fe_ratio(self):
        return self._config.getfloat('EDGESPLIT', 'fe_ratio')

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
