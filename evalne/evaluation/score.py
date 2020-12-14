#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# This file contains methods and classes that simplify the management and storage of evaluation results, both for
# individual methods as well as complete evaluations.

from __future__ import division

import os
import warnings
import numpy as np
import pandas as pd
try:
    import cPickle as pickle
except ImportError:
    import pickle

from collections import OrderedDict
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score

from evalne.utils import viz_utils as viz


class Scoresheet:
    """
    Class that simplifies the logging and management of evaluation results and execution times. Functions for logging,
    plotting and writing the results to files are provided. The Scoresheet does not log the complete train or test
    model predictions.

    Parameters
    ----------
    tr_te : string, optional
        A string indicating if the 'train' or 'test' results should be stored. Default is 'test'.
    precatk_vals : list of int or None, optional
        The values for which the precision at k should be computed. Default is None.
    """

    def __init__(self, tr_te='test', precatk_vals=None):
        self._tr_te = tr_te
        self._precatk_vals = precatk_vals
        self._scoresheet = OrderedDict()
        self._all_methods = OrderedDict()

    def log_results(self, results):
        """
        Logs in the Scoresheet all the performance metrics (and execution time) extracted from the input Results object
        or list of Results objects. Multiple Results for the same method on the same network can be provided and will
        all be stored (these are assumed to correspond to different repetitions of the experiment).

        Parameters
        ----------
        results : Results or list of Results
            The Results object or objects to be logged in the Scoresheet.

        Examples
        --------
        Evaluate the common neighbours baseline and log the train and test results:

        >>> tr_scores = Scoresheet(tr_te='train')
        >>> te_scores = Scoresheet(tr_te='test')
        >>> result = nee.evaluate_baseline(method='common_neighbours')
        >>> tr_scores.log_results(result)
        >>> te_scores.log_results(result)

        """
        if isinstance(results, Results):
            self._log_result(results)
        else:
            for res in results:
                self._log_result(res)

    def _log_result(self, result):
        """
        Logs in the Scoresheet all the performance metrics (and execution time) extracted from the input Results object.

        Parameters
        ----------
        result : Results
            The Results object to be logged in the Scoresheet.
        """
        # Get the dictionary keys
        k1 = result.params['nw_name']       # First key is network name
        k2 = result.method                  # Second key is method name
        self._all_methods[k2] = 0

        # Store the results
        if k1 in self._scoresheet:
            # Dataset exists in the dictionary, so we extend it
            if k2 in self._scoresheet[k1]:
                # Method exists in the dictionary, so we extend its metrics with vals of new exp repeat
                metrics, vals = result.get_all(self._tr_te, self._precatk_vals)
                for i in range(len(metrics)):
                    self._scoresheet[k1][k2][metrics[i]].append(np.around(vals[i], 4))
                self._scoresheet[k1][k2]['eval_time'].append(result.params['eval_time'])
                self._scoresheet[k1][k2]['edge_embed_method'].append(result.params.get('edge_embed_method', 'None'))

            else:
                # Method is not yet in the dict, so we add method and metrics
                metrics, vals = result.get_all(self._tr_te, self._precatk_vals)
                self._scoresheet[k1][k2] = OrderedDict(zip(metrics, map(lambda x: [np.around(x, 4)], vals)))
                self._scoresheet[k1][k2]['eval_time'] = [result.params['eval_time']]
                self._scoresheet[k1][k2]['edge_embed_method'] = [result.params.get('edge_embed_method', 'None')]

        else:
            # Dataset is not yet in the dict, so we add dataset, method and metrics
            metrics, vals = result.get_all(self._tr_te, self._precatk_vals)
            aux = OrderedDict(zip(metrics, map(lambda x: [np.around(x, 4)], vals)))
            self._scoresheet[k1] = OrderedDict({k2: aux})
            self._scoresheet[k1][k2]['eval_time'] = [result.params['eval_time']]
            self._scoresheet[k1][k2]['edge_embed_method'] = [result.params.get('edge_embed_method', 'None')]

    def get_pandas_df(self, metric='auroc', repeat=None):
        """
        Returns a view of the Scoresheet as a pandas DataFrame for the specified metric. The columns of the DataFrame
        represent different networks and the rows different methods. If multiple Results for the same network/method
        combination were logged (multiple repetitions of the experiment), one can select any of these repeats or get
        the average over all.

        Parameters
        ----------
        metric : string, optional
            Can be one of 'tn', 'fp', 'fn', 'tp', 'auroc', 'precision', 'recall', 'fallout', 'miss', 'accuracy',
            'f_score', 'eval_time' or 'edge_embed_method'. Default is 'auroc'.
        repeat : int, optional
            An int indicating the experiment repeat for which the results should be returned. If not indicated, the
            average over all repeats will be computed and returned. Default is None (computes average over repeats).

        Returns
        -------
        df : DataFrame
            A pandas DataFrame view of the Scoresheet for the specified metric.

        Raises
        ------
        ValueError
            If the requested metric does not exist.
            If the Scoresheet is empty so a DataFrame can not be generated.

        Notes
        -----
        For uncountable 'metrics' such as the node pair embedding operator (i.e 'edge_embed_method'), avg returns the
        most frequent item in the vector.

        Examples
        --------
        Read a scoresheet and get the auroc scores as a pandas DataFrame

        >>> scores = pickle.load(open('lp_eval_1207_1638/eval.pkl', 'rb'))
        >>> df = scores.get_pandas_df()
        >>> df
                            Network_1 Network_2
        katz                   0.8203    0.8288
        common_neighbours      0.3787    0.3841
        jaccard_coefficient    0.3787    0.3841

        Read a scoresheet and get the f scores of the first repetition of the experiment

        >>> scores = pickle.load(open('lp_eval_1207_1638/eval.pkl', 'rb'))
        >>> df = scores.get_pandas_df('f_score', repeat=0)
        >>> df
                            Network_1 Network_2
        katz                        0         0
        common_neighbours      0.7272    0.7276
        jaccard_coefficient    0.7265    0.7268

        """
        if len(self._scoresheet) != 0:
            nw = next(iter(self._scoresheet))
            if metric not in iter(self._scoresheet[nw][next(iter(self._scoresheet[nw]))].keys()):
                raise ValueError('Requested metric `{}` does not exist!'.format(metric))
        else:
            raise ValueError('Scoresheet is empty, can not generate pandas df! Try logging some results first.')

        cols = self._scoresheet.keys()
        rows = list(self._all_methods)
        df = pd.DataFrame(index=rows, columns=cols)
        for k1 in cols:
            for k2 in rows:
                d = self._scoresheet[k1].get(k2)
                if d is not None:
                    if repeat is None:
                        if metric == 'edge_embed_method':
                            count = Counter(d.get(metric))
                            df[k1][k2] = count.most_common(1)[0][0]
                        else:
                            df[k1][k2] = np.around(np.mean(np.array(d.get(metric))), 4)
                    else:
                        arr = d.get(metric)
                        if len(arr) >= repeat+1:
                            df[k1][k2] = d.get(metric)[repeat]
                        else:
                            df[k1][k2] = None
        return df

    def get_latex(self, metric='auroc'):
        """
        Returns a view of the Scoresheet as a Latex table for the specified metric. The columns of the table
        represent different networks and the rows different methods. If multiple Results for the same network/method
        combination were logged (multiple repetitions of the experiment), the average is returned.

        Parameters
        ----------
        metric : string, optional
            Can be one of 'tn', 'fp', 'fn', 'tp', 'auroc', 'precision', 'recall', 'fallout', 'miss', 'accuracy',
            'f_score', 'eval_time' or 'edge_embed_method'. Default is 'auroc'.

        Returns
        -------
        latex_table : string
            A latex table as a string.
        """
        df = self.get_pandas_df(metric)
        return df.to_latex()

    def print_tabular(self, metric='auroc'):
        """
        Prints a tabular view of the Scoresheet for the specified metric. The columns of the table represent different
        networks and the rows different methods. If multiple Results for the same network/method combination were logged
        (multiple repetitions of the experiment), the average is showed.

        Parameters
        ----------
        metric : string, optional
            Can be one of 'tn', 'fp', 'fn', 'tp', 'auroc', 'precision', 'recall', 'fallout', 'miss', 'accuracy',
            'f_score', 'eval_time' or 'edge_embed_method'. Default is 'auroc'.

        Examples
        --------
        Read a scoresheet and get the average execution times over all experiment repeats as tabular output:

        >>> scores = pickle.load(open('lp_eval_1207_1638/eval.pkl', 'rb'))
        >>> scores.print_tabular('eval_time')
                            Network_1 Network_2
        katz                   0.0350    0.0355
        common_neighbours      0.0674    0.0676
        jaccard_coefficient    0.6185    0.6693

        """
        print(self.get_pandas_df(metric))

    def write_tabular(self, filename, metric='auroc'):
        """
        Writes a tabular view of the Scoresheet for the specified metric to a file. The columns of the table represent
        different networks and the rows different methods. If multiple Results for the same network/method combination
        were logged (multiple repetitions of the experiment), the average is used.

        Parameters
        ----------
        filename : string
            A file where to store the results.
        metric : string, optional
            Can be one of 'tn', 'fp', 'fn', 'tp', 'auroc', 'precision', 'recall', 'fallout', 'miss', 'accuracy',
            'f_score' or 'eval_time'. Default is 'auroc'.
        """
        header = '\n\nEvaluation results ({}):\n-----------------------\n'.format(metric)
        f = open(filename, 'a')
        f.write(header)
        df = self.get_pandas_df(metric)
        df.to_csv(f, sep='\t', na_rep='NA')
        f.close()

    def write_all(self, filename, repeats='avg'):
        """
        Writes for all networks, methods and performance metrics the corresponding values to a file. If multiple Results
        for the same network/method combination were logged (multiple repetitions of the experiment), the method can
        return the average or all logged values.

        Parameters
        ----------
        filename : string
            A file where to store the results.
        repeats : string, optional
            Can be one of 'all', 'avg'. Default is 'avg'.

        Notes
        -----
        For uncountable 'metrics' such as the node pair embedding operator (i.e 'edge_embed_method'), avg returns the
        most frequent item in the vector.

        Examples
        --------
        Read a scoresheet and write all metrics to a file with repeats='avg':

        >>> scores = pickle.load(open('lp_eval_1207_1638/eval.pkl', 'rb'))
        >>> scores.write_all('./test.txt')
        >>> print(open('test.txt', 'rb').read())
        Network_1 Network
        ---------------------------
        katz:
         tn:  	 684.0
         fp:  	 0.0
         fn:  	 684.0
         tp:  	 0.0
         auroc:  	 0.8203
        ...

        Read a scoresheet and write all metrics to a file with repeats='all':

        >>> scores = pickle.load(open('lp_eval_1207_1638/eval.pkl', 'rb'))
        >>> scores.write_all('./test.txt', 'all')
        >>> print(open('test.txt', 'rb').read())
        Network_1 Network
        ---------------------------
        katz:
         tn:  	 [684 684]
         fp:  	 [0 0]
         fn:  	 [684 684]
         tp:  	 [0 0]
         auroc:  	 [0.8155 0.8252]
        ...

        """
        f = open(filename, 'a+b')

        # Loop over all datasets
        for k1 in self._scoresheet:
            f.write(('\n\n{} Network'.format(k1)).encode())
            f.write('\n---------------------------'.encode())
            # Loop over all methods
            for k2 in self._scoresheet[k1]:
                f.write(('\n{}:'.format(k2)).encode())
                f.write('\n '.encode())
                # Loop over all metrics (auroc, pr, f_score...)
                for k3 in self._scoresheet[k1][k2]:
                    if repeats == 'avg':
                        # Compute average over all exp repeats for each metric
                        if k3 == 'edge_embed_method':
                            count = Counter(self._scoresheet[k1][k2][k3])
                            f.write((k3 + ':  \t ' + count.most_common(1)[0][0] + '\n ').encode())
                        else:
                            avg = np.around(np.mean(np.array(self._scoresheet[k1][k2][k3])), 4)
                            f.write((k3 + ':  \t ' + str(avg) + '\n ').encode())
                    else:
                        # Report all values for each exp repeat
                        if k3 == 'edge_embed_method':
                            vals = self._scoresheet[k1][k2][k3]
                            f.write((k3 + ':  \t ' + str(vals) + '\n ').encode())
                        else:
                            vals = np.around(np.array(self._scoresheet[k1][k2][k3]), 4)
                            f.write((k3 + ':  \t ' + str(vals) + '\n ').encode())

        # Close the file
        f.close()

    def write_pickle(self, filename):
        """
        Writes a pickle representation of this object to a file.

        Parameters
        ----------
        filename : string
            A file where to store the pickle representation.
        """
        pickle.dump(self, open(filename, "wb"))


class Results(object):
    """
    Class that encapsulates the train and test predictions of one method on a specific network and set of parameters.
    The train and test predictions are stored as Scores objects. Functions for plotting, printing and saving to files
    the train and test scores are provided. Supports binary classification only.

    Parameters
    ----------
    method : string
        A string representing the name of the method associated with these results.
    params : dict
        A dictionary of parameters used to obtain these results. Includes wall clock time of method evaluation.
    train_pred : ndarray
        An array containing the train predictions.
    train_labels : ndarray
        An array containing the train labels.
    test_pred : ndarray, optional
        An array containing the test predictions. Default is None.
    test_labels : ndarray, optional
        An array containing the test labels. Default is None.
    label_binarizer : string or Sklearn binary classifier, optional
        If the predictions returned by the model are not binary, this parameter indicates how these binary
        predictions should be computed in order to be able to provide metrics such as the confusion matrix.
        Any Sklear binary classifier can be used or the keyword 'median' which will used the prediction medians
        as binarization thresholds. Default is LogisticRegression(solver='liblinear')

    Attributes
    ----------
    method : string
        A string representing the name of the method associated with these results.
    params : dict
        A dictionary of parameters used to obtain these results. Includes wall clock time of method evaluation.
    binary_preds : bool
        A bool indicating if the train and test predictions are binary or not.
    train_scores : Scores
        A Scores object containing train scores.
    test_scores : Scores, optional
        A Scores object containing test scores. Default is None.
    label_binarizer : string or Sklearn binary classifier, optional
        If the predictions returned by the model are not binary, this parameter indicates how these binary
        predictions should be computed in order to be able to provide metrics such as the confusion matrix.
        Any Sklearn binary classifier can be used or the keyword 'median' which will used the prediction medians
        as binarization thresholds. Default is LogisticRegression(solver='liblinear')

    Raises
    ------
    AttributeError
        If the label binarizer is set to an incorrect value.
    """

    def __init__(self, method, params, train_pred, train_labels, test_pred=None, test_labels=None,
                 label_binarizer=LogisticRegression(solver='liblinear')):
        self.params = params
        self.method = method
        self.label_binarizer = label_binarizer
        self.binary_preds = self._check_binary(train_pred, test_pred)
        self.train_scores = None
        self.test_scores = None
        self._init_scores(train_pred, train_labels, test_pred, test_labels)

    @staticmethod
    def _check_binary(train_pred, test_pred):
        """
        Method that checks if the train and test predictions are binary.

        Parameters
        ----------
        train_pred : ndarray
            An array containing the train predictions.
        test_pred : ndarray, optional
            An array containing the test predictions.

        Returns
        -------
        binary_preds : bool
            A bool indicating if the train and test predictions are binary or not.
        """
        if test_pred is None:
            if ((train_pred == 0) | (train_pred == 1)).all():
                return True
            else:
                return False
        else:
            if ((train_pred == 0) | (train_pred == 1)).all() and ((test_pred == 0) | (test_pred == 1)).all():
                return True
            else:
                return False

    def _init_scores(self, train_pred, train_labels, test_pred, test_labels):
        """
        Method that creates the train and test Scores objects.

        Parameters
        ----------
        train_pred : ndarray
            An array containing the train predictions.
        train_labels : ndarray
            An array containing the train labels.
        test_pred : ndarray, optional
            An array containing the test predictions.
        test_labels : ndarray, optional
            An array containing the test labels.
        """
        # Check if the predictions are binary or not
        if self.binary_preds:
            # Create the score objects
            self.train_scores = Scores(y_true=train_labels, y_pred=train_pred, y_bin=train_pred)
            if test_pred is not None:
                self.test_scores = Scores(y_true=test_labels, y_pred=test_pred, y_bin=test_pred)
        else:
            if self.label_binarizer == 'median':
                # Compute binarized predictions using the median
                th1 = np.median(train_pred)
                train_bin = np.where(train_pred >= th1, 1, 0)
                if test_pred is not None:
                    th2 = np.median(test_pred)
                    test_bin = np.where(test_pred >= th2, 1, 0)
            else:
                try:
                    # Compute the binarized predictions
                    self.label_binarizer.fit(train_pred.reshape(-1, 1), train_labels)
                    train_bin = self.label_binarizer.predict(train_pred.reshape(-1, 1))
                    if test_pred is not None:
                        test_bin = self.label_binarizer.predict(test_pred.reshape(-1, 1))
                except AttributeError:
                    print('The label_binarizer is set to an incorrect value! '
                          'Method predictions are not binary so a correct label_binarizer is required.')
                    raise

            # Create the score objects
            self.train_scores = Scores(y_true=train_labels, y_pred=train_pred, y_bin=train_bin)
            if test_pred is not None:
                self.test_scores = Scores(y_true=test_labels, y_pred=test_pred, y_bin=test_bin)

    def plot(self, filename=None, results='auto', curve='all'):
        """
        Plots PR or ROC curves of the train or test predictions. If a filename is provided, the method will store the
        plot in pdf format to a file named <filename>+'_PR.pdf' or <filename>+'_ROC.pdf'.

        Parameters
        ----------
        filename : string, optional
            A string indicating the path and name of the file where to store the plot. If None, the plots are only
            shown on screen. Default is None.
        results : string, optional
            A string indicating if the 'train' or 'test' predictions should be used. Default is 'auto' (selects
            'test' if test predictions are logged and 'train' otherwise).
        curve : string, optional
            Can be one of 'all', 'pr' or 'roc'. Default is 'all' (generates both curves).

        Raises
        ------
        ValueError
            If test results are requested but not initialized in constructor.
        """
        # Get the appropriate train or test scores
        if results == 'train':
            scores = self.train_scores
        elif results == 'test':
            if self.test_scores is not None:
                scores = self.test_scores
            else:
                raise ValueError('Test scores not initialized!')
        else:
            if self.test_scores is not None:
                results = 'test'
                scores = self.test_scores
            else:
                results = 'train'
                scores = self.train_scores

        if curve == 'all' or curve == 'pr':
            precision, recall, _ = precision_recall_curve(scores.y_true, scores.y_pred)
            viz.plot_curve('{}_{}_PR.pdf'.format(filename, results), recall, precision, 'Recall', 'Precision',
                           '{} {} PR curve'.format(self.method, results))

        if curve == 'all' or curve == 'roc':
            tolerance = 0.25
            if np.sum(scores.y_true) < tolerance * len(scores.y_true) or \
                    np.sum(scores.y_true) > (1 - tolerance) * len(scores.y_true):
                warnings.warn('ROC curves are not recommended in the case of extreme class imbalance. '
                              'PR curves should be preferred.', Warning)
            fpr, tpr, thresholds = roc_curve(scores.y_true, scores.y_pred)
            viz.plot_curve('{}_{}_ROC.pdf'.format(filename, results), fpr, tpr, 'False positive rate',
                           'True positive rate', '{} {} ROC curve'.format(self.method, results))

    def save(self, filename, results='auto', precatk_vals=None):
        """
        Writes the method name, execution parameters, and all available performance metrics (for train or test
        predictions) to a file.

        Parameters
        ----------
        filename : string or file
            A file or filename where to store the output.
        results : string, optional
            A string indicating if the 'train' or 'test' predictions should be used. Default is 'auto' (selects
            'test' if test predictions are logged and 'train' otherwise).
        precatk_vals : list of int or None, optional
            The values for which the precision at k should be computed. Default is None.

        Raises
        ------
        ValueError
            If test results are required but not initialized in constructor.

        See Also
        --------
        get_all : Describes all the performance metrics that can be computed from train or test predictions.
        """
        f = open(filename, 'a+')
        f.write("Method: {}".format(self.method))
        f.write("\nParameters: ")
        for k, v in self.params.items():
            f.write(str(k) + ": " + str(v) + ", ")

        # Get the appropriate train or test scores
        if results == 'train':
            f.write("\nTrain scores: ")
        elif results == 'test':
            if self.test_scores is not None:
                f.write("\nTest scores: ")
            else:
                raise ValueError('Test scores not initialized!')
        else:
            if self.test_scores is not None:
                f.write("\nTest scores: ")
            else:
                f.write("\nTrain scores: ")

        metric_names, metric_vals = self.get_all(results, precatk_vals)
        for i in range(len(metric_names)):
            f.write("\n {} = {}".format(metric_names[i], metric_vals[i]))
        f.write("\n\n")
        f.close()

    def pretty_print(self, results='auto', precatk_vals=None):
        """
        Prints to screen the method name, execution parameters, and all available performance metrics (for train or test
        predictions).

        Parameters
        ----------
        results : string, optional
            A string indicating if the 'train' or 'test' predictions should be used. Default is 'auto' (selects
            'test' if test predictions are logged and 'train' otherwise).
        precatk_vals : list of int or None, optional
            The values for which the precision at k should be computed. Default is None.

        Raises
        ------
        ValueError
            If test results are requested but not initialized in constructor.

        See Also
        --------
        get_all : Describes all the performance metrics that can be computed from train or test predictions.
        """
        print("Method: {}".format(self.method))
        print("Parameters: ")
        print(self.params.items())

        # Get the appropriate train or test scores
        if results == 'train':
            print("Train scores: ")
        elif results == 'test':
            if self.test_scores is not None:
                print("Test scores: ")
            else:
                raise ValueError('Test scores not initialized!')
        else:
            if self.test_scores is not None:
                print("Test scores: ")
            else:
                print("Train scores: ")

        metric_names, metric_vals = self.get_all(results, precatk_vals)
        for i in range(len(metric_names)):
            print("{} = {}".format(metric_names[i], metric_vals[i]))
        print("")

    def get_all(self, results='auto', precatk_vals=None):
        """
        Returns the names of all performance metrics that can be computed from train or test predictions and their
        associated values. These metrics are: 'tn', 'fp', 'fn', 'tp', 'auroc', 'precision', 'precisionatk', 'recall',
        'fallout', 'miss', 'accuracy' and 'f_score'.

        Parameters
        ----------
        results : string, optional
            A string indicating if the 'train' or 'test' predictions should be used. Default is 'auto' (selects
            'test' if test predictions are logged and 'train' otherwise).
        precatk_vals : list of int or None, optional
            The values for which the precision at k should be computed. Default is None.

        Raises
        ------
        ValueError
            If test results are requested but not initialized in constructor.
        """
        # Get the appropriate train or test scores
        if results == 'train':
            scores = self.train_scores
        elif results == 'test':
            if self.test_scores is not None:
                scores = self.test_scores
            else:
                raise ValueError('Test scores not initialized!')
        else:
            if self.test_scores is not None:
                scores = self.test_scores
            else:
                scores = self.train_scores

        # Add the available scores
        metric_names = ['tn', 'fp', 'fn', 'tp', 'auroc', 'precision', 'recall',
                        'fallout', 'miss', 'accuracy', 'f_score']
        metric_vals = [scores.tn, scores.fp, scores.fn, scores.tp, scores.auroc(), scores.precision(), scores.recall(),
                       scores.fallout(), scores.miss(), scores.accuracy(), scores.f_score()]

        # Add precision at k values
        if precatk_vals is not None:
            for i in precatk_vals:
                metric_names.append('prec@{}'.format(i))
                metric_vals.append(scores.precisionatk(i))

        return metric_names, metric_vals

    def save_predictions(self, filename, results='auto'):
        """
        Writes the method name, execution parameters, and the train or test predictions and corresponding labels to a
        file.

        Parameters
        ----------
        filename : string or file
            A file or filename where to store the output.
        results : string, optional
            A string indicating if the 'train' or 'test' predictions should be used. Default is 'auto' (selects
            'test' if test predictions are logged and 'train' otherwise).

        Raises
        ------
        ValueError
            If test results are required but not initialized in constructor.
        """
        f = open(filename, 'a+')
        f.write("Method: {}".format(self.method))
        f.write("\nParameters: ")
        for k, v in self.params.items():
            f.write(str(k) + ": " + str(v) + ", ")

        # Get the appropriate train or test predictions
        if results == 'train':
            scores = self.train_scores
            f.write("\nTrain predictions | Train labels ")
        elif results == 'test':
            if self.test_scores is not None:
                scores = self.test_scores
                f.write("\nTest predictions | Test labels ")
            else:
                raise ValueError('Test scores not initialized!')
        else:
            if self.test_scores is not None:
                scores = self.test_scores
                f.write("\nTest predictions | Test labels ")
            else:
                scores = self.train_scores
                f.write("\nTrain predictions | Train labels ")

        for i in range(len(scores.y_true)):
            f.write("\n {} {}".format(scores.y_pred[i].item(), scores.y_true[i].item()))
        f.write("\n")
        f.close()


class Scores(object):
    """
    Class that encapsulates train or test predictions and exposes methods to compute different performance metrics.
    Supports binary classification only.

    Parameters
    ----------
    y_true : ndarray
        An array containing the true labels.
    y_pred : ndarray
       An array containing the predictions.
    y_bin : ndarray
        An array containing binarized predictions.

    Attributes
    ----------
    y_true : ndarray
        An array containing the true labels.
    y_pred : ndarray
       An array containing the predictions.
    y_bin : ndarray
        An array containing binarized predictions.
    tn : float
        The number of true negative in prediction.
    fp : float
        The number of false positives in prediction.
    fn : float
        The number of false negatives in prediction.
    tp : float
        The number of true positives in prediction.
    """

    def __init__(self, y_true, y_pred, y_bin):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_bin = np.array(y_bin)
        self._sorted = sorted(zip(self.y_true, self.y_pred), key=lambda x: x[1], reverse=True)
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.y_true, self.y_bin).ravel()

    def precision(self):
        """
        Computes the precision in prediction.

        Returns
        -------
        precision : float
            The prediction precision score.
        """
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) != 0 else float('NaN')

    def precisionatk(self, k=100):
        """
        Computes the precision at k score.

        Parameters
        ----------
        k : int, optional
            The k value for which to compute the precision score. Default is 100.

        Returns
        -------
        precisionatk : float
            The prediction precision score for value k.
        """
        if k > len(self._sorted):
            MAX = len(self._sorted)
        else:
            MAX = k

        aux = list(zip(*self._sorted))[0]
        rel = sum(aux[:MAX])
        return (1.0 * rel) / k if k != 0 else float('NaN')

    def recall(self):
        """
        Computes the recall in prediction.

        Returns
        -------
        recall : float
            The prediction recall score.
        """
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) != 0 else float('NaN')

    def fallout(self):
        """
        Computes the fallout in prediction.

        Returns
        -------
        fallout : float
            The prediction fallout score.
        """
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) != 0 else float('NaN')

    def miss(self):
        """
        Computes the miss in prediction.

        Returns
        -------
        miss : float
            The prediction miss score.
        """
        return self.fn / (self.fn + self.tn) if (self.fn + self.tn) != 0 else float('NaN')

    def accuracy(self):
        """
        Computes the accuracy score.

        Returns
        -------
        accuracy : float
            The prediction accuracy score.
        """
        return accuracy_score(self.y_true, self.y_bin)

    def f_score(self, beta=1):
        """
        Computes the F-score as the weighted harmonic mean of precision and recall.

        Parameters
        ----------
        beta : float, optional
            Allows to assign more weight to precision or recall.
            If beta > 1, recall is emphasized over precision.
            If beta < 1, precision is emphasized over recall.

        Returns
        -------
        f_score : float
            The prediction f_score.

        Notes
        -----
        The generalized form is used, where P and R represent precision and recall, respectively:

        .. math::

            F = (\\beta^2 + 1) \\cdot P \\cdot R / (\\beta^2 \\cdot P + R)

            F = (\\beta^2 + 1) \\cdot tp / ((\\beta^2 + 1) \\cdot tp + \\beta^2 \\cdot fn + fp)

        """
        beta2 = beta ** 2
        beta2_tp = (beta2 + 1) * self.tp
        den = (beta2_tp + beta2 * self.fn + self.fp)
        return beta2_tp / den if den != 0 else float('NaN')

    def auroc(self):
        """
        Computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

        Returns
        -------
        auroc : float
            The prediction auroc score.

        Notes
        -----
        Throws a warning if class imbalance is detected.
        """
        tolerance = 0.1
        if np.sum(self.y_true) < tolerance * len(self.y_true) or \
                np.sum(self.y_true) > (1 - tolerance) * len(self.y_true):
            warnings.warn('AUROC is not recommended in the case of extreme class imbalance. ', Warning)
        return roc_auc_score(self.y_true, self.y_pred)


class NCResults(object):
    """
    Class that encapsulates the train and test predictions of one method on a specific network and set of parameters.
    The train and test predictions are stored as NCScores objects. Functions for plotting, printing and saving to files
    the train and test scores are provided. Supports multi-label classification.

    Parameters
    ----------
    method : string
        A string representing the name of the method associated with these results.
    params : dict
        A dictionary of parameters used to obtain these results. Includes wall clock time of method evaluation.
    train_pred : ndarray
        An array containing the train predictions.
    train_labels : ndarray
        An array containing the train labels.
    test_pred : ndarray, optional
        An array containing the test predictions. Default is None.
    test_labels : ndarray, optional
        An array containing the test labels. Default is None.

    Attributes
    ----------
    method : string
        A string representing the name of the method associated with these results.
    params : dict
        A dictionary of parameters used to obtain these results. Includes wall clock time of method evaluation.
    train_scores : Scores
        An NCScores object containing train scores.
    test_scores : Scores, optional
        An NCScores object containing test scores. Default is None.
    """

    def __init__(self, method, params, train_pred, train_labels, test_pred=None, test_labels=None):
        self.params = params
        self.method = method
        self.train_scores = None
        self.test_scores = None
        self._init_scores(train_pred, train_labels, test_pred, test_labels)

    def _init_scores(self, train_pred, train_labels, test_pred, test_labels):
        """
        Method that creates the train and test NCScores objects.

        Parameters
        ----------
        train_pred : ndarray
            An array containing the train predictions.
        train_labels : ndarray
            An array containing the train labels.
        test_pred : ndarray, optional
            An array containing the test predictions.
        test_labels : ndarray, optional
            An array containing the test labels.
        """
        # Create the NCScores
        self.train_scores = NCScores(y_true=train_labels, y_pred=train_pred)
        if test_pred is not None:
            self.test_scores = NCScores(y_true=test_labels, y_pred=test_pred)

    def save(self, filename, results='auto'):
        """
        Writes the method name, execution parameters, and all available performance metrics (for train or test
        predictions) to a file.

        Parameters
        ----------
        filename : string or file
            A file or filename where to store the output.
        results : string, optional
            A string indicating if the 'train' or 'test' predictions should be used. Default is 'auto' (selects
            'test' if test predictions are logged and 'train' otherwise).

        Raises
        ------
        ValueError
            If test results are required but not initialized in constructor.

        See Also
        --------
        get_all : Describes all the performance metrics that can be computed from train or test predictions.
        """
        f = open(filename, 'a+')
        f.write("Method: {}".format(self.method))
        f.write("\nParameters: ")
        for k, v in self.params.items():
            f.write(str(k) + ": " + str(v) + ", ")

        # Get the appropriate train or test scores
        if results == 'train':
            f.write("\nTrain scores: ")
        elif results == 'test':
            if self.test_scores is not None:
                f.write("\nTest scores: ")
            else:
                raise ValueError('Test scores not initialized!')
        else:
            if self.test_scores is not None:
                f.write("\nTest scores: ")
            else:
                f.write("\nTrain scores: ")

        metric_names, metric_vals = self.get_all(results)
        for i in range(len(metric_names)):
            f.write("\n {} = {}".format(metric_names[i], metric_vals[i]))
        f.write("\n\n")
        f.close()

    def pretty_print(self, results='auto'):
        """
        Prints to screen the method name, execution parameters, and all available performance metrics (for train or test
        predictions).

        Parameters
        ----------
        results : string, optional
            A string indicating if the 'train' or 'test' predictions should be used. Default is 'auto' (selects
            'test' if test predictions are logged and 'train' otherwise).

        Raises
        ------
        ValueError
            If test results are requested but not initialized in constructor.

        See Also
        --------
        get_all : Describes all the performance metrics that can be computed from train or test predictions.
        """
        print("Method: {}".format(self.method))
        print("Parameters: ")
        print(self.params.items())

        # Get the appropriate train or test scores
        if results == 'train':
            print("Train scores: ")
        elif results == 'test':
            if self.test_scores is not None:
                print("Test scores: ")
            else:
                raise ValueError('Test scores not initialized!')
        else:
            if self.test_scores is not None:
                print("Test scores: ")
            else:
                print("Train scores: ")

        metric_names, metric_vals = self.get_all(results)
        for i in range(len(metric_names)):
            print("{} = {}".format(metric_names[i], metric_vals[i]))
        print("")

    def get_all(self, results='auto', precatk_vals=None):
        """
        Returns the names of all performance metrics that can be computed from train or test predictions and their
        associated values. These metrics are: 'f1_micro', 'f1_macro', 'f1_weighted'.

        Parameters
        ----------
        results : string, optional
            A string indicating if the 'train' or 'test' predictions should be used. Default is 'auto' (selects
            'test' if test predictions are logged and 'train' otherwise).
        precatk_vals : None, optional
            Not used.

        Raises
        ------
        ValueError
            If test results are requested but not initialized in constructor.
        """
        # Get the appropriate train or test scores
        if results == 'train':
            scores = self.train_scores
        elif results == 'test':
            if self.test_scores is not None:
                scores = self.test_scores
            else:
                raise ValueError('Test scores not initialized!')
        else:
            if self.test_scores is not None:
                scores = self.test_scores
            else:
                scores = self.train_scores

        # Add the available scores
        metric_names = ['f1_micro', 'f1_macro', 'f1_weighted']
        metric_vals = [scores.f1_micro(), scores.f1_macro(), scores.f1_weighted()]

        return metric_names, metric_vals

    def save_predictions(self, filename, results='auto'):
        """
        Writes the method name, execution parameters, and the train or test predictions to a file.

        Parameters
        ----------
        filename : string or file
            A file or filename where to store the output.
        results : string, optional
            A string indicating if the 'train' or 'test' predictions should be used. Default is 'auto' (selects
            'test' if test predictions are logged and 'train' otherwise).

        Raises
        ------
        ValueError
            If test results are required but not initialized in constructor.
        """
        f = open(filename, 'a+')
        f.write("Method: {}".format(self.method))
        f.write("\nParameters: ")
        for k, v in self.params.items():
            f.write(str(k) + ": " + str(v) + ", ")

        # Get the appropriate train or test predictions
        if results == 'train':
            scores = self.train_scores
            f.write("\nTrain predictions | Train labels ")
        elif results == 'test':
            if self.test_scores is not None:
                scores = self.test_scores
                f.write("\nTest predictions | Test labels ")
            else:
                raise ValueError('Test scores not initialized!')
        else:
            if self.test_scores is not None:
                scores = self.test_scores
                f.write("\nTest predictions | Test labels ")
            else:
                scores = self.train_scores
                f.write("\nTrain predictions | Train labels ")

        for i in range(len(scores.y_true)):
            f.write("\n {} {}".format(scores.y_pred[i].item(), scores.y_true[i].item()))
        f.write("\n")
        f.close()


class NCScores(object):
    """
    Class that encapsulates train or test predictions and exposes methods to compute different performance metrics.
    Supports multi-label classification.

    Parameters
    ----------
    y_true : ndarray
        An array containing the true labels.
    y_pred : ndarray
       An array containing the predictions.

    Attributes
    ----------
    y_true : ndarray
        An array containing the true labels.
    y_pred : ndarray
       An array containing the predictions.
    """

    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self._sorted = sorted(zip(self.y_true, self.y_pred), key=lambda x: x[1], reverse=True)

    def f1_micro(self):
        """
        Computes the f1 score globally for all labels (i.e. sums the tp for all classes and divides by the sum of all
        tp+fp).

        Returns
        -------
        f1_micro : float
            The f1 micro score.
        """
        return f1_score(self.y_true, self.y_pred, average='micro')

    def f1_macro(self):
        """
        Computes the f1 score for each label, and finds their unweighted average. This metric does not take label
        imbalance into account.

        Returns
        -------
        f1_macro : float
            The f1 macro score.
        """
        return f1_score(self.y_true, self.y_pred, average='macro')

    def f1_weighted(self):
        """
        Computes the f1 score for each label, and finds their average, weighted by support (the number of true instances
        for each label).

        Returns
        -------
        f1_weighted : float
            The weighted f1 score.
        """
        return f1_score(self.y_true, self.y_pred, average='weighted')
