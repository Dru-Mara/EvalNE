#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018


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

import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt


class Scoresheet:
    """
    This class simplifies the logging and management of the evaluation results and execution times. Functions for
    logging, plotting and saving the results are provided. The Scoresheet only logs the specified metrics and not
    the full train or test predictions.

    Parameters
    ----------
    tr_te : basestring, optional
        A string indicating if the 'train' or 'test' results should be stored. Default is 'test'.
    precatk_vals : list of int or None, optional
        The values for which the precision at k should be computed. Default is None
    """

    def __init__(self, tr_te='test', precatk_vals=None):
        self._tr_te = tr_te
        self._precatk_vals = precatk_vals
        self._scoresheet = OrderedDict()
        self._all_methods = OrderedDict()

    def log_results(self, results):
        r"""
        Logs the Results object or list of Results objects given as input. All metrics are stored including execution
        time which is extracted from the Results class parameter list. Is the same combination of network/algorithm is
        found more than once, the results are stored in a vector.

        Parameters
        ----------
        results : Results or list of Results
            The Results object or list of objects to be logged in the Scoresheet.
        """
        if isinstance(results, Results):
            self._log_result(results)
        else:
            for res in results:
                self._log_result(res)

    def _log_result(self, result):
        r"""
        Logs the Results object given as input.

        Parameters
        ----------
        result : Results
            The Results object obtained form the evaluation of a certain method which is to be logged.
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
                    self._scoresheet[k1][k2][metrics[i]].append(vals[i])
                self._scoresheet[k1][k2]['eval_time'].append(result.params['eval_time'])
                self._scoresheet[k1][k2]['edge_embed_method'].append(result.params.get('edge_embed_method', 'None'))

            else:
                # Method is not yet in the dict, so we add method and metrics
                metrics, vals = result.get_all(self._tr_te, self._precatk_vals)
                self._scoresheet[k1][k2] = OrderedDict(zip(metrics, map(lambda x: [x], vals)))
                self._scoresheet[k1][k2]['eval_time'] = [result.params['eval_time']]
                self._scoresheet[k1][k2]['edge_embed_method'] = [result.params.get('edge_embed_method', 'None')]

        else:
            # Dataset is not yet in the dict, so we add dataset, method and metrics
            metrics, vals = result.get_all(self._tr_te, self._precatk_vals)
            aux = OrderedDict(zip(metrics, map(lambda x: [x], vals)))
            self._scoresheet[k1] = OrderedDict({k2: aux})
            self._scoresheet[k1][k2]['eval_time'] = [result.params['eval_time']]
            self._scoresheet[k1][k2]['edge_embed_method'] = [result.params.get('edge_embed_method', 'None')]

    def get_pandas_df(self, metric='auroc', repeat=None):
        r"""
        Returns a view of the Scoresheet as a pandas DataFrame for the specified metric. The columns of the DataFrame
        represent different networks and the rows different methods. If the same network/method combination is present
        more than once, the average is computed.

        Parameters
        ----------
        metric : basestring, optional
            Can be one of 'tn', 'fp', 'fn', 'tp', 'auroc', 'precision', 'recall', 'fallout', 'miss', 'accuracy',
            'f_score', 'eval_time' or 'edge_embed_method'. Default is 'auroc'.
        repeat : int, optional
            An int indicating the experiment repeat for which the results should be returned. If not indicated, the
            average over all repeats will be computed and returned. Default is None (computes average over repeats).

        Returns
        -------
        df : pandas.DataFrame
            A pandas DataFrame view of the Scoresheet for the specified metric.

        Raises
        ------
        ValueError
            If the requested metric does not exist.
            If the Scoresheet is empty so a dataframe can not be generated.
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
        r"""
        Returns a latex table containing the specified metric value for each combination of network/algorithm logged.
        If the same network/method combination is present more than once in the Scoresheet, the average is returned.

        Parameters
        ----------
        metric : basestring, optional
            Can be one of 'tn', 'fp', 'fn', 'tp', 'auroc', 'precision', 'recall', 'fallout', 'miss', 'accuracy',
            'f_score', 'eval_time' or 'edge_embed_method'. Default is 'auroc'.

        Returns
        -------
        latex_table : basestring
            A string containing the latex representation of the DataFrame for the input metric.
        """
        df = self.get_pandas_df(metric)
        return df.to_latex()

    def print_tabular(self, metric='auroc'):
        r"""
        Prints in tabular format the average over all logs of the specified metric for each existing network/algorithm
        combinations.

        Parameters
        ----------
        metric : basestring, optional
            Can be one of 'tn', 'fp', 'fn', 'tp', 'auroc', 'precision', 'recall', 'fallout', 'miss', 'accuracy',
            'f_score', 'eval_time' or 'edge_embed_method'. Default is 'auroc'.
        """
        print(self.get_pandas_df(metric))

    def write_tabular(self, filename, metric='auroc'):
        r"""
        Writes in tabular format the average over all logs of the specified metric for each existing network/algorithm
        combinations to the specified file.

        Parameters
        ----------
        filename : basestring
            The file where to store the results.
        metric : basestring, optional
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
        r"""
        Writes for each network and algorithm combination the results corresponding to every metric available.
        If the same network/method combination has been logged more than once, and `repeats` is set to `avg` the
        average over all repeats is computed. Otherwise all values are written as an array.

        Parameters
        ----------
        filename : basestring
            The file where to store the results.
        repeats : basestring, optional
            Can be one of 'all', 'avg'. Default is 'avg'.
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
        r"""
        Stores to the given file a representation of this object as a pickle file.

        Parameters
        ----------
        filename : basestring
            The file where to store the results.
        """
        pickle.dump(self, open(filename, "wb"))


class Results(object):
    """
    Contains the train and test results of the link prediction task for a certain method and set of parameters.
    Exposes the train and test score objects and implements functionality for conveniently retrieving results as
    plots, text files or command line outputs.

    Parameters
    ----------
    method : basestring
        A string representing the name of the method associated with these scores.
    params : dict
        A dictionary of parameters used to obtain these scores. Includes wall clock time of method evaluation.
    train_pred : ndarray
        An array containing the train predictions.
    train_labels : ndarray
        An array containing the train true labels.
    test_pred : ndarray, optional
        An array containing the test predictions. Default is None.
    test_labels : ndarray, optional
        An array containing the test true labels. Default is None.
    label_binarizer : string or Sklearn binary classifier, optional
        If the predictions returned by the model are not binary, this parameter indicates how these binary
        predictions should be computed in order to be able to provide metrics such as the confusion matrix.
        Any Sklear binary classifier can be used or the keyword 'median' which will used the prediction medians
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
        self._init(train_pred, train_labels, test_pred, test_labels)

    @staticmethod
    def _check_binary(train_pred, test_pred):
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

    def _init(self, train_pred, train_labels, test_pred, test_labels):
        r"""
        Initializes the train and test scores.
        """
        # Check if the predictions are binary or not
        if self.binary_preds:
            # Create the scoresheets
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

            # Create the scoresheets
            self.train_scores = Scores(y_true=train_labels, y_pred=train_pred, y_bin=train_bin)
            if test_pred is not None:
                self.test_scores = Scores(y_true=test_labels, y_pred=test_pred, y_bin=test_bin)

    def _plot(self, results, x, y, x_label, y_label, curve, filename):
        r"""
        Contains the actual plot functionality.
        """
        plt.plot(x, y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        if results == 'test':
            plt.title('{} test set {} curve'.format(self.method, curve))
        else:
            plt.title('{} train set {} curve'.format(self.method, curve))
        if filename is not None:
            plt.savefig(filename + '_' + curve + '.pdf')
            plt.close()
        else:
            plt.show()

    def plot(self, filename=None, results='auto', curve='all'):
        r"""
        Plots the required curve. The filenames will be appended a '_PR.pdf' or '_Roc.pdf'.

        Parameters
        ----------
        filename : basestring, optional
            The name that will be given to the output plots. If None, the plots are only shown on screen.
            Default is None.
        results : basestring, optional
            A string indicating if the 'train' or 'test' results should be shown. Default is 'auto' which selects 'test'
            if test_scores is not None and 'train' otherwise.
        curve : basestring, optional
            Can be one of 'all', 'pr' or 'roc'.
            Default is 'all'.

        Raises
        ------
        ValueError
            If test results are required but not initialized in constructor.
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

        if curve == 'all' or curve == 'pr':
            precision, recall, _ = precision_recall_curve(scores.y_true, scores.y_pred)
            self._plot(results, recall, precision, 'Recall', 'Precision', 'PR', filename)

        if curve == 'all' or curve == 'roc':
            tolerance = 0.25
            if np.sum(scores.y_true) < tolerance * len(scores.y_true) or \
                    np.sum(scores.y_true) > (1 - tolerance) * len(scores.y_true):
                warnings.warn('ROC curves are not recommended in the case of extreme class imbalance. '
                              'PR curves should be preferred.', Warning)
            fpr, tpr, thresholds = roc_curve(scores.y_true, scores.y_pred)
            self._plot(results, fpr, tpr, 'False positive rate', 'True positive rate', 'Roc', filename)

    def save(self, filename, results='auto', precatk_vals=None):
        r"""
        Saves to a file the method name, parameters, if scores are for train or test data and all the scores.

        Parameters
        ----------
        filename : basestring
            The name of the output file where the results will be stored.
        results : basestring, optional
            A string indicating if the 'train' or 'test' results should be saved. Default is 'auto' which selects 'test'
            if test_scores is not None and 'train' otherwise.
        precatk_vals : list of int or None, optional
            The values for which the precision at k should be computed. Default is None

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
        r"""
        Prints to screen the method name, parameters, if scores are for train or test data and all the scores available.

        Parameters
        ----------
        results : basestring, optional
            A string indicating if the 'train' or 'test' results should be shown. Default is 'auto' which selects 'test'
            if test_scores is not None and 'train' otherwise.
        precatk_vals : list of int or None, optional
            The values for which the precision at k should be computed. Default is None.

        Raises
        ------
        ValueError
            If test results are required but not initialized in constructor.
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
        r"""
        Returns all the metrics available and their associated values as two lists.

        Parameters
        ----------
        results : basestring, optional
            A string indicating if the 'train' or 'test' results should be shown. Default is 'auto' which selects 'test'
            if test_scores is not None and 'train' otherwise.
        precatk_vals : list of int or None, optional
            The values for which the precision at k should be computed. Default is None.

        Raises
        ------
        ValueError
            If test results are required but not initialized in constructor.
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


class Scores(object):
    """
    Object that encapsulates the results (train or test) and exposes methods to compute different scores

    Parameters
    ----------
    y_true : ndarray
        An array containing the true labels.
    y_pred : ndarray
       An array containing the predictions.
    y_bin : ndarray
        An array containing binarized predictions.
    """

    def __init__(self, y_true, y_pred, y_bin):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_bin = np.array(y_bin)
        self._sorted = sorted(zip(self.y_true, self.y_pred), key=lambda x: x[1], reverse=True)
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.y_true, self.y_bin).ravel()

    def precision(self):
        r"""
        Computes the precision in prediction defined as:
            TP / (TP + FP)

        Returns
        -------
        precision : float
            A value indicating the precision.
        """
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) != 0 else float('NaN')

    def precisionatk(self, k=100):
        r"""
        Computes the precision at k score.

        Parameters
        ----------
        k : int, optional
            The k value for which to compute the precision score.
            Default is 100.

        Returns
        -------
        precisionatk : float
            A value indicating the precision at K.
        """
        if k > len(self._sorted):
            MAX = len(self._sorted)
        else:
            MAX = k

        aux = list(zip(*self._sorted))[0]
        rel = sum(aux[:MAX])
        return (1.0 * rel) / k if k != 0 else float('NaN')

    def recall(self):
        r"""
        Computes the recall in prediction.

        Returns
        -------
        recall : float
            A value indicating the recall.
        """
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) != 0 else float('NaN')

    def fallout(self):
        r"""
        Computes the fallout in prediction.

        Returns
        -------
        fallout : float
            A value indicating the prediction fallout.
        """
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) != 0 else float('NaN')

    def miss(self):
        r"""
        Computes the miss in prediction.

        Returns
        -------
        miss : float
            A value indicating the prediction miss.
        """
        return self.fn / (self.fn + self.tn) if (self.fn + self.tn) != 0 else float('NaN')

    def accuracy(self):
        r"""
        Computes the accuracy score.

        Returns
        -------
        accuracy : float
            A value indicating the accuracy score.
        """
        return accuracy_score(self.y_true, self.y_bin)

    def f_score(self, beta=1):
        r"""
        Computes the F-score as the harmonic mean of precision and recall:
            F = 2PR / (P + R)
        The generalized form is used:
            F = (beta^2 + 1)PR / (beta^2 P + R)
              = (beta^2 + 1)tp / ((beta^2 + 1)tp + beta^2fn + fp)

        Parameters
        ----------
        beta : float, optional
            Allows to assign more weight to precision or recall.
            If beta > 1, recall is emphasized over precision.
            If beta < 1, precision is emphasized over recall.

        Returns
        -------
        f_score : float
            A value indicating the f_score.
        """
        beta2 = beta ** 2
        beta2_tp = (beta2 + 1) * self.tp
        den = (beta2_tp + beta2 * self.fn + self.fp)
        return beta2_tp / den if den != 0 else float('NaN')

    def auroc(self):
        r"""
        Computes the area under the ROC curve score.

        Returns
        -------
        auroc : float
            A value indicating the area under the ROC curve score.
        """
        tolerance = 0.25
        if np.sum(self.y_true) < tolerance * len(self.y_true) or \
                np.sum(self.y_true) > (1 - tolerance) * len(self.y_true):
            warnings.warn('AUROC is not recommended in the case of extreme class imbalance. ', Warning)
        return roc_auc_score(self.y_true, self.y_pred)


class NCResults(object):
    """
    Contains the train and test results of the link prediction task for a certain method and set of parameters.
    Exposes the train and test score objects and implements functionality for conveniently retrieving results as
    plots, text files or command line outputs.

    Parameters
    ----------
    method : basestring
        A string representing the name of the method associated with these scores.
    params : dict
        A dictionary of parameters used to obtain these scores. Includes wall clock time of method evaluation.
    train_pred : ndarray
        An array containing the train predictions.
    train_labels : ndarray
        An array containing the train true labels.
    test_pred : ndarray, optional
        An array containing the test predictions. Default is None.
    test_labels : ndarray, optional
        An array containing the test true labels. Default is None.

    Raises
    ------
    AttributeError
        If the label binarizer is set to an incorrect value.
    """

    def __init__(self, method, params, train_pred, train_labels, test_pred=None, test_labels=None):
        self.params = params
        self.method = method
        self.train_scores = None
        self.test_scores = None
        self._init(train_pred, train_labels, test_pred, test_labels)

    def _init(self, train_pred, train_labels, test_pred, test_labels):
        r"""
        Initializes the train and test scores.
        """
        # Create the scoresheets
        self.train_scores = NCScores(y_true=train_labels, y_pred=train_pred)
        if test_pred is not None:
            self.test_scores = NCScores(y_true=test_labels, y_pred=test_pred)

    def save(self, filename, results='auto'):
        r"""
        Saves to a file the method name, parameters, if scores are for train or test data and all the scores.

        Parameters
        ----------
        filename : basestring
            The name of the output file where the results will be stored.
        results : basestring, optional
            A string indicating if the 'train' or 'test' results should be saved. Default is 'auto' which selects 'test'
            if test_scores is not None and 'train' otherwise.

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
        r"""
        Prints to screen the method name, parameters, if scores are for train or test data and all the scores available.

        Parameters
        ----------
        results : basestring, optional
            A string indicating if the 'train' or 'test' results should be shown. Default is 'auto' which selects 'test'
            if test_scores is not None and 'train' otherwise.

        Raises
        ------
        ValueError
            If test results are required but not initialized in constructor.
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
        r"""
        Returns all the metrics available and their associated values as two lists.

        Parameters
        ----------
        results : basestring, optional
            A string indicating if the 'train' or 'test' results should be shown. Default is 'auto' which selects 'test'
            if test_scores is not None and 'train' otherwise.
        precatk_vals : list of int or None, optional
            Not used.

        Raises
        ------
        ValueError
            If test results are required but not initialized in constructor.
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


class NCScores(object):
    """
    Object that encapsulates the results (train or test) and exposes methods to compute different scores

    Parameters
    ----------
    y_true : ndarray
        An array containing the true labels.
    y_pred : ndarray
       An array containing the predictions.
    y_bin : ndarray
        An array containing binarized predictions.
    """

    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self._sorted = sorted(zip(self.y_true, self.y_pred), key=lambda x: x[1], reverse=True)

    def f1_micro(self):
        return f1_score(self.y_true, self.y_pred, average='micro')

    def f1_macro(self):
        return f1_score(self.y_true, self.y_pred, average='macro')

    def f1_weighted(self):
        return f1_score(self.y_true, self.y_pred, average='weighted')
