#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# TODO: check if denominator is not 0 for the scores.

from __future__ import division

import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


class ScoreSheet(object):
    """
    Contains the results of the link prediction task for a certain method and certain parameters.
    Provides functions to compute different scores and plot curves.

    Parameters
    ----------
    method : basestring
        The methods name
    test_results : bool
        True if the scores refer to the train data and false if they refer to test data.
    params : dict
        A dictionary of parameters used to compute the results.
    y_true : ndarray
        A binary array containing the true edge labels.
    y_pred : ndarray
        An array containing the values predicted by the algorithm.
    y_bin : ndarray
        An array containing the predictions as binary values.
    """

    def __init__(self, method, test_results, params, y_true, y_pred, y_bin):
        self.method = method
        self.test_results = test_results
        self.params = params
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_bin = y_bin
        self._sorted = sorted(zip(self.y_true, self.y_pred), key=lambda x: x[1], reverse=True)
        self.tn = None
        self.fp = None
        self.fn = None
        self.tp = None
        # Compute the initial scores
        self._compute_scores()

    def _compute_scores(self):
        r"""
        Computes the confusion matrix using the binary predictions.
        """
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.y_true, self.y_bin).ravel()

    def _plot(self, x, y, x_label, y_label, curve, filename):
        r"""
        Contains the actual plot functionality.
        """
        plt.plot(x, y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        if self.test_results:
            plt.title('{} test set {} curve'.format(self.method, curve))
        else:
            plt.title('{} train set {} curve'.format(self.method, curve))
        if filename is not None:
            plt.savefig(filename + '_' + curve + '.pdf')
        else:
            plt.show()
        plt.clf()

    def plot(self, curve='all', filename=None):
        r"""
        Plots the required curve. The filenames will be appended a `_PR.pdf` or `_Roc.pdf`.

        Parameters
        ----------
        curve : basestring, optional
            Can be one of 'all', 'pr' or 'roc'.
            Default is 'all'.
        filename : basestring, optional
            The name that will be given to the output plots. If none the plots are only shown on screen.
            Default is None.
        """
        if curve == 'all' or curve == 'pr':
            precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred)
            self._plot(precision, recall, 'Precision', 'Recall', 'PR', filename)

        if curve == 'all' or curve == 'roc':
            warnings.warn('ROC curves are not recommended in the case of class imbalance. '
                          'PR curves should be preferred.', Warning)
            fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred)
            self._plot(fpr, tpr, 'False positive rate', 'True positive rate', 'Roc', filename)

    def save(self, filename):
        r"""
        Saves to a file the method name, parameters, if scores are for train or test data and all the scores.

        Parameters
        ----------
        filename : basestring
            The name of the output file where the results will be stored.
        """
        try:
            f = open(filename, 'a+')
            f.write("Method: {}".format(self.method))
            f.write("\nParameters: ")
            for k, v in self.params.items():
                f.write(str(k) + ": " + str(v) + ", ")
            if self.test_results:
                f.write("\nTest scores: ")
            else:
                f.write("\nTrain scores: ")
            f.write("\n tn = {}, fp = {}, fn = {}, tp = {}".format(self.tn, self.fp, self.fn, self.tp))
            f.write("\n Precision = {} \n Recall = {}".format(self.precision(), self.recall()))
            f.write("\n Fallout = {} \n Miss = {}".format(self.fallout(), self.miss()))
            f.write("\n Accuracy = {} \n f_score = {}".format(self.accuracy(), self.f_score()))
            f.write("\n Average_prec = {}".format(average_precision_score(self.y_true, self.y_pred)))
            f.write("\n AUROC = {}".format(self.auroc()))
            f.write("\n prec@1 = {}, prec@10 = {}, prec@100 = {}, prec@200 = {}, prec@500 = {}, "
                    "prec@1000 = {}, prec@10000 = {}".format(self.precisionatk(1), self.precisionatk(10),
                                                             self.precisionatk(100), self.precisionatk(200),
                                                             self.precisionatk(500), self.precisionatk(1000),
                                                             self.precisionatk(10000)))
            f.write("\n\n")
            f.close()
        except IOError:
            print("Could not read file:", filename)

    def pretty_print(self):
        r"""
        Prints to screen the method name, parameters, if scores are for train or test data and all the scores.
        """
        print("Method: {}".format(self.method))
        print("Parameters: ")
        print(self.params.items())
        if self.test_results:
            print("Test scores: ")
        else:
            print("Train scores: ")
        print(" tn = {} \n fp = {} \n fn = {} \n tp = {}".format(self.tn, self.fp, self.fn, self.tp))
        print(" Precision = {} \n Recall = {}".format(self.precision(), self.recall()))
        print(" Fallout = {} \n Miss = {}".format(self.fallout(), self.miss()))
        print(" AUROC = {} \n Accuracy = {} \n f_score = {}".format(self.auroc(), self.accuracy(), self.f_score()))
        print("")

    def precision(self):
        r"""
        Computes the precision in prediction defined as:
            TP / (TP + FP)

        Returns
        -------
        precision : float
            A value indicating the precision.
        """
        return self.tp / (self.tp + self.fp)

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
        rel = 0
        for i in range(MAX):
            rel += self._sorted[i][0]
        return rel/k

    def recall(self):
        r"""
        Computes the recall in prediction.

        Returns
        -------
        recall : float
            A value indicating the recall.
        """
        return self.tp / (self.tp + self.fn)

    def fallout(self):
        r"""
        Computes the fallout in prediction.

        Returns
        -------
        fallout : float
            A value indicating the prediction fallout.
        """
        return self.fp / (self.fp + self.tn)

    def miss(self):
        r"""
        Computes the miss in prediction.

        Returns
        -------
        miss : float
            A value indicating the prediction miss.
        """
        return self.fn / (self.fn + self.tn)

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
        Computes the F-score as the harmonic mean of precision adn recall:
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
        return beta2_tp / (beta2_tp + beta2 * self.fn + self.fp)

    def auroc(self):
        r"""
        Computes the area under the ROC curve score.

        Returns
        -------
        auroc : float
            A value indicating the area under the ROC curve score.
        """
        if np.sum(self.y_true) != len(self.y_true)/2.0:
            warnings.warn('AUROC is not recommended in the case of class imbalance. ', Warning)
        return roc_auc_score(self.y_true, self.y_pred)


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
        A dictionary of parameters used to obtain these scores.
    train_pred : ndarray
        An array containing the train predictions.
    train_labels : ndarray
        An array containing the train true labels.
    test_pred : ndarray
        An array containing the test predictions.
    test_labels : ndarray
        An array containing the test true labels.
    label_binarizer : string or Sklearn binary classifier, optional
        If the predictions returned by the model are not binary, this parameter indicates how these binary
        predictions should be computed in order to be able to provide metrics such as the confusion matrix.
        Any Sklear binary classifier can be used or the keyword 'median' which will used the prediction medians
        as binarization thresholds.
        Default is LogisticRegression(solver='liblinear')
    """

    def __init__(self, method, params, train_pred, train_labels, test_pred, test_labels,
                 label_binarizer=LogisticRegression(solver='liblinear')):
        self.method = method
        self.params = params
        self.label_binarizer = label_binarizer
        self.train_scores = None
        self.test_scores = None
        self._init(train_pred, train_labels, test_pred, test_labels)

    def _init(self, train_pred, train_labels, test_pred, test_labels):
        r"""
        Initializes the train and test scores.
        """
        # Check if the predictions are binary or not
        if ((train_pred == 0) | (train_pred == 1)).all():
            # Create the scoresheets
            self.train_scores = Scores(y_true=train_labels, y_pred=train_pred, y_bin=train_pred)
            self.test_scores = Scores(y_true=test_labels, y_pred=test_pred, y_bin=test_pred)
        else:
            if self.label_binarizer == 'median':
                # Compute binarized predictions using the median
                th1 = np.median(train_pred)
                th2 = np.median(test_pred)
                train_bin = np.where(train_pred >= th1, 1, 0)
                test_bin = np.where(test_pred >= th2, 1, 0)
            else:
                # Compute the binarized predictions
                self.label_binarizer.fit(train_pred.reshape(-1, 1), train_labels)
                train_bin = self.label_binarizer.predict(train_pred.reshape(-1, 1))
                test_bin = self.label_binarizer.predict(test_pred.reshape(-1, 1))

            # Create the scoresheets
            self.train_scores = Scores(y_true=train_labels, y_pred=train_pred, y_bin=train_bin)
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
        else:
            plt.show()
        plt.clf()

    def plot(self, results='test', curve='all', filename=None):
        r"""
        Plots the required curve. The filenames will be appended a `_PR.pdf` or `_Roc.pdf`.

        Parameters
        ----------
        curve : basestring, optional
            Can be one of 'all', 'pr' or 'roc'.
            Default is 'all'.
        filename : basestring, optional
            The name that will be given to the output plots. If none the plots are only shown on screen.
            Default is None.
        results : basestring
            A string indicating if the 'train' or 'test' plots should be provided.
        """
        if results == 'test':
            scores = self.test_scores
        else:
            scores = self.train_scores

        if curve == 'all' or curve == 'pr':
            precision, recall, _ = precision_recall_curve(scores.y_true, scores.y_pred)
            self._plot(results, precision, recall, 'Precision', 'Recall', 'PR', filename)

        if curve == 'all' or curve == 'roc':
            warnings.warn('ROC curves are not recommended in the case of class imbalance. '
                          'PR curves should be preferred.', Warning)
            fpr, tpr, thresholds = roc_curve(scores.y_true, scores.y_pred)
            self._plot(results, fpr, tpr, 'False positive rate', 'True positive rate', 'Roc', filename)

    def save(self, filename, results='test'):
        r"""
        Saves to a file the method name, parameters, if scores are for train or test data and all the scores.

        Parameters
        ----------
        results : basestring
            A string indicating if the 'train' or 'test' results should be stored.
        filename : basestring
            The name of the output file where the results will be stored.
        """
        try:
            f = open(filename, 'a+')
            f.write("Method: {}".format(self.method))
            f.write("\nParameters: ")
            for k, v in self.params.items():
                f.write(str(k) + ": " + str(v) + ", ")
            if results == 'test':
                f.write("\nTest scores: ")
                scores = self.test_scores
            else:
                f.write("\nTrain scores: ")
                scores = self.train_scores
            f.write("\n tn = {}, fp = {}, fn = {}, tp = {}".format(scores.tn, scores.fp, scores.fn, scores.tp))
            f.write("\n Precision = {} \n Recall = {}".format(scores.precision(), scores.recall()))
            f.write("\n Fallout = {} \n Miss = {}".format(scores.fallout(), scores.miss()))
            f.write("\n Accuracy = {} \n f_score = {}".format(scores.accuracy(), scores.f_score()))
            f.write("\n Average_prec = {}".format(average_precision_score(scores.y_true, scores.y_pred)))
            f.write("\n AUROC = {}".format(scores.auroc()))
            f.write("\n prec@1 = {}, prec@10 = {}, prec@100 = {}, prec@200 = {}, prec@500 = {}, "
                    "prec@1000 = {}, prec@10000 = {}".format(scores.precisionatk(1), scores.precisionatk(10),
                                                             scores.precisionatk(100), scores.precisionatk(200),
                                                             scores.precisionatk(500), scores.precisionatk(1000),
                                                             scores.precisionatk(10000)))
            f.write("\n\n")
            f.close()
        except IOError:
            print("Could not read file:", filename)

    def pretty_print(self, results='test'):
        r"""
        Prints to screen the method name, parameters, if scores are for train or test data and all the scores.

        results : basestring
            A string indicating if the 'train' or 'test' results should be plotted.
        """
        print("Method: {}".format(self.method))
        print("Parameters: ")
        print(self.params.items())
        if results == 'test':
            print("Test scores: ")
            scores = self.test_scores
        else:
            print("Train scores: ")
            scores = self.train_scores
        print(" tn = {} \n fp = {} \n fn = {} \n tp = {}".format(scores.tn, scores.fp, scores.fn, scores.tp))
        print(" Precision = {} \n Recall = {}".format(scores.precision(), scores.recall()))
        print(" Fallout = {} \n Miss = {}".format(scores.fallout(), scores.miss()))
        print(" AUROC = {} \n Accuracy = {} \n f_score = {}"
              .format(scores.auroc(), scores.accuracy(), scores.f_score()))
        print("")


class Scores(object):
    """
    Object that encapsulates the results (train or test) and exposes methods to compute different scores
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
        rel = 0
        for i in range(MAX):
            rel += self._sorted[i][0]
        return rel/k if k != 0 else float('NaN')

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
        Computes the F-score as the harmonic mean of precision adn recall:
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
        if np.sum(self.y_true) != len(self.y_true)/2.0:
            warnings.warn('AUROC is not recommended in the case of class imbalance. ', Warning)
        return roc_auc_score(self.y_true, self.y_pred)
