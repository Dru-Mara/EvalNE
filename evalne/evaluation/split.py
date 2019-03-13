#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

from __future__ import division

import networkx as nx
import numpy as np

from evalne.preprocessing import preprocess as pp
from evalne.preprocessing import split_train_test as stt


class EvalSplit(object):
    r"""
    Object that encapsulates properties related to train/test splits and
    exposes functions for managing these splits. Also can provide a training graph spanned by the training edges.
    """

    def __init__(self):
        self._train_edges = None
        self._test_edges = None
        self._train_labels = None
        self._test_labels = None
        self._TG = None
        # Data related statistics
        self._train_frac = None
        self._fast_split = None
        self._owa = None
        self._num_fe_train = None
        self._num_fe_test = None
        self._split_id = None

    @property
    def train_edges(self):
        """Returns the set of training edges in this split."""
        return self._train_edges

    @property
    def test_edges(self):
        """Returns the set of training edges in this split."""
        return self._test_edges

    @property
    def train_labels(self):
        """Returns the set of training edges in this split."""
        return self._train_labels

    @property
    def test_labels(self):
        """Returns the set of training edges in this split."""
        return self._test_labels

    @property
    def TG(self):
        """Returns a training graph containing only the training edges in this split."""
        return self._TG

    @property
    def train_frac(self):
        """Returns a float indicating the fraction of train edges in the split."""
        return self._train_frac

    @property
    def fast_split(self):
        """Returns a parameter indicating if the split was performed using broder's alg. (otherwise naive approach)."""
        return self._fast_split

    @property
    def owa(self):
        """Returns a parameter indicating if the false edges have been generated using the OWA (otherwise CWA)."""
        return self._owa

    @property
    def num_fe_train(self):
        """Returns the number of false train edges in this split."""
        return self._num_fe_train

    @property
    def num_fe_test(self):
        """Returns the number of false test edges in this split."""
        return self._num_fe_test

    @property
    def split_id(self):
        """Returns a ID that identifies this particular split."""
        return self._split_id

    def set_splits(self, train_E, train_E_false, test_E, test_E_false, directed, split_id=0, verbose=False):
        """
        This method allows the user to set the train/test true and false edge sets manually.
        All sets are required as well as a parameter indicating if the graph is directed or not.
        The sets of false edges can be empty.

        Parameters
        ----------
        train_E : set
            Set of train edges
        train_E_false : set
            Set of train non-edges
        test_E : set
            Set of test edges
        test_E_false : set
            Set of test non-edges
        directed : bool
            True if the splits correspond to a directed graph, false otherwise
        split_id : int, optional
            An ID that identifies this particular train/test split.
        verbose : bool, optional
            If True print progress info. Default is False.
        """
        if train_E is not None and train_E_false is not None \
                and test_E is not None and test_E_false is not None:

            if len(train_E_false) != 0:
                # Stack the true and false edges together.
                self._train_edges = np.vstack((list(train_E), list(train_E_false)))

                # Create labels vectors with 1s for true edges and 0s for false edges
                self._train_labels = np.hstack((np.ones(len(train_E)), np.zeros(len(train_E_false))))

            else:
                # Stack the true and false edges together.
                self._train_edges = np.array(list(train_E))

                # Create labels vectors with 1s for true edges and 0s for false edges
                self._train_labels = np.ones(len(train_E))

            if len(test_E_false) != 0:
                # Stack the true and false edges together.
                self._test_edges = np.vstack((list(test_E), list(test_E_false)))

                # Create labels vectors with 1s for true edges and 0s for false edges
                self._test_labels = np.hstack((np.ones(len(test_E)), np.zeros(len(test_E_false))))

            else:
                # Stack the true and false edges together.
                self._test_edges = np.array(list(test_E))

                # Create labels vectors with 1s for true edges and 0s for false edges
                self._test_labels = np.ones(len(test_E))

            # Initialize the training graph
            if directed:
                TG = nx.DiGraph()
            else:
                TG = nx.Graph()
            TG.add_edges_from(train_E)
            self._TG = TG

            # Fill data related parameters
            self._train_frac = len(train_E) / (len(train_E) + len(test_E))
            self._num_fe_train = len(train_E_false)
            self._num_fe_test = len(test_E_false)
            self._split_id = split_id
        else:
            raise ValueError("All sets of edges are required!")

        # Print the process
        if verbose:
            print("Train/test splits ready.")

    def read_splits(self, filename, split_id, directed, verbose=False):
        """
        Reads true and false train and test edge splits from file.

        Parameters
        ----------
        filename : string
            The filename shared by all edge splits as given by the 'store_train_test_splits' method
        split_id : int
            The ID of the edge splits to read. As provided by the 'store_train_test_splits' method
        directed : bool
            True if the splits correspond to a directed graph, false otherwise
        verbose : bool, optional
            If True print progress info. Default is False.
        """
        # Read edge sets from file
        train_E, train_E_false, test_E, test_E_false = pp.read_train_test(filename, split_id)

        # Reset some unknown parameters of the split
        self._fast_split = None
        self._owa = None

        # Set edge sets to new values
        self.set_splits(train_E, train_E_false, test_E, test_E_false, directed, split_id, verbose)

    def compute_splits(self, G, train_frac=0.51, fast_split=True, owa=True, num_fe_train=None, num_fe_test=None,
                       split_id=0, verbose=False):
        """
        Computes true and false train and test edge splits according to the given parameters.
        The sets of edges computed are both stored as properties of this object and returned from the method.

        Parameters
        ----------
        G : graph
            A NetworkX graph
        train_frac : float, optional
            The relative size (in (0.0, 1.0]) of the train set with respect to the total number of edges in the graph.
            Default is 0.51.
        fast_split : bool, optional
            If true the spanning tree split is used, else the naive train test edge split is used. Default is True.
        owa : bool, optional
            Encodes the belief that the network respects or not the open world assumption. Default is True.
            If OWA=True, false train edges can be true test edges. False edges sampled from train graph.
            If OWA=False, closed world is assumed so false train edges are known to be false (not in G)
        num_fe_train : int, optional
            The number of train false edges to generate. Default is same number as true train edges.
        num_fe_test : int, optional
            The number of test false edges to generate. Default is same number as true test edges.
        split_id : int, optional
            The id to be assigned to the train/test splits generated. Default is 0.
        verbose : bool, optional
            If True print progress info. Default is False.

        Returns
        -------
        train_E : set
            The set of train edges
        train_false_E : set
            The set of false train edges
        test_E : set
            The set of test edges
        test_false_E : set
            The set of false test edges
        """
        # Compute train/test split
        if fast_split:
            train_E, test_E = stt.split_train_test(G, train_frac)
        else:
            train_E, test_E = stt.naive_split_train_test(G, train_frac)

        # Compute false edges
        if owa:
            train_E_false, test_E_false = stt.generate_false_edges_owa(G, train_E, test_E,
                                                                       num_fe_train, num_fe_test)
        else:
            train_E_false, test_E_false = stt.generate_false_edges_cwa(G, train_E, test_E,
                                                                       num_fe_train, num_fe_test)

        # Initialize some parameters of the split
        self._fast_split = fast_split
        self._owa = owa

        # Set edge sets to new values
        self.set_splits(train_E, train_E_false, test_E, test_E_false, G.is_directed(), split_id, verbose)

        return train_E, train_E_false, test_E, test_E_false

    def get_parameters(self):
        """
        Returns the split parameters.

        Returns
        -------
        parameters : dict
            The split parameters as a dictionary of parameter : value
        """
        params = {"train_frac": self.train_frac, "fast_split": self.fast_split, "owa": self.owa,
                  "num_fe_train": self.num_fe_train, "num_fe_test": self.num_fe_test, "split_id": self.split_id}
        return params

    def get_data(self):
        """
        Returns the sets of train and test edges (true and false together) and the associated label vectors.

        Returns
        -------
        train_edges : set
            Set of all true and false train edges.
        test_edges : set
            Set of all true and false test edges.
        train_labels : set
            Set of labels indicating if train edges are true or false (1 or 0).
        test_labels : set
            Set of labels indicating if test edges are true or false (1 or 0).
        """
        return self.train_edges, self.train_labels, self.test_edges, self.test_labels

    def save_tr_graph(self, output_path, delimiter, write_stats=False, write_weights=False, write_dir=True):
        """
        Saves the graph to a file.

        Parameters
        ----------
        output_path : file or string
            File or filename to write. If a file is provided, it must be opened in 'wb' mode.
        delimiter : string, optional
            The string used to separate values. Default is .
        write_stats : bool, optional
            Sets if graph statistics should be added to the edgelist or not. Default is False.
        write_weights : bool, optional
            If True data will be stored as weighted edgelist (e.g. triplets src, dst, weight) otherwise as normal
            edgelist. If the graph edges have no weight attribute and this parameter is set to True,
            a weight of 1 will be assigned to each edge. Default is False.
        write_dir : bool, optional
            This option is only relevant for undirected graphs. If False, the train graph will be stored with a single
            direction of the edges. If True, both directions of edges will be stored. Default is True.
        """
        pp.save_graph(self._TG, output_path=output_path, delimiter=delimiter, write_stats=write_stats,
                      write_weights=write_weights, write_dir=write_dir)

    def store_edgelists(self, train_path, test_path):
        r"""
        Writes the train and test edgelists to files with the specified names.

        Parameters
        ----------
        train_path : string
           Indicates the path where the train data will be stored.
        test_path : string
           Indicates the path where the test data will be stored.
        """
        stt.store_edgelists(train_path, test_path, self.train_edges, self.test_edges)
