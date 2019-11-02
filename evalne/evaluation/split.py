#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

from __future__ import division

import networkx as nx
import numpy as np

from evalne.utils import preprocess as pp
from evalne.utils import split_train_test as stt


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
        self._split_alg = None
        self._owa = None
        self._fe_ratio = None
        self._nw_name = None
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
    def split_alg(self):
        """Returns a param. indicating the alg. used to perform the train/test split (spanning_tree, random, naive)"""
        return self._split_alg

    @property
    def owa(self):
        """Returns a parameter indicating if the false edges have been generated using the OWA (otherwise CWA)."""
        return self._owa

    @property
    def fe_ratio(self):
        """Returns the ratio of false to true edges in this split."""
        return self._fe_ratio

    @property
    def nw_name(self):
        """Returns the name of the dataset form which this split was generated."""
        return self._nw_name

    @property
    def split_id(self):
        """Returns a ID that identifies this particular split."""
        return self._split_id

    def set_splits(self, train_E, train_E_false=None, test_E=None, test_E_false=None, directed=False, nw_name='test',
                   TG=None, split_id=0, split_alg='spanning_tree', owa=True, verbose=False):
        """
        This method allows the user to set the train graph and train/test true and false edge sets manually.
        The test edges as well as false train and test edges can be empty.

        Parameters
        ----------
        train_E : set
            Set of train edges
        train_E_false : set, optional
            Set of train non-edges. Default is None.
        test_E : set, optional
            Set of test edges. Default is None, in this case will be initialized to empty list.
        test_E_false : set, optional
            Set of test non-edges. Default is None, in this case will be initialized to empty list.
        directed : bool, optional
            True if the splits correspond to a directed graph, false otherwise. Default is False.
        nw_name : basestring, optional
            A string indicating the name of the dataset from which this split was generated. Default is `test`.
            This is required in order to keep track of the evaluation results.
        TG : nx.Graph, optional
            A train graph containing all the train edges or being a superset of them. If not provided will be
            computed from the train edges. Default is None.
        split_id : int, optional
            An ID that identifies this particular train/test split. Default is 0.
        split_alg : basestring, optional
            Indicates the algorithm used to generate the train/test splits. Options are method based on spanning tree
            (`spanning_tree`), random edge split (`random`), naive removal and connectedness check (`naive`) and
            fast BFS spanning tree (`fast`). Default is `spanning_tree`.
        owa : bool, optional
            Encodes the belief that the network respects or not the open world assumption. Default is True.
        verbose : bool, optional
            If True print progress info. Default is False.

        Raises
        ------
        ValueError
            If the train edge set is not provided.
        """
        if len(train_E) != 0:
            if train_E_false is not None:
                # Stack the true and false edges together.
                self._train_edges = np.vstack((list(train_E), list(train_E_false)))

                # Create labels vectors with 1s for true edges and 0s for false edges
                self._train_labels = np.hstack((np.ones(len(train_E)), np.zeros(len(train_E_false))))

            else:
                # Stack the true and false edges together.
                self._train_edges = np.array(list(train_E))

                # Create labels vectors with 1s for true edges and 0s for false edges
                self._train_labels = np.ones(len(train_E))

            if test_E is not None:
                if test_E_false is not None:
                    # Stack the true and false edges together.
                    self._test_edges = np.vstack((list(test_E), list(test_E_false)))

                    # Create labels vectors with 1s for true edges and 0s for false edges
                    self._test_labels = np.hstack((np.ones(len(test_E)), np.zeros(len(test_E_false))))

                else:
                    # We only have true test edges
                    self._test_edges = np.array(list(test_E))

                    # Create labels vectors with 1s for true edges
                    self._test_labels = np.ones(len(test_E))
            else:
                self._test_edges = []
                self._test_labels = []

            # Initialize the training graph
            if TG is None:
                if directed:
                    self._TG = nx.DiGraph()
                else:
                    self._TG = nx.Graph()
                self._TG.add_edges_from(train_E)
            else:
                self._TG = TG.copy()

            # Fill the object parameters
            if test_E is not None:
                self._train_frac = len(train_E) / (len(train_E) + len(test_E))
            else:
                self._train_frac = 1
            self._split_alg = split_alg
            self._owa = owa
            if train_E_false is not None:
                self._fe_ratio = len(train_E_false) / len(train_E)
            else:
                self._fe_ratio = 1
            self._split_id = split_id
            self._nw_name = nw_name
        else:
            raise ValueError("Train edges are always required!")

        # Print the process
        if verbose:
            print("Edge splits computed using {} alg. ready.".format(self.split_alg))

    def read_splits(self, filename, split_id, directed=False, nw_name='test', verbose=False):
        """
        Reads true and false train and test edge splits from file.

        Parameters
        ----------
        filename : string
            The filename shared by all edge splits as given by the 'store_train_test_splits' method
        split_id : int
            The ID of the edge splits to read. As provided by the 'store_train_test_splits' method
        directed : bool, optional
            True if the splits correspond to a directed graph, false otherwise. Default is False.
        nw_name : basestring, optional
            A string indicating the name of the dataset from which this split was generated.
            This is required in order to keep track of the evaluation results in a Scoresheet object. Default is 'test'.
        verbose : bool, optional
            If True print progress info. Default is False.
        """
        # Read edge sets from file
        train_E, train_E_false, test_E, test_E_false = pp.read_train_test(filename, split_id)

        # Set edge sets to new values
        self.set_splits(train_E, train_E_false, test_E, test_E_false, directed=directed, nw_name=nw_name,
                        split_id=split_id, verbose=verbose)

    def compute_splits(self, G, nw_name='test', train_frac=0.51, split_alg='spanning_tree', owa=True, fe_ratio=1,
                       split_id=0, verbose=False):
        """
        Computes true and false train and test edge splits according to the given parameters.
        The sets of edges computed are both stored as properties of this object and returned from the method.

        Parameters
        ----------
        G : graph
            A NetworkX graph
        nw_name : basestring, optional
            A string indicating the name of the dataset from which this split was generated.
            This is required in order to keep track of the evaluation results. Default is 'test'.
        train_frac : float, optional
            The relative size (in (0.0, 1.0]) of the train set with respect to the total number of edges in the graph.
            Default is 0.51.
        split_alg : basestring, optional
            Indicates the algorithm used to generate the train/test splits. Options are method based on spanning tree,
            random edge split and naive removal and connectedness evaluation. Default is 'spanning_tree'.
        owa : bool, optional
            Encodes the belief that the network respects or not the open world assumption. Default is True.
            If OWA=True, false train edges can be true test edges. False edges sampled from train graph.
            If OWA=False, closed world is assumed so false train edges are known to be false (not in G)
        fe_ratio : float, optional
            The ratio of false to true edge to generate. Default is 1, same number as true edges.
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

        Raises
        ------
        ValueError
            If the edge split algorithm is unknown.
        """
        # Compute train/test split
        if split_alg == 'random':
            tr_E, te_E = stt.rand_split_train_test(G, train_frac)
            train_E, test_E, G, mp = pp.relabel_nodes(tr_E, te_E, G.is_directed())
        elif split_alg == 'naive':
            train_E, test_E = stt.naive_split_train_test(G, train_frac)
        elif split_alg == 'spanning_tree':
            train_E, test_E = stt.split_train_test(G, train_frac)
        elif split_alg == 'fast':
            train_E, test_E = stt.quick_split(G, train_frac)
            train_E_false, test_E_false = stt.quick_nonedges(G, train_frac, fe_ratio)
        else:
            raise ValueError('Split alg. {} unknown!'.format(split_alg))

        # Compute false edges
        if split_alg != 'fast':
            num_fe_train = len(train_E) * fe_ratio
            num_fe_test = len(test_E) * fe_ratio
            if owa:
                train_E_false, test_E_false = stt.generate_false_edges_owa(G, train_E, test_E,
                                                                           num_fe_train, num_fe_test)
            else:
                train_E_false, test_E_false = stt.generate_false_edges_cwa(G, train_E, test_E,
                                                                           num_fe_train, num_fe_test)

        # Set edge sets to new values
        self.set_splits(train_E, train_E_false, test_E, test_E_false, directed=G.is_directed(), nw_name=nw_name,
                        split_id=split_id, split_alg=split_alg, owa=owa, verbose=verbose)

        return train_E, train_E_false, test_E, test_E_false

    def get_parameters(self):
        """
        Returns the split parameters.

        Returns
        -------
        parameters : dict
            The split parameters as a dictionary of parameter : value
        """
        params = {"train_frac": self.train_frac, "split_alg": self.split_alg, "owa": self.owa,
                  "fe_ratio": self.fe_ratio, "nw_name": self._nw_name, "split_id": self.split_id}
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
