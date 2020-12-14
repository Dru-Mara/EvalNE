#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# This file contains methods and classes that simplify the management and splitting of edges in sets of train and test
# or train and validation.
# TODO v0.4.0: Change naming from train_edges/test_edges to train_data/test_data.
# TODO v0.4.0: Change naming from train_E/train_E_false to train_pos/train_neg.

from __future__ import division

from abc import abstractmethod

import networkx as nx
import numpy as np

from evalne.utils import preprocess as pp
from evalne.utils import split_train_test as stt


class BaseEvalSplit(object):
    """
    Base class that provides a high level interface for managing/computing sets of train and test edges and non-edges
    for LP, SP and NR tasks. The class exposes the train edges and non-edges through the `train_edges` property and
    the test edges and non-edges through the `test_edges` property. Parameters used to compute these sets are also made
    available.
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
        self._nw_name = None
        self._split_id = None

    @property
    def train_edges(self):
        """The set of training node pairs."""
        return self._train_edges

    @property
    def test_edges(self):
        """The set of test node pairs."""
        return self._test_edges

    @property
    def train_labels(self):
        """A list of train node-pair labels. Labels can be either 0 or 1 and denote non-edges and edges,
        respectively (for SP they denote negative and positive links, respectively)."""
        return self._train_labels

    @property
    def test_labels(self):
        """A list of test node-pair labels. Labels can be either 0 or 1 and denote non-edges and edges,
        respectively (for SP they denote negative and positive links, respectively)."""
        return self._test_labels

    @property
    def TG(self):
        """A NetworkX graph or digraph to be used for training the embedding methods. For LP this should be the graph
        spanned by all train edges, for SP the graph spanned by the positive and negative train edges (with signs as
        edge weights) and for NR the entire graph being evaluated."""
        return self._TG

    @property
    def train_frac(self):
        """A float indicating the fraction of train edges out of all train and test edges."""
        return self._train_frac

    @property
    def split_alg(self):
        """A string indicating the algorithm used to split edges in train and test sets."""
        return self._split_alg

    @property
    def nw_name(self):
        """A string indicating the name of the dataset used to generate the sets of edges."""
        return self._nw_name

    @property
    def split_id(self):
        """An int used as an ID for this particular train/test split."""
        return self._split_id

    def _set_splits(self, train_E, train_E_false=None, test_E=None, test_E_false=None, directed=False, nw_name='test',
                    TG=None, split_id=0, split_alg='spanning_tree', verbose=False):
        """
        Sets the class attributes to the provided input values. The input train edges and non-edges as well as the
        test edges and non-edges are respectively joined to form the `train_edges` and `test_edges` class attributes.
        Train and test labels are also inferred from the input data.

        Parameters
        ----------
        train_E : set
            Set of train edges.
        train_E_false : set, optional
            Set of train non-edges. Default is None.
        test_E : set, optional
            Set of test edges. Default is None.
        test_E_false : set, optional
            Set of test non-edges. Default is None.
        directed : bool, optional
            True if the splits correspond to a directed graph, false otherwise. Default is False.
        nw_name : string, optional
            A string indicating the name of the dataset from which this split was generated.
            This is required in order to keep track of the evaluation results. Default is `test`.
        TG : graph, optional
            A NetworkX graph or digraph to be used for training the embedding methods. If None, the graph will be
            generated from the set of train edges. Default is None.
        split_id : int, optional
            An ID that identifies this particular train/test split. Default is 0.
        split_alg : string, optional
            A string indicating the algorithm used to generate the train/test splits. Options are `spanning_tree`,
            `random`, `naive`, `fast`, `timestamp` and `random_edge_sample`. Default is `spanning_tree`.
        verbose : bool, optional
            If True prints progress info. Default is False.

        Raises
        ------
        ValueError
            If the train edge set is not provided.
        """
        if len(train_E) != 0:
            if train_E_false is not None:
                # Stack the edges and non-edges together.
                self._train_edges = np.vstack((list(train_E), list(train_E_false)))

                # Create labels vector with 1s for edges and 0s for non-edges
                self._train_labels = np.hstack((np.ones(len(train_E)), np.zeros(len(train_E_false))))

            else:
                # Stack the edges and non-edges together.
                self._train_edges = np.array(list(train_E))

                # Create labels vector with 1s for edges and 0s for non-edges
                self._train_labels = np.ones(len(train_E))

            if test_E is not None:
                if test_E_false is not None:
                    # Stack the edges and non-edges together.
                    self._test_edges = np.vstack((list(test_E), list(test_E_false)))

                    # Create labels vector with 1s for edges and 0s for non-edges
                    self._test_labels = np.hstack((np.ones(len(test_E)), np.zeros(len(test_E_false))))

                else:
                    # We only have test edges (no test non-edges)
                    self._test_edges = np.array(list(test_E))

                    # Create labels vector with 1s for edges
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

            # Set class attributes to new values
            if test_E is not None:
                self._train_frac = np.around(len(train_E) / (len(train_E) + len(test_E)), 4)
            else:
                self._train_frac = 1
            self._split_alg = split_alg
            self._split_id = split_id
            self._nw_name = nw_name
        else:
            raise ValueError("Train edges are always required!")

        # Print the process
        if verbose:
            print("Edge splits computed using {} alg. ready.".format(self.split_alg))

    def get_parameters(self):
        """
        Returns the class properties except the sets of train and test node pairs, labels and train graph.

        Returns
        -------
        parameters : dict
            The parameters used when computing this split as a dictionary of parameters and values.
        """
        params = {"train_frac": self.train_frac, "split_alg": self.split_alg,
                  "nw_name": self._nw_name, "split_id": self.split_id}
        return params

    def get_data(self):
        """
        Returns the sets of train and test node pairs and label vectors.

        Returns
        -------
        train_edges : set
            Set of all train edges and non-edges.
        test_edges : set
            Set of all test edges and non-edges.
        train_labels : list
            A list of labels indicating if each train node-pair is an edge or non-edge (1 or 0).
        test_labels : list
            A list of labels indicating if each test node-pair is an edge or non-edge (1 or 0).
        """
        return self.train_edges, self.train_labels, self.test_edges, self.test_labels

    def save_tr_graph(self, output_path, delimiter, write_stats=False, write_weights=False, write_dir=True):
        """
        Saves the TG graph to a file.

        Parameters
        ----------
        output_path : file or string
            File or filename to write. If a file is provided, it must be opened in 'wb' mode.
        delimiter : string, optional
            The string used to separate values. Default is ','.
        write_stats : bool, optional
            Adds basic graph statistics to the file as a header or not. Default is True.
        write_weights : bool, optional
            If True data will be stored as weighted edgelist i.e. triplets (src, dst, weight), otherwise, as regular
            (src, dst) pairs. For unweighted graphs, setting this parameter to True will add weight 1 to all edges.
            Default is False.
        write_dir : bool, optional
            This parameter is only relevant for undirected graphs. If True, it forces the method to write both edge
            directions in the file i.e. (src, dst) and (dst, src). If False, only one direction is stored.
            Default is True.

        See also
        --------
        evalne.utils.preprocess.save_graph
        """
        pp.save_graph(self._TG, output_path=output_path, delimiter=delimiter, write_stats=write_stats,
                      write_weights=write_weights, write_dir=write_dir)

    def store_edgelists(self, train_path, test_path):
        """
        Writes the sets of train and test node pairs to files with the specified names.

        Parameters
        ----------
        train_path : string
           Indicates the path where the train data will be stored.
        test_path : string
           Indicates the path where the test data will be stored.

        See also
        --------
        evalne.utils.split_train_test.store_edgelists
        """
        stt.store_edgelists(train_path, test_path, self.train_edges, self.test_edges)


class NREvalSplit(BaseEvalSplit):
    """
    Class that provides a high level interface for managing/computing sets of train edges and non-edges
    for NR tasks. The class exposes the train edges and non-edges through the `train_edges` property. Test edges
    are not used for NR and therefore the `test_edges` property will be left empty. Parameters used to compute
    these sets are also made available.

    Notes
    -----
    In network reconstruction the aim is to asses how well an embedding method captures the structure of a given graph.
    The embedding methods are trained on a complete input graph. Hyperparameter tuning is performed directly on this
    graph (overfitting is, in this case, expected and desired). The embeddings obtained are used to perform link
    predictions and their quality is evaluated. Checking the link predictions for all node pairs is generally
    unfeasible, therefore a subset of all node pairs in the input graph are selected for evaluation.
    """

    def __init__(self):
        self._samp_frac = None
        super(NREvalSplit, self).__init__()

    @property
    def samp_frac(self):
        """A float indicating the fraction of node pairs out of all possible ones sampled for NR evaluation."""
        return self._samp_frac

    def set_splits(self, TG, train_E, train_E_false=None, samp_frac=None, directed=False, nw_name='test',
                   split_id=0, verbose=False):
        """
        Sets the class attributes to the provided input values. The input train edges and non-edges are joined to form
        the `train_edges` class attribute. Train labels are also inferred from the input data.

        Parameters
        ----------
        TG : graph
            A NetworkX graph or digraph, the complete network from which train_E and train_E_false were sampled.
        train_E : set
            Set of train edges.
        train_E_false : set, optional
            Set of train non-edges. Default is None.
        samp_frac : float, optional
            The fraction of node-pairs out of all possible ones sampled for NR evaluation. Default is None.
        directed : bool, optional
            True if the splits correspond to a directed graph, false otherwise. Default is False.
        nw_name : string, optional
            A string indicating the name of the dataset from which this split was generated.
            This is required in order to keep track of the evaluation results. Default is `test`.
        split_id : int, optional
            An ID that identifies this particular train/test split. Default is 0.
        verbose : bool, optional
            If True prints progress info. Default is False.

        Raises
        ------
        ValueError
            If the train edge set is not provided.
        """
        # Set the NR specific parameters
        self._samp_frac = samp_frac

        # Set the remaining parameters by calling the parent class private set method
        # For NR we do not have test data, so initialize these sets to None
        super(NREvalSplit, self)._set_splits(train_E=train_E, train_E_false=train_E_false, test_E=None,
                                             test_E_false=None, directed=directed, nw_name=nw_name,
                                             TG=TG, split_id=split_id, split_alg='random_edge_sample', verbose=verbose)

    def compute_splits(self, G, nw_name='test', samp_frac=0.01, split_id=0, verbose=False):
        """
        Computes sets of train edges and non-edges by randomly sampling elements from the adjacency matrix of G and
        initializes the class attributes.

        Parameters
        ----------
        G : graph
            A NetworkX graph or digraph to sample node pairs from.
        nw_name : string, optional
            A string indicating the name of the dataset from which this split was generated.
            This is required in order to keep track of the evaluation results. Default is 'test'.
        samp_frac : float, optional
            The fraction of node-pairs out of all possible ones to sample for NR evaluation. Default is 0.01 (1%).
        split_id : int, optional
            The id to be assigned to the train/test splits generated. Default is 0.
        verbose : bool, optional
            If True print progress info. Default is False.

        Returns
        -------
        train_E : set
            The set of train edges.
        train_false_E : set
            The set of train non-edges.

        Raises
        ------
        ValueError
            If the edge split algorithm is unknown.
        """
        # Sample the required number of node pairs from the graph
        train_E, train_E_false = stt.random_edge_sample(nx.adj_matrix(G), samp_frac, nx.is_directed(G))

        # Raise an error if no edges were selected while sampling matrix entries (both edges and non-edges are required)
        if len(train_E) == 0:
            raise ValueError("Sampling fraction {} on {} network is too low, no edges were selected.".format(samp_frac,
                                                                                                             nw_name))

        # Set class attributes to new values
        self.set_splits(TG=G, train_E=train_E, train_E_false=train_E_false, samp_frac=samp_frac,
                        directed=nx.is_directed(G), nw_name=nw_name, split_id=split_id, verbose=verbose)

        return train_E, train_E_false

    def get_parameters(self):
        """
        Returns the class properties except the sets of train and test node pairs, labels and train graph.

        Returns
        -------
        parameters : dict
            The parameters used when computing this split as a dictionary of parameters and values.
        """
        # Get the parameters from the parent class
        params = super(NREvalSplit, self).get_parameters()

        # Add the LP specific parameters
        params.update({"samp_frac": self._samp_frac})
        return params


class SPEvalSplit(BaseEvalSplit):
    """
    Class that provides a high level interface for managing/computing sets of train and test positive and negative edges
    for SP tasks. The class exposes the train positive and negative edges through the `train_edges` property and
    the test positive and negative edges through the `test_edges` property. Parameters used to compute these sets are
    also made available.

    Notes
    -----
    In sign prediction the aim is to predict the sign (positive or negative) of given edges. The existence of the edges
    is assumed (i.e. we do not predict the sign of unconnected node pairs). Therefore, sign prediction is also a binary
    classification task similar to link prediction where, instead of predicting the existence of edges or not, we
    predict the signs for edges we know exist. Unlike for link prediction, in this case we do not need to perform
    negative sampling, since we already have both classes (the positively and the negatively connected node pairs).
    """

    def __init__(self):
        super(SPEvalSplit, self).__init__()

    def set_splits(self, train_E, train_E_false=None, test_E=None, test_E_false=None, directed=False, nw_name='test',
                   TG=None, split_id=0, split_alg='spanning_tree', verbose=False):
        """
        Sets the class attributes to the provided input values. The input train positive and negative edges as well as
        the test positive and negative edges are respectively joined to form the `train_edges` and `test_edges` class
        attributes. Train and test labels (0 or 1 representing negative and positive edges, respectively) are also
        inferred from the input data.

        Parameters
        ----------
        train_E : set
            Set of positive train edges.
        train_E_false : set, optional
            Set of negative train edges. Default is None.
        test_E : set, optional
            Set of positive test edges. Default is None.
        test_E_false : set, optional
            Set of negative test edges. Default is None.
        directed : bool, optional
            True if the splits correspond to a directed graph, false otherwise. Default is False.
        nw_name : string, optional
            A string indicating the name of the dataset from which this split was generated.
            This is required in order to keep track of the evaluation results. Default is `test`.
        TG : graph, optional
            A NetworkX graph or digraph containing all the train edges (positive and negative). If None, the graph will
            be generated from the sets of positive and negative train edges. Default is None.
        split_id : int, optional
            An ID that identifies this particular train/test split. Default is 0.
        split_alg : string, optional
            A string indicating the algorithm used to generate the train/test splits. Options are `spanning_tree`,
            `random`, `naive`, `fast` and `timestamp`. Default is `spanning_tree`.
        verbose : bool, optional
            If True prints progress info. Default is False.

        Raises
        ------
        ValueError
            If the train edge set is not provided.
        """
        # Initialize the training graph
        if TG is None:
            if directed:
                TG = nx.DiGraph()
            else:
                TG = nx.Graph()
            TG.add_edges_from(train_E)
            TG.add_edges_from(train_E_false)

        # Set the parameters by calling the parent class private set method
        super(SPEvalSplit, self)._set_splits(train_E=train_E, train_E_false=train_E_false, test_E=test_E,
                                             test_E_false=test_E_false, directed=directed, nw_name=nw_name,
                                             TG=TG, split_id=split_id, split_alg=split_alg, verbose=verbose)

    def compute_splits(self, G, nw_name='test', train_frac=0.51, split_alg='spanning_tree', split_id=0, verbose=False):
        """
        Computes sets of train and test positive and negative edges according to the given input parameters and
        initializes the class attributes.

        Parameters
        ----------
        G : graph
            A NetworkX graph or digraph to compute the train test split from.
        nw_name : string, optional
            A string indicating the name of the dataset from which this split was generated.
            This is required in order to keep track of the evaluation results. Default is 'test'.
        train_frac : float, optional
            The proportion of train edges w.r.t. the total number of edges in the input graph (range (0.0, 1.0]).
            Default is 0.51.
        split_alg : string, optional
            A string indicating the algorithm to use for generating the train/test splits. Options are `spanning_tree`,
            `random`, `naive`, `fast` and `timestamp`. Default is `spanning_tree`.
        split_id : int, optional
            The id to be assigned to the train/test splits generated. Default is 0.
        verbose : bool, optional
            If True print progress info. Default is False.

        Returns
        -------
        train_E : set
            The set of train positive edges.
        train_false_E : set
            The set of train negative edges.
        test_E : set
            The set of test positive edges.
        test_false_E : set
            The set of test negative edges.

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
        elif split_alg == 'timestamp':
            train_E, test_E, _ = stt.timestamp_split(G, train_frac)
        else:
            raise ValueError('Split alg. {} unknown!'.format(split_alg))

        # Make sure the edges are numpy arrays
        train_E = np.array(list(train_E))
        test_E = np.array(list(test_E))

        # Get the labels of train and test
        a = nx.adj_matrix(G)
        tr_labels = np.ravel(a[train_E[:, 0], train_E[:, 1]])
        te_labels = np.ravel(a[test_E[:, 0], test_E[:, 1]])

        # Split train and test edges in those with positive and negative signs
        pos_tr_e = train_E[np.where(tr_labels == 1)[0], :]
        neg_tr_e = train_E[np.where(tr_labels == -1)[0], :]
        pos_te_e = test_E[np.where(te_labels == 1)[0], :]
        neg_te_e = test_E[np.where(te_labels == -1)[0], :]

        # Make a train graph with appropriate weights +1 / -1
        H = G.copy()
        H.remove_edges_from(test_E)

        # Set class attributes to new values
        self.set_splits(train_E=pos_tr_e, train_E_false=neg_tr_e, test_E=pos_te_e, test_E_false=neg_te_e,
                        directed=G.is_directed(), nw_name=nw_name, TG=H, split_id=split_id,
                        split_alg=split_alg, verbose=verbose)

        return pos_tr_e, neg_tr_e, pos_te_e, neg_te_e


class LPEvalSplit(BaseEvalSplit):
    """
    Class that provides a high level interface for managing/computing sets of train and test edges and non-edges
    for LP tasks. The class exposes the train edges and non-edges through the `train_edges` property and
    the test edges and non-edges through the `test_edges` property. Parameters used to compute these sets are
    also made available.

    Notes
    -----
    In link prediction the aim is to predict, given a set of node pairs, if they should be connected or not. This is
    generally solved as a binary classification task. For training the binary classifier, we sample a set of edges as
    well as a set of unconnected node pairs. We then compute the node-pair embeddings of this training data. We use
    the node-pair embeddings together with the corresponding labels (0 for non-edges and 1 for edges) to train the
    classifier. Finally, the performance is evaluated on the test data (the remaining edges not used in training plus
    another set of randomly selected non-edges).
    """

    def __init__(self):
        self._owa = None
        self._fe_ratio = None
        super(LPEvalSplit, self).__init__()

    @property
    def owa(self):
        """A bool parameter indicating if the non-edges have been generated using the OWA (otherwise CWA)."""
        return self._owa

    @property
    def fe_ratio(self):
        """A float indicating the ratio of non-edges to edges."""
        return self._fe_ratio

    def set_splits(self, train_E, train_E_false=None, test_E=None, test_E_false=None, directed=False, nw_name='test',
                   TG=None, split_id=0, split_alg='spanning_tree', owa=True, verbose=False):
        """
        Sets the class attributes to the provided input values. The input train edges and non-edges as well as the
        test edges and non-edges are respectively joined to form the `train_edges` and `test_edges` class attributes.
        Train and test labels are also inferred from the input data.

        Parameters
        ----------
        train_E : set
            Set of train edges.
        train_E_false : set, optional
            Set of train non-edges. Default is None.
        test_E : set, optional
            Set of test edges. Default is None.
        test_E_false : set, optional
            Set of test non-edges. Default is None.
        directed : bool, optional
            True if the splits correspond to a directed graph, false otherwise. Default is False.
        nw_name : string, optional
            A string indicating the name of the dataset from which this split was generated.
            This is required in order to keep track of the evaluation results. Default is `test`.
        TG : graph, optional
            A NetworkX graph or digraph containing all the train edges. If None, the graph will be generated from the
            set of train edges. Default is None.
        split_id : int, optional
            An ID that identifies this particular train/test split. Default is 0.
        split_alg : string, optional
            A string indicating the algorithm used to generate the train/test splits. Options are `spanning_tree`,
            `random`, `naive`, `fast` and `timestamp`. Default is `spanning_tree`.
        owa : bool, optional
            Encodes the belief that the network respects or not the open world assumption. Default is True.
            If owa=True, train non-edges are sampled from the train graph only and can overlap with test edges.
            If owa=False, train non-edges are sampled from the full graph and cannot overlap with test edges.
        verbose : bool, optional
            If True prints progress info. Default is False.

        Raises
        ------
        ValueError
            If the train edge set is not provided.
        """
        # Set the LP specific parameters
        self._owa = owa
        if train_E_false is not None:
            self._fe_ratio = np.around(len(train_E_false) / len(train_E), 4)
        else:
            self._fe_ratio = 1

        # Set the remaining parameters by calling the parent class private set method
        super(LPEvalSplit, self)._set_splits(train_E=train_E, train_E_false=train_E_false, test_E=test_E,
                                             test_E_false=test_E_false, directed=directed, nw_name=nw_name,
                                             TG=TG, split_id=split_id, split_alg=split_alg, verbose=verbose)

    def compute_splits(self, G, nw_name='test', train_frac=0.51, split_alg='spanning_tree', owa=True, fe_ratio=1,
                       split_id=0, verbose=False):
        """
        Computes sets of train and test edges and non-edges according to the given input parameters and initializes
        the class attributes.

        Parameters
        ----------
        G : graph
            A NetworkX graph or digraph to compute the train test split from.
        nw_name : string, optional
            A string indicating the name of the dataset from which this split was generated.
            This is required in order to keep track of the evaluation results. Default is 'test'.
        train_frac : float, optional
            The proportion of train edges w.r.t. the total number of edges in the input graph (range (0.0, 1.0]).
            Default is 0.51.
        split_alg : string, optional
            A string indicating the algorithm to use for generating the train/test splits. Options are `spanning_tree`,
            `random`, `naive`, `fast` and `timestamp`. Default is `spanning_tree`.
        owa : bool, optional
            Encodes the belief that the network should respect or not the open world assumption. Default is True.
            If owa=True, train non-edges are sampled from the train graph only and can overlap with test edges.
            If owa=False, train non-edges are sampled from the full graph and cannot overlap with test edges.
        fe_ratio : float, optional
            The ratio of non-edges to edges to sample. For fr_ratio > 0 and < 1 less non-edges than edges will be
            generated. For fe_edges > 1 more non-edges than edges will be generated. Default 1, same amounts.
        split_id : int, optional
            The id to be assigned to the train/test splits generated. Default is 0.
        verbose : bool, optional
            If True print progress info. Default is False.

        Returns
        -------
        train_E : set
            The set of train edges
        train_false_E : set
            The set of train non-edges
        test_E : set
            The set of test edges
        test_false_E : set
            The set of test non-edges

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
        elif split_alg == 'timestamp':
            train_E, test_E, G = stt.timestamp_split(G, train_frac)
            train_E = set(zip(train_E[:, 0], train_E[:, 1]))
            test_E = set(zip(test_E[:, 0], test_E[:, 1]))
        else:
            raise ValueError('Split alg. {} unknown!'.format(split_alg))

        # Compute non-edges
        if split_alg != 'fast':
            num_fe_train = len(train_E) * fe_ratio
            num_fe_test = len(test_E) * fe_ratio
            if owa:
                train_E_false, test_E_false = stt.generate_false_edges_owa(G, train_E, test_E,
                                                                           num_fe_train, num_fe_test)
            else:
                train_E_false, test_E_false = stt.generate_false_edges_cwa(G, train_E, test_E,
                                                                           num_fe_train, num_fe_test)

        # Set class attributes to new values
        self.set_splits(train_E, train_E_false, test_E, test_E_false, directed=G.is_directed(), nw_name=nw_name,
                        split_id=split_id, split_alg=split_alg, owa=owa, verbose=verbose)

        return train_E, train_E_false, test_E, test_E_false

    def get_parameters(self):
        """
        Returns the class properties except the sets of train and test node pairs, labels and train graph.

        Returns
        -------
        parameters : dict
            The parameters used when computing this split as a dictionary of parameters and values.
        """
        # Get the parameters from the parent class
        params = super(LPEvalSplit, self).get_parameters()

        # Add the LP specific parameters
        params.update({"owa": self._owa, "fe_ratio": self._fe_ratio})
        return params


class EvalSplit(LPEvalSplit):
    """
    Deprecated and will be removed in v0.4.0. Use LPEvalSplit instead.
    """

    def __init__(self):
        super(LPEvalSplit, self).__init__()

    def read_splits(self, filename, split_id, directed=False, nw_name='test', verbose=False):
        """
        Reads the train and test edges and non-edges from files and initializes the class attributes.

        Parameters
        ----------
        filename : string
            The filename shared by all edge splits as given by the 'store_train_test_splits' method
        split_id : int
            The ID of the edge splits to read. As provided by the 'store_train_test_splits' method
        directed : bool, optional
            True if the splits correspond to a directed graph, false otherwise. Default is False.
        nw_name : string, optional
            A string indicating the name of the dataset from which this split was generated.
            This is required in order to keep track of the evaluation results. Default is `test`.
        verbose : bool, optional
            If True print progress info. Default is False.

        See also
        --------
        evalne.utils.preprocess.read_train_test :
            The low level function used for reading the sets of edges and non-edges.
        evalne.utils.split_train_test.store_train_test_splits :
            The files in the provided input path are expected to follow the naming convention of this function.
        """
        # Read edge sets from file
        train_E, train_E_false, test_E, test_E_false = pp.read_train_test(filename, split_id)

        # Set class attributes to new values
        self.set_splits(train_E, train_E_false, test_E, test_E_false, directed=directed, nw_name=nw_name,
                        split_id=split_id, verbose=verbose)
