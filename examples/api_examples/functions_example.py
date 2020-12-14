#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# Note: This is just an example of how to use some of the low level functions in EvalNE.
# If possible the Evaluator and Split classes should be used as they simplify the pipeline.
# Diagram of the evaluation:
# preprocess_data -> split_train_test -> compute_node/edge_emb -> predict_edges -> evaluate_accuracy
# preprocess_data -> split_train_test -> link_prediction -> evaluate_accuracy

import os
import random
from time import time

import networkx as nx
import numpy as np

from evalne.evaluation import score
from evalne.methods import similarity as sim
from evalne.utils import preprocess as pp
from evalne.utils import split_train_test as stt

###########################
#       Variables
###########################


dataset_path = "../../evalne/tests/data/network.edgelist"
output_path = "./output/"
directed = False
random.seed(99)

###########################
#          Main
###########################

# Time count
start = time()

# Create folders for the results if these do not exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

traintest_path = os.path.join(output_path, "lp_train_test_splits")
if not os.path.exists(traintest_path):
    os.makedirs(traintest_path)

# ---------------
# Preprocess data
# ---------------

# Load the data as a directed graph
G = pp.load_graph(dataset_path, delimiter=",", comments='#', directed=directed)

# Get some graph statistics
pp.get_stats(G)

# Or store them to a file
pp.get_stats(G, os.path.join(output_path, "stats.txt"))

# Preprocess the graph
SG, ids = pp.prep_graph(G, relabel=True, del_self_loops=True)

# Get non-edges so that the reversed edge exists in the graph
if directed:
    redges = pp.get_redges_false(SG, output_path=os.path.join(output_path, "redges.csv"))

# Store the graph to a file
pp.save_graph(SG, output_path=os.path.join(output_path, "network_prep.edgelist"), delimiter=',', write_stats=True)

# ----------------
# Split train test
# ----------------

# Compute train/test splits and false edges in parallel
stt.compute_splits_parallel(SG, os.path.join(traintest_path, "network_prep_51"), owa=True,
                            train_frac=0.51, num_fe_train=None, num_fe_test=None, num_splits=5)

# The overlap between the 5 generated sets can be easily checked
print("Overlap check for train sets: ")
stt.check_overlap(filename=os.path.join(traintest_path, "network_prep_51", "trE"), num_sets=5)
print("Overlap check for test sets: ")
stt.check_overlap(filename=os.path.join(traintest_path, "network_prep_51", "teE"), num_sets=5)

# The same computations can be performed for the sets of non-edges
# print "Overlap check for negative train sets: "
# stt.check_overlap(filename=output_path + "lp_train_test_splits/network_prep_51_negTrE", num_sets=5)
# print "Overlap check for negative test sets: "
# stt.check_overlap(filename=output_path + "lp_train_test_splits/network_prep_51_negTeE", num_sets=5)

# Alternatively, train/test splits can be computed one at a time
train_E, test_E = stt.split_train_test(SG, train_frac=0.50)

# Compute set of false edges
# train_E_false, test_E_false = stt.generate_false_edges_owa(SG, train_E=train_E, test_E=test_E, num_fe_train=None,
#                                                            num_fe_test=None)
train_E_false, test_E_false = stt.generate_false_edges_cwa(SG, train_E=train_E, test_E=test_E, num_fe_train=None,
                                                           num_fe_test=None)

# Store the computed edge sets to a file
filenames = stt.store_train_test_splits(os.path.join(output_path, "lp_train_test_splits", "network_prep_51"),
                                        train_E=train_E, train_E_false=train_E_false, test_E=test_E,
                                        test_E_false=test_E_false, split_id=0)

# -------------------------------------------
# Link prediction (LP) using baseline methods
# -------------------------------------------

# Create a graph using only the train edges
if directed:
    TG = nx.DiGraph()
else:
    TG = nx.Graph()
TG.add_edges_from(train_E)

# Stack the true and false edges together.
train_edges = np.vstack((list(train_E), list(train_E_false)))
test_edges = np.vstack((list(test_E), list(test_E_false)))

# Create labels vectors with 1s for true edges and 0s for false edges
train_labels = np.hstack((np.ones(len(train_E)), np.zeros(len(train_E_false))))
test_labels = np.hstack((np.ones(len(test_E)), np.zeros(len(test_E_false))))

# Run common neighbours and obtain the probability of links
# Other methods are: jaccard_coefficient, adamic_adar_index, preferential_attachment, resource_allocation_index
train_pred = np.array(sim.common_neighbours(TG, ebunch=train_edges, neighbourhood='in'))
test_pred = np.array(sim.common_neighbours(TG, ebunch=test_edges, neighbourhood='in'))

# -------------------------------
# Evaluate LP using Results class
# -------------------------------
# The results class automatically binarizes the method predictions and provides useful functions for plotting and
# outputting the method results.

# Instantiate a Results class
results = score.Results(method='CN', params={}, train_pred=train_pred, train_labels=train_labels,
                        test_pred=test_pred, test_labels=test_labels)

# Compute auroc
train_auroc = results.train_scores.auroc()
test_auroc = results.test_scores.auroc()

# Visualize the results
print("Train AUROC {}: {}".format("common neighbours", train_auroc))
print("Test AUROC {}: {}".format("common neighbours", test_auroc))

# Compute precision recall curves
# Options are: pr, roc, all
results.plot(filename=os.path.join(output_path, 'CN_train'), results='train', curve='all')
results.plot(filename=os.path.join(output_path, 'CN_test'), results='test', curve='all')

# -----------------------------
# Evaluate LP using Score class
# -----------------------------
# Alternatively, the user can directly use the Scores class. This class acts as an interface providing different scores.
# The Scores class requires the user to provide binary representations of the predictions in the initialization.

# Binarize predictions using a simple threshold
train_bin = np.where(train_pred >= 0.5, 1, 0)
test_bin = np.where(test_pred >= 0.5, 1, 0)

# Compute the area under the AUC curve
train_scores = score.Scores(y_true=train_labels, y_pred=train_pred, y_bin=train_bin)
test_scores = score.Scores(y_true=test_labels, y_pred=test_pred, y_bin=test_bin)

# Compute auroc
train_auroc = train_scores.auroc()
test_auroc = test_scores.auroc()

# Visualize the results
print("Train AUROC {}: {}".format("common neighbours", train_auroc))
print("Test AUROC {}: {}".format("common neighbours", test_auroc))

# Get the execution time
end = time()-start
print("Processed in: {}".format(end))
