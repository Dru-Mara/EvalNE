#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# This simple example is the one presented in the README.md file.

# NOTE: In order to run this experiment `as is` OpenNE is required.

from evalne.evaluation import evaluator
from evalne.preprocessing import preprocess as pp

# Load and preprocess the network
G = pp.load_graph('../evalne/tests/data/network.edgelist')
G, _ = pp.prep_graph(G)

# Create an evaluator and generate train/test edge split
nee = evaluator.Evaluator()
_ = nee.traintest_split.compute_splits(G)

# Set the baselines
methods = ['random_prediction', 'common_neighbours', 'jaccard_coefficient']

# Evaluate baselines
nee.evaluate_baseline(methods=methods)

# Set embedding methods from OpenNE
methods = ['node2vec', 'deepwalk', 'GraRep']
commands = [
    'python -m openne --method node2vec --graph-format edgelist --p 1 --q 1',
    'python -m openne --method deepWalk --graph-format edgelist --number-walks 40',
    'python -m openne --method grarep --graph-format edgelist --epochs 10']
edge_emb = ['average', 'hadamard']

# Evaluate embedding methods
for i in range(len(methods)):
    command = commands[i] + " --input {} --output {} --representation-size {}"
    nee.evaluate_cmd(method_name=methods[i], method_type='ne', command=command,
                     edge_embedding_methods=edge_emb, input_delim=' ', output_delim=' ')

# Get output
results = nee.get_results()
for result in results:
    result.pretty_print()
