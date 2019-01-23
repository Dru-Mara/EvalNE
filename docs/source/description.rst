Features
========

For Methodologists
------------------

A command line interface in combination with a configuration file allow the user
to evaluate any publicly available implementation of a NE method. These
implementations can be obtained from libraries such as 
OpenNE_ or GEM_ as well as directly from the web pages of the authors e.g. CNE_,
PRUNE_. 

.. _OpenNE: https://github.com/thunlp/OpenNE
.. _GEM: https://github.com/palash1992/GEM
.. _CNE: https://bitbucket.org/ghentdatascience/cne-public/src/master/
.. _PRUNE: https://github.com/ntumslab/PRUNE

EvalNE also includes the following LP heuristics for both directed and
undirected networks (in and out node neighbourhoods), which can be used as
baselines:

* Random Prediction
* Common Neighbours
* Jaccard Coefficient
* Adamic Adar Index
* Preferential Attachment
* Resource Allocation Index

For Practitioners
-----------------

When used as an API, EvalNE provides functions to:

* Load and preprocess graphs
* Obtain general graph statistics
* Compute train/test/validation splits
* Generate false edges
* Evaluate link prediction from: 
    * Node Embeddings
    * Edge Embeddings
    * Similarity scores (e.g. the ones given by LP heuristics)
* Provides functions that compute edge embeddings from node feature vectors
    * Average
    * Hadamard
    * Weighted L1
    * Weighted L2
* Any sklearn binary classifier can be used as a LP algorithm
* Implements several accuracy metrics.
* Includes parameter tuning subroutines
