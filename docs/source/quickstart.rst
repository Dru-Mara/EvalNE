Quickstart
==========

As a command line tool
----------------------

The library takes as input an *.ini* configuration file. This file allows the user 
to specify the evaluation settings, from the methods and baselines to be evaluated
to the edge embedding methods, parameters to tune or scores to report.

An example `conf.ini` file is provided describing the available options
for each parameter. This file can be either modified to simulate different
evaluation settings or used as a template to generate other *.ini* files.

Another configuration example provided is `conf_node2vec.ini`. This file simulates 
the link prediction experiments of the paper "Scalable Feature Learning for 
Networks" by A. Grover and J. Leskovec.

Once the configuration is set, the evaluation can be run as indicated in the next
subsection.

**Running the conf examples**

In order to run the evaluations using the provided `conf.ini` and 
`conf_node2vec.ini` files, the following steps are necessary: 

1. Download and install the libraries/methods used in the examples:

  * OpenNE_
  * PRUNE_
  * Metapath2Vec_

2. Download the datasets used in the examples:

  * For `conf.ini`:

    * StudentDB_
    * Arxiv GR-QC_

  * For `conf_node2vec.ini`:

    * Facebook_ combined network
    * Arxiv Astro-Ph_
    * PPI_
    
3. Set the correct dataset paths in the INPATHS option of the corresponding *.ini* file. 
And the correct path for PRUNE under the METHODS_OTHER option. 

4. Run the evaluation:

    ::
    
        # For conf.ini run:
        python evalne ./examples/conf.ini
    
        # For conf_node2vec.ini run:
        python evalne ./examples/node2vec/conf_node2vec.ini

.. note::

    The networks provided as input to EvalNE are required to be in edgelist format.

.. _OpenNE: https://github.com/thunlp/OpenNE
.. _PRUNE: https://github.com/ntumslab/PRUNE
.. _Metapath2Vec: https://www.dropbox.com/s/w3wmo2ru9kpk39n/code_metapath2vec.zip?dl=0
.. _StudentDB: http://adrem.ua.ac.be/smurfig
.. _GR-QC: https://snap.stanford.edu/data/ca-GrQc.html
.. _Facebook: https://snap.stanford.edu/data/egonets-Facebook.html
.. _Astro-Ph: http://snap.stanford.edu/data/ca-AstroPh.html
.. _PPI: http://snap.stanford.edu/node2vec/Homo_sapiens.mat

As an API
---------

The library can be imported and used like any other Python module. Next we
present a very basic example, for more complete ones we refer the user to the
`examples/` folder.

::

    from evalne.evaluation import evaluator
    from evalne.preprocessing import preprocess as pp
    
    # Load and preprocess the network
    G = pp.load_graph('./evalne/tests/data/network.edgelist')
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
        nee.evaluate_ne_cmd(method_name=methods[i], command=command, 
                            edge_embedding_methods=edge_emb, input_delim=' ', emb_delim=' ')

    # Get output
    results = nee.get_results()
    for result in results:
        result.pretty_print()
    

Output
------

The library can provide two types of outputs, depending on the value of the SCORES option
of the configuration file. If the keyword 'all' is specified, all the available scores for 
every algorithm on each network and experiment repeat will be stored. These results will 
be written to files named `eval_output_rep_x.txt` where `x` is an integer corresponding 
to each repeat ID. These files will be stored in the corresponding output folders as
specified in the OUTPATHS option of the configuration file used.

Setting the SCORES option to `%(maximize)` will generate a tabular output and write it
to a file named `eval_output.txt`. This file can be located in the same path from where
the execution was run and will contain a table of Alg.\Network which will be populated
with the averaged results over all the experiment repeats. 

Additionally, if the option TRAINTEST_PATH contains a valid filename, EvalNE will create
a file with that name under each of the OUTPATHS provided. In each of these paths the
library will store the true and false train and test sets of edge. 

.. note::
    The tabular output is not available for mixes of directed and undirected networks.
    If this type of output is desired, all values of the option DIRECTED must be either
    True or False.


