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

Additional configuration (*.ini*) files are provided replicating the experimental 
sections of different papers in the NE literature. These can be found in different
folders under `examples/`. One such configuration file is 
`examples/node2vec/conf_node2vec.ini`. This file simulates the link prediction 
experiments of the paper "Scalable Feature Learning for Networks" by A. Grover 
and J. Leskovec.

Once the configuration is set, the evaluation can be run as indicated in the next
subsection.

**Running the conf examples**

In order to run the evaluations using the provided `conf.ini` or any other *.ini*
file, the following steps are necessary: 

1. Download/Install the libraries/methods you want to test:

  * For running `conf.ini`:

    * OpenNE_
    * PRUNE_
    
  * For running other *.ini* files you may need:

    * Deepwalk_
    * Node2vec_
    * LINE_
    * Metapath2Vec_
    * CNE_

2. Download the datasets used in the examples:

  * For `conf.ini`:

    * StudentDB_
    * Arxiv GR-QC_

  * For other *.ini* files you may need:

    * Facebook_ combined network
    * Facebook-wallpost_
    * Arxiv Astro-Ph_
    * ArXiv Hep-Ph_ (https://snap.stanford.edu/data/cit-HepPh.html)
    * BlogCatalog_ (http://socialcomputing.asu.edu/datasets/BlogCatalog3)
    * Wikipedia_ (http://snap.stanford.edu/node2vec)
    * PPI_
    
3. Set the correct dataset paths in the INPATHS option of the corresponding *.ini* 
file. And the correct method paths under METHODS_OPNE and/or METHODS_OTHER options.  

4. Run the evaluation:

    .. code-block:: console
    
        # For conf.ini run:
        foo@bar:~$ python evalne ./examples/conf.ini
    
        # For conf_node2vec.ini run:
        foo@bar:~$ python evalne ./examples/node2vec/conf_node2vec.ini

.. note::

    The networks provided as input to EvalNE are required to be in edgelist format.

.. _OpenNE: https://github.com/thunlp/OpenNE
.. _PRUNE: https://github.com/ntumslab/PRUNE
.. _Deepwalk: https://github.com/phanein/deepwalk
.. _Node2vec: https://github.com/aditya-grover/node2vec
.. _LINE: https://github.com/tangjianpku/LINE
.. _Metapath2Vec: https://www.dropbox.com/s/w3wmo2ru9kpk39n/code_metapath2vec.zip?dl=0
.. _CNE: https://bitbucket.org/ghentdatascience/cne/

.. _StudentDB: http://adrem.ua.ac.be/smurfig
.. _GR-QC: https://snap.stanford.edu/data/ca-GrQc.html
.. _Facebook: https://snap.stanford.edu/data/egonets-Facebook.html
.. _Facebook-wallpost: http://socialnetworks.mpi-sws.org/data-wosn2009.html
.. _Astro-Ph: http://snap.stanford.edu/data/ca-AstroPh.html
.. _Hep-Ph: https://snap.stanford.edu/data/cit-HepPh.html
.. _BlogCatalog: http://socialcomputing.asu.edu/datasets/BlogCatalog3
.. _Wikipedia: http://snap.stanford.edu/node2vec
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
	G = pp.load_graph('../evalne/tests/data/network.edgelist')
	G, _ = pp.prep_graph(G)

	# Create an evaluator and generate train/test edge split
	nee = evaluator.Evaluator()
	_ = nee.traintest_split.compute_splits(G)

	# Set the baselines
	methods = ['random_prediction', 'common_neighbours', 'jaccard_coefficient']

	# Evaluate baselines
	nee.evaluate_baseline(methods=methods)

	try:
	    # Check if OpenNE is installed
	    import openne

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

	except ImportError:
	    print("The OpenNE library is not installed. Reporting results only for the baselines...")
	    pass

	# Get output
	results = nee.get_results()
	for result in results:
	    result.pretty_print()
    

Output
------

The library can provide two types of outputs, depending on the value of the SCORES option
of the configuration file. If the keyword *all* is specified, the library will generate a 
file named `eval_output.txt` containing for each method and network analysed all the 
metrics available (auroc, precision, f-score, etc.). If more than one experiment repeat 
is requested the values reported will be the average over all the repeats. The output 
file will be located in the same path from which the evaluation was run.

Setting the SCORES option to `%(maximize)` will generate a similar output file as before.
The content of this file, however, will be a table (Alg. x Networks) containing exclusively 
the score specified in the MAXIMIZE option for each combination of method and network
averaged over all experiment repeats. 

Additionally, if the option TRAINTEST_PATH contains a valid filename, EvalNE will create
a file with that name under each of the OUTPATHS provided. In each of these paths the
library will store the true and false train and test sets of edge. 

.. note::
    The tabular output is not available for mixes of directed and undirected networks.
    If this type of output is desired, all values of the option DIRECTED must be either
    True or False.


