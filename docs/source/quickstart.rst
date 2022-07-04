Quickstart
==========

As a command line tool
----------------------

The library takes as input an *.ini* configuration file. This file allows the user 
to specify the evaluation settings, from the task to perform to the networks to use, data preprocessing, methods and baselines to evaluate, and types of output to provide.

An example `conf.ini` file is provided describing the available options
for each parameter. This file can be either modified to simulate different
evaluation settings or used as a template to generate other *.ini* files.

Additional configuration (*.ini*) files are provided replicating the experimental 
sections of different papers in the NE literature. These can be found in different
folders under `examples/replicated_setups`. One such configuration file is 
`examples/replicated_setups/node2vec/conf_node2vec.ini`. This file simulates the link prediction 
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
    * Facebook_ combined network
    * Arxiv GR-QC_

  * For other *.ini* files you may need:

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
        foo@bar:~$ python -m evalne ./examples/conf.ini
    
        # For conf_node2vec.ini run:
        foo@bar:~$ python -m evalne ./examples/node2vec/conf_node2vec.ini

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

The library can be imported and used like any other Python module. Next, we
present a very basic LP example, for more complete ones we refer the user to the
`examples` folder and the docstring documentation of the evaluator and the split submodules.

::

	from evalne.evaluation.evaluator import LPEvaluator
	from evalne.evaluation.split import LPEvalSplit
	from evalne.evaluation.score import Scoresheet
	from evalne.utils import preprocess as pp

	# Load and preprocess the network
	G = pp.load_graph('../evalne/tests/data/network.edgelist')
	G, _ = pp.prep_graph(G)

	# Create an evaluator and generate train/test edge split
	traintest_split = LPEvalSplit()
	traintest_split.compute_splits(G)
	nee = LPEvaluator(traintest_split)

	# Create a Scoresheet to store the results
	scoresheet = Scoresheet()

	# Set the baselines
	methods = ['random_prediction', 'common_neighbours', 'jaccard_coefficient']

	# Evaluate baselines
	for method in methods:
	    result = nee.evaluate_baseline(method=method)
	    scoresheet.log_results(result)

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
		results = nee.evaluate_cmd(method_name=methods[i], method_type='ne', command=command,
		                           edge_embedding_methods=edge_emb, input_delim=' ', output_delim=' ')
		scoresheet.log_results(results)

	except ImportError:
	    print("The OpenNE library is not installed. Reporting results only for the baselines...")
	    pass

	# Get output
	scoresheet.print_tabular()
    

Output
------

The library stores all the output generated in a single folder per execution. The name
of this folder is: `{task}_eval_{month}{day}_{hour}{min}`. Where `{task}` is one of:
lp, sp, nr or nc.

The library can provide two types of outputs, depending on the value of the SCORES option
of the configuration file. If the keyword *all* is specified, the library will generate a 
file named `eval_output.txt` containing for each method and network analysed all the 
metrics available (auroc, precision, f-score, etc.). If more than one experiment repeat 
is requested the values reported will be the average over all the repeats. 

Setting the SCORES option to `%(maximize)` will generate a similar output file as before.
The content of this file, however, will be a table (Alg. x Networks) containing exclusively 
the score specified in the MAXIMIZE option for each combination of method and network
averaged over all experiment repeats. In addition a second table indicating the average 
execution time per method and dataset will be generated.

If the option CURVES is set to a valid option then for each method dataset and experiment 
repeat a PR or ROC curve will be generated. If the option SAVE_PREP_NW is set to True, each
evaluated network will be stored, in edgelist format, in a folder with the same name as the 
network.

Finally, the library also generates an `eval.log` file and a `eval.pkl`. The first file 
contains important information regarding the evaluation process such as methods whose 
execution has failed, or validation scores. The second one encapsulates all the evaluation
results as a pickle file. This file can be conveniently loaded and the results can be 
transformed into e.g. pandas dataframes or latex tables.

Parallelization
---------------

EvalNE makes extensive use of numpy for most operations. Numpy, in turn, 
uses other libraries such as OpenMP, MKL, etc., to provide parallelization. In order to allow for 
certain control on the maximum number of threads used during execution, we include a simple bash 
script (`set_numpy_threads.sh`). The script located inside the `scripts` folder can be given execution permissions and run as follows:

.. code-block:: console

    # Give execution permissions:
    chmod +x set_numpy_threads.sh

    # Run the script:
    source set_numpy_threads.sh
    # The script will then ask for the maximum number of threads to use.


