# EvalNE: A Python library for evaluating Network Embedding methods #

[![Documentation Status](https://readthedocs.org/projects/evalne/badge/?version=latest)](https://evalne.readthedocs.io/en/latest/?badge=latest)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Dru-Mara/EvalNE/issues)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Dru-Mara/EvalNE/blob/master/LICENSE)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-sphinx-doc](https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg)](https://www.sphinx-doc.org/)

This repository provides the source code for EvalNE, an open-source Python
library designed for assessing and comparing the performance of Network
Embedding (NE) methods on Link Prediction (LP), Sign prediction (SP), 
Network Reconstruction (NR) and Node Classification (NC) tasks. 
The library intends to simplify these complex and time consuming evaluation 
processes by providing automation and abstraction of tasks such as 
hyper-parameter tuning and model validation, node and edge sampling, node-pair 
embedding computation, results reporting and data visualization.

The library can be used both as a command line tool and an API. In its current 
version, EvalNE can evaluate unweighted directed and undirected simple networks.

The library is maintained by Alexandru Mara (alexandru.mara(at)ugent.be). The full
documentation of EvalNE is hosted by *Read the Docs* and can be found 
[here](https://evalne.readthedocs.io/en/latest/).

#### For Methodologists ####
A command line interface in combination with a configuration file (describing datasets, 
methods and evaluation setup) allows the user to evaluate any embedding method and compare 
it to the state of the art or replicate the experimental setup of existing papers without 
the need to write additional code. EvalNE does not provide implementations of any NE methods
but offers the necessary environment to evaluate any off-the-shelf algorithm. 
Implementations of NE methods can be obtained from libraries 
such as 
[OpenNE](https://github.com/thunlp/OpenNE) or
[GEM](https://github.com/palash1992/GEM) 
as well as directly from the web pages of the authors e.g. 
[Deepwalk](https://github.com/phanein/deepwalk),
[Node2vec](https://github.com/aditya-grover/node2vec),
[LINE](https://github.com/tangjianpku/LINE),
[PRUNE](https://github.com/ntumslab/PRUNE),
[Metapath2vec](https://ericdongyx.github.io/metapath2vec/m2v.html),
[CNE](https://bitbucket.org/ghentdatascience/cne/).

EvalNE does, however, includes the following LP heuristics for both directed and
undirected networks (in and out node neighbourhoods), which can be used as
baselines for different downstream tasks:

* Random Prediction
* Common Neighbours
* Jaccard Coefficient
* Adamic Adar Index
* Preferential Attachment
* Resource Allocation Index
* Cosine Similarity
* Leicht-Holme-Newman index
* Topological Overlap
* Katz similarity
* All baselines (a combination of the first 5 heuristics in a 5-dim embedding)

#### For practitioners ####
When used as an API, EvalNE provides functions to:

* Load and preprocess graphs
* Obtain general graph statistics
* Conveniently read node/edge embeddings from files
* Sample nodes/edges to form train/test/validation sets
* Different approaches for edge sampling:
    * Timestamp based sampling: latest nodes are used for testing
    * Random sampling: random split of edges in train and test sets
    * Spanning tree sampling: train set will contain a spanning tree of the graph
    * Fast depth first search sampling: similar to spanning tree but based of DFS
* Negative sampling or generation of non-edge pairs using:
    * Open world assumption: train non-edges do not overlap with train edges
    * Closed world assumption: train non-edges do not overlap with either train nor test edges
* Evaluate LP, SP and NR for methods that output: 
    * Node Embeddings
    * Node-pair Embeddings
    * Similarity scores (e.g. the ones given by LP heuristics)
* Implements simple visualization routines for embeddings and graphs 
* Includes NC evaluation for node embedding methods
* Provides binary operators to compute edge embeddings from node feature vectors:
    * Average
    * Hadamard
    * Weighted L1
    * Weighted L2
* Can use any scikit-learn classifier for LP/SP/NR/NC tasks
* Provides routines to run command line commands or functions with a given timeout
* Includes hyperparameter tuning based on grid search
* Implements over 10 different evaluation metrics such as AUC, F-score, etc.
* AUC and PR curves can be provided as output
* Includes routines to generate tabular outputs and directly parse them to Latex tables


## Instalation ##

The library has been tested on Python 2.7 and Python 3.6.

EvalNE depends on the following packages:
* Numpy
* Scipy
* Scikit-learn
* Matplotlib
* NetworkX
* Pandas
* tqdm

Before installing EvalNE make sure that `pip` and `python-tk` packages are installed 
on your system, this can be done by running:
```bash
# Python 2
sudo apt-get install python-pip
sudo apt-get install python-tk

# Python 3
sudo apt-get install python3-pip
sudo apt-get install python3-tk
```

**Option 1:** Install the library using pip:
```bash
# Python 2
pip install evalne

# Python 3
pip3 install evalne
```

**Option 2:** Cloning the code and installing:

- Clone the EvalNE repository:
    ```bash
    git clone https://github.com/Dru-Mara/EvalNE.git
    cd EvalNE
    ```

- Download strict library dependencies and install:
    ```bash
    # Python 2
    pip install -r requirements.txt
    sudo python setup.py install
    
    # Python 3
    pip3 install -r requirements.txt
    sudo python3 setup.py install
    ```

Check the installation by running `simple_example.py` or `functions_example.py` as shown below.
If you have installed the package using pip, you will need to download the examples folder from
the github repository first.
```bash
# Python 2
cd examples/
python simple_example.py

# Python 3
cd examples/
python3 simple_example.py
```

**NOTE:** In order to run the `evaluator_example.py` script, the 
OpenNE library, PRUNE and Metapath2Vec are required. The instructions for installing 
them are available 
[here](https://github.com/thunlp/OpenNE), [here](https://github.com/ntumslab/PRUNE), 
and [here](https://www.dropbox.com/s/w3wmo2ru9kpk39n/code_metapath2vec.zip?dl=0), 
respectively. The instructions on how to run evaluations using *.ini* files are 
provided in the next section. 


## Usage ##

### As a command line tool ###

The library takes as input an *.ini* configuration file. This file allows the user 
to specify the evaluation settings, from the task to perform to the networks to use, 
data preprocessing, methods and baselines to evaluate, and types of output to provide.

An example `conf.ini` file is provided describing the available options
for each parameter. This file can be either modified to simulate different
evaluation settings or used as a template to generate other *.ini* files.

Additional configuration (*.ini*) files are provided replicating the experimental 
sections of different papers in the NE literature. These can be found in different
folders under `examples/replicated_setups`. One such configuration file is 
`examples/replicated_setups/node2vec/conf_node2vec.ini`. This file simulates the link 
prediction experiments of the paper "Scalable Feature Learning for Networks" by A. Grover 
and J. Leskovec.

Once the configuration is set, the evaluation can be run as indicated in the next
subsection.

#### Running the conf examples ####

In order to run the evaluations using the provided `conf.ini` or any other *.ini*
file, the following steps are necessary: 

1. Download/Install the methods you want to test:
    * For `conf.ini`:
        * Install [OpenNE](https://github.com/thunlp/OpenNE) 
        * Install [PRUNE](https://github.com/ntumslab/PRUNE)
    * For other *.ini* files you may need:
        *   [Deepwalk](https://github.com/phanein/deepwalk),
            [Node2vec](https://github.com/aditya-grover/node2vec),
            [LINE](https://github.com/tangjianpku/LINE),
            [Metapath2vec](https://ericdongyx.github.io/metapath2vec/m2v.html), and/or
            [CNE](https://bitbucket.org/ghentdatascience/cne/).

2. Download the datasets used in the examples:
   * For `conf.ini`:
      * [StudentDB](http://adrem.ua.ac.be/smurfig)
      * [Facebook](https://snap.stanford.edu/data/egonets-Facebook.html) (combined network)
      * [ArXiv GR-QC](https://snap.stanford.edu/data/ca-GrQc.html)
   * For other *.ini* files you may need:
      * [Facebook-wallpost](http://socialnetworks.mpi-sws.org/data-wosn2009.html)
      * [ArXiv Astro-Ph](http://snap.stanford.edu/data/ca-AstroPh.html)
      * [ArXiv Hep-Ph](https://snap.stanford.edu/data/cit-HepPh.html)
      * [BlogCatalog](http://socialcomputing.asu.edu/datasets/BlogCatalog3)
      * [Wikipedia](http://snap.stanford.edu/node2vec)
      * [PPI](http://snap.stanford.edu/node2vec/Homo_sapiens.mat)

3. Set the correct dataset paths in the INPATHS option of the corresponding *.ini* 
file. And the correct method paths under METHODS_OPNE and/or METHODS_OTHER options. 

4. Run the evaluation:
    ```bash
    # For conf.ini run:
    python -m evalne ./examples/conf.ini

    # For conf_node2vec.ini run:
    python -m evalne ./examples/node2vec/conf_node2vec.ini
    ```

**Note**: The input networks for EvalNE are required to be in edgelist format.

### As an API ###

The library can be imported and used like any other Python module. Next, we
present a very basic LP example, for more complete ones we refer the user to the
`examples` folder and the docstring documentation of the evaluator and the split submodules.

```python
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

``` 

### Output ###

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

## Citation ##

If you have found EvaNE useful in your research, please cite our 
[arXiv paper](https://arxiv.org/abs/1901.09691):

```
    @misc{Mara2019,
      author = {Alexandru Mara and Jefrey Lijffijt and Tijl De Bie},
      title = {EvalNE: A Framework for Evaluating Network Embeddings on Link Prediction},
      year = {2019},
      archivePrefix = {arXiv},
      eprint = {1901.09691}
    }
```

