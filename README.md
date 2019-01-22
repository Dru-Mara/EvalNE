# EvalNE: A Python library for evaluating Network Embedding methods on Link Prediction #

This repository provides the source code for EvalNE, an open source Python
library designed for assessing and comparing the performance of Network
Embedding (NE) methods on Link Prediction (LP) tasks. The library intends to
simplify this complex and time consuming evaluation process by providing
automation and abstraction of tasks such as hyper-parameter tuning, selection of
train and test edges, negative sampling, selection of the scoring function, etc.

The library can be used both as a command line tool and an API. 

The library is maintained by Alexandru Mara (alexandru.mara(at)ugent.be).

#### For Methodologists ####
A command line interface in combination with a configuration file allow the user
to evaluate any publicly available implementation of a NE method. These
implementations can be obtained from libraries such as 
[OpenNE](https://github.com/thunlp/OpenNE) or
[GEM](https://github.com/palash1992/GEM) 
as well as directly from the web pages of the authors e.g. [CNE](),
[PRUNE](https://github.com/ntumslab/PRUNE). 

EvalNE also includes the following LP heuristics for both directed and
undirected networks (in and out node neighbourhoods), which can be used as
baselines:

* Random Prediction
* Common Neighbours
* Jaccard Coefficient
* Adamic Adar Index
* Preferential Attachment
* Resource Allocation Index

#### For practitioners ####
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


## Instalation ##

The library has been tested on Python 2.7 and Python 3.6.

EvalNE depends on the following packages:
- Numpy
- Scipy
- Sklearn
- Matplotlib
- Networkx 2.2

To install, clone this repository, `cd` to the parent folder or the library and
run:
```bash
pip install -r requirements.txt
python setup.py install
```

## Usage ##

### As a command line tool ###

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

#### Running the conf examples ####

In order to run the evaluations using the provided `conf.ini` and 
`conf_node2vec.ini` files, the following steps are necessary: 

1. Download the libraries/methods used in the examples:
   * [OpenNE](https://github.com/thunlp/OpenNE) 
   * [PRUNE](https://github.com/ntumslab/PRUNE)
2. Download the datasets used in the examples:
   * For `conf.ini`:
      * [StudentDB](http://adrem.ua.ac.be/smurfig)
      * [Arxiv GR-QC](https://snap.stanford.edu/data/ca-GrQc.html)
   * For `conf_node2vec.ini`:
      * [Facebook](https://snap.stanford.edu/data/egonets-Facebook.html) combined network
      * [Arxiv Astro-Ph](http://snap.stanford.edu/data/ca-AstroPh.html)
      * [PPI](http://snap.stanford.edu/node2vec/Homo_sapiens.mat)
    
3. Set the correct dataset paths in the INPATHS option of the corresponding *.ini* 
file. And the correct path for PRUNE under the METHODS_OTHER option. 

4. Run the evaluation:
    ```bash
    # For conf.ini run:
    python evalne ./examples/conf.ini

    # For conf_node2vec.ini run:
    python evalne ./examples/conf_node2vec.ini
    ```

**Note**: The input networks for EvalNE are required to be in edgelist form.

### As an API ###

The library can be imported and used like any other Python module. Next we
present a very basic example, for more complete ones we refer the user to the
`examples/` folder.

```python
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

``` 

### Output ###

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

NOTE: The tabular output is not available for mixes of directed and undirected networks.

 