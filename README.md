# EvalNE: A Python library for evaluating Network Embedding methods on Link Prediction #

[![Documentation Status](https://readthedocs.org/projects/evalne/badge/?version=latest)](https://evalne.readthedocs.io/en/latest/?badge=latest)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Dru-Mara/EvalNE/issues)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Dru-Mara/EvalNE/blob/master/LICENSE)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-sphinx-doc](https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg)](https://www.sphinx-doc.org/)

This repository provides the source code for EvalNE, an open-source Python
library designed for assessing and comparing the performance of Network
Embedding (NE) methods on Link Prediction (LP) tasks. The library intends to
simplify this complex and time consuming evaluation process by providing
automation and abstraction of tasks such as hyper-parameter tuning, selection of
train and test edges, negative sampling, selection of the scoring function, etc.

The library can be used both as a command line tool and an API. In its current 
version, EvalNE can evaluate unweighted directed and undirected simple networks.

The library is maintained by Alexandru Mara (alexandru.mara(at)ugent.be). The full
documentation of EvalNE is hosted by *Read the Docs* and can be found 
[here](https://evalne.readthedocs.io/en/latest/).

#### For Methodologists ####
A command line interface in combination with a configuration file allows the user
to evaluate any publicly available implementation of a NE method without the need
to write additional code. These implementations can be obtained from libraries 
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
* Numpy
* Scipy
* Sklearn
* Matplotlib
* Networkx 2.2

Before installing EvalNE make sure that `pip` and `python-tk` packages are installed 
on your system, this can be done 
by running:
```bash
# Python 2
sudo apt-get install python-pip
sudo apt-get install python-tk

# Python 3
sudo apt-get install python3-pip
sudo apt-get install python3-tk
```

Clone the EvalNE repository:
```bash
git clone https://github.com/Dru-Mara/EvalNE.git
cd EvalNE
```

Install strict library requirements:
```bash
# Python 2
pip install -r requirements.txt
sudo python setup.py install

# Python 3
pip3 install -r requirements.txt
sudo python3 setup.py install
```

**NOTE:** In order to run the examples the OpenNE library, PRUNE and Metapath2Vec are 
required. The instructions for installing them are available 
[here](https://github.com/thunlp/OpenNE), [here](https://github.com/ntumslab/PRUNE), 
and [here](https://www.dropbox.com/s/w3wmo2ru9kpk39n/code_metapath2vec.zip?dl=0), 
respectively.


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

1. Install OpenNE and PRUNE as shown in the *Instalation* section.

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
    python evalne ./examples/node2vec/conf_node2vec.ini
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

``` 

### Output ###

The library can provide two types of outputs, depending on the value of the SCORES option
of the configuration file. If the keyword *all* is specified, the library will generate a 
file named `eval_output.txt` containing for each method and network analysed all the 
metrics available (auroc, precision, f-score, etc.). If more than one experiment repeat 
is requested the values reported will be the average over all the repeats. The output 
file will be located in the same path from which the evaluation was run.

Setting the SCORES option to `%(maximize)` will generate a similar output file as before.
The content of this file, however, will be a table (Alg.\Network) containing exclusively 
the score specified in the MAXIMIZE option for each combination of method and network
averaged over all experiment repeats. 

Additionally, if the option TRAINTEST_PATH contains a valid filename, EvalNE will create
a file with that name under each of the OUTPATHS provided. In each of these paths the
library will store the true and false train and test sets of edge. 

**NOTE**: The tabular output is not available for mixes of directed and undirected networks.


## Citation ##

If you have found EvaNE usefull in your research, please cite our 
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

