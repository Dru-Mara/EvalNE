EvalNE - A Framework for Evaluating Network Embeddings on Link Prediction
=========================================================================

.. image:: EvalNE-logo.jpg
    :width: 220px
    :alt: EvalNE logo
    :align: center

EvalNE is an open source Python library designed for assessing and comparing the performance of Network Embedding (NE) methods on Link Prediction (LP) tasks. The library intends to simplify this complex and time consuming evaluation process by providing automation and abstraction of tasks such as model hyper-parameter tuning, selection of train and test edges, negative edge sampling and selection of the evaluation metrics, among many others.
EvalNE can be used both as a command line tool and as an API and is compatible with Python 2 and Python 3. 
In its current version, EvalNE can evaluate unweighted directed and undirected simple networks.

EvalNE is provided under the MIT_ free software licence and is maintained by Alexandru Mara (alexandru(dot)mara(at)ugent(dot)be). The source code can be found on GitHub_. 

.. _MIT: https://opensource.org/licenses/MIT
.. _GitHub: https://github.com/Dru-Mara/EvalNE


See :doc:`the quickstart <quickstart>` to get started.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   description
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Developer

   api
   release
   contributing

.. toctree::
   :maxdepth: 1
   :caption: More

   license
   acknowledgements
   help


Citation
--------

If you have found EvaNE usefull in your research, please cite our arXiv paper_ :

.. _paper: https://arxiv.org/abs/1901.09691

.. code-block:: console

    @misc{Mara2019,
      author = {Alexandru Mara and Jefrey Lijffijt and Tijl De Bie},
      title = {EvalNE: A Framework for Evaluating Network Embeddings on Link Prediction},
      year = {2019},
      archivePrefix = {arXiv},
      eprint = {1901.09691}
    }

