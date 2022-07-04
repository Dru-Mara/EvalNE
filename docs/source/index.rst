EvalNE - A Framework for Evaluating Network Embeddings
=========================================================================

.. image:: EvalNE-logo.jpg
    :width: 220px
    :alt: EvalNE logo
    :align: center

EvalNE is an open source Python library designed for assessing and comparing the performance of Network Embedding (NE) methods on Link Prediction (LP), Sign Prediction (SP), Network Reconstruction (NR), 
Node Classification (NC) and vizualization downstream tasks. The library intends to simplify these complex and time consuming evaluation processes by providing automation and abstraction of tasks such as model hyper-parameter tuning and model validation, node and edge sampling, node-pair 
embedding computation, results reporting and data visualization, among many others.
EvalNE can be used both as a command line tool and as an API and is compatible with Python 3. 
In its current version, EvalNE can evaluate weighted directed and undirected simple networks.

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

If you have found EvaNE useful in your research, please cite our arXiv paper_ :

.. _paper: https://arxiv.org/abs/1901.09691

.. code-block:: console

    @article{MARA2022evalne,
      title = {EvalNE: A Framework for Network Embedding Evaluation},
      author = {Alexandru Mara and Jefrey Lijffijt and Tijl {De Bie}},
      journal = {SoftwareX},
      volume = {17},
      pages = {},
      year = {2022},
      issn = {100997},
      doi = {10.1016/j.softx.2022.100997},
      url = {https://www.sciencedirect.com/science/article/pii/S2352711022000139}
    }

