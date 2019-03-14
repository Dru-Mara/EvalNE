Installation
============

The library has been tested on Python 2.7 and Python 3.6. The supported platforms
include Linux, Mac OS and Microsoft Windows.

EvalNE depends on the following open-source packages:
  * Numpy
  * Scipy
  * Sklearn
  * Matplotlib
  * Networkx 2.2

Linux/MacOS
-----------

Before installing EvalNE make sure that **pip** and **python-tk** packages are installed 
on your system, this can be done by running:

.. code-block:: console

    # Python 2
    foo@bar:~$ sudo apt-get install python-pip
    foo@bar:~$ sudo apt-get install python-tk

    # Python 3
    foo@bar:~$ sudo apt-get install python3-pip
    foo@bar:~$ sudo apt-get install python3-tk

**Option 1:** Install the library using `pip`

.. code-block:: console

    # Python 2
    foo@bar:~$ pip install evalne

    # Python 3
    foo@bar:~$ pip3 install evalne

**Option 2:** Cloning the code and installing

	- Clone the EvalNE repository:

	.. code-block:: console

	    foo@bar:~$ git clone https://github.com/Dru-Mara/EvalNE.git
	    foo@bar:~$ cd EvalNE

	- Download the required dependencies and install the library:

	.. code-block:: console

	    # Python 2
	    foo@bar:~$ pip install -r requirements.txt
	    foo@bar:~$ sudo python setup.py install

	    # Python 3
	    foo@bar:~$ pip3 install -r requirements.txt
	    foo@bar:~$ sudo python3 setup.py install

Check the installation by running `simple_example.py` or `functions_example.py` e.g.:

.. code-block:: console

    # Python 2
    foo@bar:~$ cd examples/
    foo@bar:~$ python simple_example.py
    
    # Python 3
    foo@bar:~$ cd examples/
    foo@bar:~$ python3 simple_example.py

.. note::

    In order to run the `evaluator_example.py` script, the OpenNE library, PRUNE and Metapath2Vec are required. Further instructions on where to obtain and how to install these methods/libraries are provided in :doc:`the quickstart <quickstart>` section.

