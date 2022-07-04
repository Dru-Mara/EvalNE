Installation
============

The library has been tested on Python 3.8. The supported platforms
include Linux, Mac OS and Microsoft Windows.

EvalNE depends on the following open-source packages:
   * Numpy
   * Scipy
   * Scikit-learn
   * Matplotlib
   * NetworkX
   * Pandas
   * tqdm
   * kiwisolver

Linux/MacOS
-----------

Before installing EvalNE make sure that **pip** and **python-tk** packages are installed 
on your system, this can be done by running:

.. code-block:: console

    foo@bar:~$ sudo apt-get install python3-pip
    foo@bar:~$ sudo apt-get install python3-tk

**Option 1:** Install the library using `pip`

.. code-block:: console

    foo@bar:~$ pip install evalne

**Option 2:** Cloning the code and installing

	- Clone the EvalNE repository:

	.. code-block:: console

	    foo@bar:~$ git clone https://github.com/Dru-Mara/EvalNE.git
	    foo@bar:~$ cd EvalNE

	- Install the library:

	.. code-block:: console

	    # System-wide install
	    foo@bar:~$ sudo python setup.py install
	    
	    # Alterntive single user install
	    foo@bar:~$ python setup.py install --user	    
	    
	- Alternatively, one can first download the required dependencies and then install:

	.. code-block:: console

	    foo@bar:~$ pip install -r requirements.txt
	    foo@bar:~$ sudo python setup.py install

Check the installation by running `simple_example.py` or `functions_example.py` as shown below.
If you have installed the package using pip, you will need to download the examples folder from
the github repository first.

.. code-block:: console

    foo@bar:~$ cd examples/
    foo@bar:~$ python simple_example.py

.. note::

    In order to run the `evaluator_example.py` script, the OpenNE library, PRUNE and Metapath2Vec are required. Further instructions on where to obtain and how to install these methods/libraries are provided in :doc:`the quickstart <quickstart>` section.

