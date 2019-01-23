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

To install, first clone the EvalNE repository by running the following command:

.. code-block:: console

    foo@bar:~$ git clone https://github.com/Dru-Mara/EvalNE.git
    foo@bar:~$ cd EvalNE

Alternatively, the repository can be cloned from BitBucket using:

.. code-block:: console

    foo@bar:~$ git clone https://dru04@bitbucket.org/ghentdatascience/evalne.git
    foo@bar:~$ cd evalne

Then, the following commands will download the required dependencies and install the library:

.. code-block:: console

    foo@bar:~$ pip install -r requirements.txt
    foo@bar:~$ python setup.py install
