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

    # Python 2
    foo@bar:~$ pip install -r requirements.txt
    foo@bar:~$ sudo python setup.py install

    # Python 3
    foo@bar:~$ pip3 install -r requirements.txt
    foo@bar:~$ sudo python3 setup.py install


