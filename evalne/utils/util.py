#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

# This file provides a set of functions for importing binary classifiers from a given sting and for running command line
# calls and python functions in independent processes with given pre-defined timeouts.

from __future__ import division
from __future__ import print_function

import importlib
import os
import shlex
from subprocess import Popen
from threading import Timer
from multiprocessing import Process, Queue
try:
    import Queue as queue
except ImportError:
    import queue


class TimeoutExpired(Exception):
    pass


def auto_import(classpath):
    """
    Imports any Sklearn binary classifier from a string.

    Parameters
    ----------
    classpath : string
        A string indicating the full path the any Sklearn classifier and its parameters.

    Returns
    -------
    clf : object
        The classifier instance.

    Raises
    ------
    ValueError
        If the classifier could not be imported.

    Examples
    --------
    Importing the SVC classifier with user-defined parameters:

    >>> auto_import("sklearn.svm.SVC(C=1.0, kernel='rbf')")
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

    Importing a decision tree classifier with no parameters:

    >>> auto_import("sklearn.ensemble.ExtraTreesClassifier()")
    ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
        max_depth=None, max_features='auto', max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
        oob_score=False, random_state=None, verbose=0, warm_start=False)

    """
    comps = classpath.split('.')
    if len(comps) > 1:
        for i, c in enumerate(reversed(comps)):
            if c[0] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                break
        class_str = '.'.join(comps[-(i+1):])
        # params = comps[-1].split('(')[1].replace(')', '')
        # param_dict = dict(map(lambda x: x.strip().split('='), params.split(',')))
        module_str = '.'.join(comps[:-(i+1)])
        module = importlib.import_module(module_str)
        clf = eval("module." + class_str)
    else:
        raise ValueError("Classifier has to be an Sklearn class given as e.g. `sklearn.svm.SVC(C=1.0, kernel='rbf')`")
    return clf


def run(cmd, timeout, verbose):
    """
    Runs the cmd command provided as input in a new process. If execution time exceeds timeout, the process is killed
    and a TimeoutExpired exception is raised.

    Parameters
    ----------
    cmd : string
        A string indicating the command to run on the command line.
    timeout : int or float
        A value indicating the maximum number of second the process is allowed to run for.
    verbose : bool
        Boolean indicating if the execution output should be shown or not (pipes stdout and stderr to devnull).

    Raises
    ------
    TimeoutExpired
        If the execution time exceeds the number of second indicated by timeout.

    Notes
    -----
    The method additionally raises ImportError, IOError and AttributeError if these are encountered during execution
    of the cmd command.

    Examples
    --------
    Runs a command that prints Start, sleeps for 5 seconds and prints Done

    >>> util.run("python -c 'import time; print(\"Start\"); time.sleep(5); print(\"Done\")'", 7, True)
    Start
    Done

    Same as previous command but now it does not print Done because it times out after 2 seconds

    >>> util.run("python -c 'import time; print(\"Start\"); time.sleep(5); print(\"Done\")'", 2, True)
    Start
    Traceback (most recent call last):
      File "<input>", line 1, in <module>
      File "EvalNE/evalne/utils/util.py", line 84, in run
        A string indicating the command to run on the command line.
    TimeoutExpired: Command `python -c 'import time; print("Start"); time.sleep(5); print("Done")'` timed out

    """
    if verbose:
        sto = None
        ste = None
    else:
        devnull = open(os.devnull, 'w')
        sto = devnull
        ste = devnull
    # Alternative without timeout
    # subprocess.run(cmd, shell=True, stdout=sto, stderr=ste)
    proc = Popen(shlex.split(cmd), stdout=sto, stderr=ste)
    timer = Timer(timeout, proc.kill)
    try:
        timer.start()
        proc.communicate()

    except (ImportError, IOError, AttributeError) as e:
        raise e

    finally:
        if not timer.is_alive() and proc.poll() != 0:
            raise TimeoutExpired('Command `{}` timed out'.format(cmd))
        timer.cancel()


def run_function(timeout, func, *args):
    """
    Runs the function provided as input in a new process. If execution time exceeds timeout, the process is killed
    and a TimeoutExpired exception is raised.

    Parameters
    ----------
    timeout : int or float
        A value indicating the maximum number of seconds the process is allowed to run for or None.
    func : object
        A function to be executed. First parameter must be a queue object. Check notes section for more details.
    *args
        Variable length argument list for function func. The list shloud **not** include the queue object.

    Raises
    ------
    TimeoutExpired
        If the execution time exceeds the number of second indicated by timeout.

    Notes
    -----
    The input function func must take as a first parameter a queue object q. This is used to communicate results
    between the process where the function is running and the main thread. The list of args does not need to include
    the queue object, it is automatically inserted by this function.

    Examples
    --------
    Runs function foo for at most 100 seconds and returns the result. Foo must put the result in q.

    >>> def foo(q, a, b):
    ...     q.put(a+b)
    ...
    >>> run_function(100, foo, *[1, 2])
    3

    """
    # Initialize a process to run the function and a queue to communicate results
    q = Queue()
    p = Process(target=func, args=(q,) + args)

    try:
        # Start the process and join with given timeout
        p.start()
        # Block until results are available in the queue and start consuming them
        res = q.get(block=True, timeout=timeout)
        # Finish the child process
        p.join()
        return res

    except queue.Empty:
        raise TimeoutExpired('Function `{}` has timed out'.format(func.__name__))

    finally:
        # Make sure we kill the process after execution if its still alive
        if p.is_alive():
            p.terminate()
