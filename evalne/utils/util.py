#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

from __future__ import division
from __future__ import print_function

import importlib
import shlex
import os
from subprocess import Popen
from threading import Timer


class TimeoutExpired(Exception):
    pass


def auto_import(classpath):
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
    finally:
        if proc.poll() != 0:
            raise TimeoutExpired('Command `{}` timed out'.format(cmd))
        timer.cancel()
