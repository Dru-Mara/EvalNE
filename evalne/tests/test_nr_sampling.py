#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 24/03/2020

import time

from scipy.sparse import rand

from evalne.utils import split_train_test as stt


def test():
    a = rand(1000, 1000, 0.3)
    a = a.tocsr()

    start = time.time()
    p1, n1 = stt.random_edge_sample(a, samp_frac=0.01)
    end = time.time() - start
    print("Exec time Impl. 1: {}".format(end))

    start = time.time()
    p2, n2 = stt.random_edge_sample(a)
    end = time.time() - start
    print("Exec time Impl. 2: {}".format(end))

    print('Results Impl. 1: {} pos, {} neg.'.format(p1.shape, n1.shape))
    print('Results Impl. 2: {} pos, {} neg.'.format(p2.shape, n2.shape))


if __name__ == "__main__":
    test()
