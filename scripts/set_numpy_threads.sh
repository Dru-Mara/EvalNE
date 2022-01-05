#!/bin/bash
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 03/01/2022
# Howto: 
# 1) give exec permisions with: chmod +x ./set_numpy_threads.sh
# 2) run script with: source ./set_numpy_threads.sh

read -p 'Input maximum number of threads: ' threads
export OMP_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export MKL_NUM_THREADS=$threads
export VECLIB_MAXIMUM_THREADS=$threads
export NUMEXPR_NUM_THREADS=$threads
echo 'Maximum number of threads set!'
