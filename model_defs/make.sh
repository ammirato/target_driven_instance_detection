#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

python setup_py3.py build_ext --inplace

rm -rf build/
