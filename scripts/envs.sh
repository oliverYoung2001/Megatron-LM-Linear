#!/bin/bash

# # CUDNN
# spack load cudnn@9.8.0  # Useless
# CUDA
spack load cuda@12.8.1
# Conda
conda deactivate && conda deactivate && conda activate linear
