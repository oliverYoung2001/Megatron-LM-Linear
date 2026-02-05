#!/bin/bash

# CUDA
spack load cuda@12.8.1
# numactl
spack load numactl@2.0.18
# Conda
conda deactivate && conda deactivate && conda activate linear
