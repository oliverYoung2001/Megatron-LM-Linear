# pip install -e .
# cd <path-to-mamba-ssm> && pip install -e .
pip install six
pip install sentencepiece
pip install pybind11
# Install TE
CONDA_PACKAGE_PATH=$(dirname `which conda`)/../envs/linear/lib/python3.12/site-packages
export CPLUS_INCLUDE_PATH=${CONDA_PACKAGE_PATH}/nvidia/cudnn/include:$CPLUS_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=${CONDA_PACKAGE_PATH}/nvidia/nccl/include:$CPLUS_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$(dirname `which nvcc`)/../targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
pip install --no-build-isolation transformer_engine[pytorch] 2>&1 | tee ./logs/install_te.log    # Need CUDNN 9.3+
# End
pip install causal_conv1d
pip uninstall triton
pip install triton==3.1.0
# pip install --no-build-isolation causal-conv1d~=1.5
