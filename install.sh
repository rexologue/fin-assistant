#!/usr/bin/env bash
set -e

# git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git
# cd vllm
# pip install -r requirements/build.txt
# pip install -r requirements/cuda.txt
# export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f/vllm-0.9.2-cp38-abi3-manylinux1_x86_64.whl
# VLLM_USE_PRECOMPILED=1 pip install -e . -v --no-build-isolation || pip install -e . -v
# cd ..
# pip install git+https://github.com/huggingface/transformers
# pip install accelerate
# pip install qwen-omni-utils -U
# pip install -U flash-attn --no-build-isolation
pip install pandas pyarrow fastparquet
