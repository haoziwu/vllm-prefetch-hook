# vllm-prefetch-hook
a vllm plugin hook which prefetches model file before startup

# install
cd {workspace}/vllm-prefetch-hook \
pip install -e .

# usage
export VLLM_PREFETCH_MODEL_PATH=your_model_directory \
export VLLM_PREFETCH_THREADS=8 (default 8,optionally set)  
