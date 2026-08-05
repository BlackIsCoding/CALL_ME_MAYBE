#!/bin/bash

mkdir -p "$HOME/goinfre/.cache/uv"
export UV_CACHE_DIR="$HOME/goinfre/.cache/uv"
export PIP_CACHE_DIR="$HOME/goinfre/.cache/pip"
export HF_HOME=/goinfre/$USER/huggingface
export HF_HUB_CACHE=/goinfre/$USER/huggingface/hub                                         
export TRANSFORMERS_CACHE=/goinfre/$USER/transformers