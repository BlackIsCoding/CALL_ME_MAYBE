#!/bin/bash

mkdir -p "$HOME/goinfre/.cache/uv"
export UV_CACHE_DIR="$HOME/goinfre/.cache/uv"
export UV_PROJECT_ENVIRONMENT=$HOME/goinfre/venvs/call_final

export PIP_CACHE_DIR="$HOME/goinfre/.cache/pip"
export MYPY_CACHE_DIR="$HOME/goinfre/mypy_cache"
export FLAKE8_CACHE_DIR="$HOME/goinfre/flake8_cache"

export HF_HOME=/goinfre/$USER/huggingface
export HF_HUB_CACHE=/goinfre/$USER/huggingface/hub                                         
export TRANSFORMERS_CACHE=/goinfre/$USER/transformers