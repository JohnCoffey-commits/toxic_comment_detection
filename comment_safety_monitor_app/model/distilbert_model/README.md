# Local Model Weights

This directory contains the tokenizer and model metadata used by the local
inference layer.

The `model.safetensors` weight file is required for local inference, but it is
not committed as a normal Git file because it is larger than GitHub's standard
100 MB single-file limit. Keep the local copy in this directory, or use Git LFS
or external model storage before deploying the backend.
