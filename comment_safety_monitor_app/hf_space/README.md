---
title: Comment Safety Monitor API
sdk: docker
app_port: 7860
pinned: false
---

# Hugging Face Space deployment

This folder contains the Docker template used to deploy the FastAPI backend to a Hugging Face Space.

The Space root should include:

- `Dockerfile` copied from `hf_space/Dockerfile`
- `requirements.txt` copied from `requirements-hf-space.txt`
- `backend/`
- `inference.py`
- `model/distilbert_model/`

The model weights are required at deploy time but are intentionally not tracked in Git.
