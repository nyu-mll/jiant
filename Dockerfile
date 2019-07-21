# Dockerfile for jiant repo. Currently intended to run in our GCP environment.
#
# To set up Docker for local use, follow the first part of the Kubernetes
# install instructions (gcp/kubernetes/README.md) to install Docker and
# nvidia-docker.
#
# For local usage, see demo.with_docker.sh
#
# To run on Kubernetes, see gcp/kubernetes/run_batch.sh
#
# Note that --remote_log currently doesn't work with containers,
# since the host name seen by main.py is the name of the container, not the
# name of the host GCE instance.

# Use CUDA base image.
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# Fix unicode issues in Python3 by setting default text file encoding.
ENV LANG C.UTF-8

# Update Ubuntu packages and install basic utils
RUN apt-get update
RUN apt-get install -y wget git bzip2

# Install Anaconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/anaconda3 \
  && rm Miniconda3-latest-Linux-x86_64.sh

# Set path to conda
ENV PATH /opt/anaconda3/bin:$PATH

# Set up conda environment (slow - installs many packages)
COPY environment.yml .
RUN conda env create -f environment.yml

# Workaround for 'conda activate' depending on shell features
# that don't necessarily work in Docker.
# This simulates the effect of 'conda activate'
# See https://github.com/ContinuumIO/docker-images/issues/89
# If this breaks in a future version of conda, add
#   RUN conda shell.posix activate jiant
# to see what conda activate jiant would do, and update the commands below
# accordingly.
ENV PATH /opt/anaconda3/envs/jiant/bin:$PATH
ENV CONDA_PREFIX "/opt/anaconda3/envs/jiant"
ENV CONDA_SHLVL "1"
ENV CONDA_DEFAULT_ENV "jiant"

# Install SpaCy and NLTK models
RUN python -m spacy download en
RUN python -m nltk.downloader -d /usr/share/nltk_data \
  perluniprops nonbreaking_prefixes

# Local AllenNLP cache, may be used for ELMo weights.
RUN mkdir -p /tmp/.allennlp && chmod a+w /tmp/.allennlp
ENV ALLENNLP_CACHE_ROOT "/tmp/.allennlp"

# Create local mount points.
# Can override with --build-arg NFS_MOUNT=/path/to/nfs/volume
# when running 'docker build'
ARG NFS_MOUNT="/nfs/jiant"
RUN mkdir -p "$NFS_MOUNT"

