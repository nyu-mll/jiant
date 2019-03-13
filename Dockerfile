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

# Add Tini to handle init.
# TODO: see if we still need this? More recent docker might have this built-in.
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

# Fix unicode issues in Python3 by setting default text file encoding.
ENV LANG C.UTF-8

# Update Ubuntu packages
RUN apt-get update && yes | apt-get upgrade

# Add utils
RUN apt-get install -y wget git bzip2

# Install Anaconda
# TODO: replace with miniconda to reduce image size.
RUN wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh \
  && bash Anaconda3-5.2.0-Linux-x86_64.sh -b -p /usr/share/anaconda3 \
  && rm Anaconda3-5.2.0-Linux-x86_64.sh

# Set path to conda
ENV PATH /usr/share/anaconda3/bin:$PATH

# Fix some package issues
RUN pip install --upgrade pip
RUN pip install msgpack
RUN pip install nose2

# Install latest TensorFlow
# TODO: pin this to a specific version!
RUN pip install --upgrade tensorflow-gpu tensorflow-hub

# Install PyTorch
RUN conda install pytorch=1.0.0 torchvision=0.2.1 cuda90 -c pytorch

# Install other requirements
RUN conda install numpy=1.14.5 nltk=3.2.5
RUN pip install ipdb tensorboard tensorboardX==1.2

# Install misc util packages.
RUN pip install --upgrade google-cloud-logging sendgrid
RUN pip install python-Levenshtein ftfy==5.4.1 spacy==2.0.11
RUN python -m spacy download en

# Install AllenNLP. Need to update some other deps first.
RUN conda install greenlet=0.4.15
RUN pip install allennlp==0.8.1
RUN pip install --upgrade pytorch-pretrained-bert==0.5.1

# Install local data files.
RUN python -m nltk.downloader -d /usr/share/nltk_data \
  perluniprops nonbreaking_prefixes punkt

RUN pip install pyhocon==0.3.35

# AllenNLP cache, may be used for ELMo weights.
RUN mkdir -p /tmp/.allennlp && chmod a+w /tmp/.allennlp
ENV ALLENNLP_CACHE_ROOT "/tmp/.allennlp"

# Create local mount points.
RUN mkdir -p /share/jiant
RUN mkdir -p /nfs/jsalt
# Set environment vars based on gcp/config/jsalt_paths.1.2.sh
# TODO: make these a generic mount, instead of requiring that they look like our
# NFS directory.
ENV JSALT_SHARE_DIR "/nfs/jsalt/share"
ENV JIANT_DATA_DIR "$JSALT_SHARE_DIR/glue_data"
ENV GLOVE_EMBS_FILE "$JSALT_SHARE_DIR/glove/glove.840B.300d.txt"
ENV FASTTEXT_EMBS_FILE "$JSALT_SHARE_DIR/fasttext/crawl-300d-2M.vec"
ENV WORD_EMBS_FILE "$FASTTEXT_EMBS_FILE"
ENV FASTTEXT_MODEL_FILE "."
ENV PATH_TO_COVE "$JSALT_SHARE_DIR/cove"
ENV ELMO_SRC_DIR "$JSALT_SHARE_DIR/elmo"

# Set these manually with -e or via Kuberentes config YAML.
# ENV NFS_PROJECT_PREFIX "/nfs/jsalt/exp/docker"
# ENV JIANT_PROJECT_PREFIX "$NFS_PROJECT_PREFIX"
