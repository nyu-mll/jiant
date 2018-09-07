# Dockerfile for jiant repo. Currently intended to run in our GCP environment.
#
# Usage:
#   docker build -t jiant-sandbox:v1 .
#   export JIANT_PATH="/nfs/jsalt/path/to/jiant"
#   docker run --runtime=nvidia --rm -v "/nfs/jsalt:/nfs/jsalt" jiant-sandbox:v1 \
#       -e "NFS_PROJECT_PREFIX=/nfs/jsalt/exp/docker" \
#       -e "JIANT_PROJECT_PREFIX=/nfs/jsalt/exp/docker" \
#       python $JIANT_PATH/main.py --config_file $JIANT_PATH/demo.conf \
#       [ ... additional args to main.py ... ]
#
# To run on Kubernetes, see gcp/kubernetes/run_batch.sh
#
# Note that --remote_log currently doesn't work with the above command,
# since the host name seen by main.py is the name of the container, not the
# name of the host GCE instance.

# Use CUDA base image.
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# Add Tini to handle init.
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

# Update Ubuntu packages
RUN apt-get update && yes | apt-get upgrade

# Add utils
RUN apt-get install -y wget git bzip2

# Install Anaconda
# TODO: replace with miniconda to reduce image size.
RUN wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh \
  && bash Anaconda3-5.2.0-Linux-x86_64.sh -b \
  && rm Anaconda3-5.2.0-Linux-x86_64.sh

# Set path to conda
ENV PATH /root/anaconda3/bin:$PATH

# Fix some package issues
RUN pip install msgpack

# Install latest TensorFlow
# TODO: pin this to a specific version!
RUN pip install --upgrade tensorflow-gpu tensorflow-hub

# Install PyTorch 0.4
RUN conda install pytorch=0.4.0 torchvision=0.2.1 cuda90 -c pytorch

# Install other requirements
RUN conda install numpy=1.14.5 nltk=3.2.5
RUN pip install ipdb tensorboard tensorboardX==1.2

# Install AllenNLP
RUN pip install allennlp==0.5.1

# Install misc util packages
RUN pip install --upgrade google-cloud-logging sendgrid
RUN pip install python-Levenshtein ftfy==5.4.1 spacy==2.0.11
RUN python -m spacy download en

# Install local data files.
RUN python -m nltk.downloader perluniprops nonbreaking_prefixes punkt

# Fix permission issues when running as non-root.
# This directory is where anaconda3 and nltk_data reside.
# TODO: see about a workaround if we can install elsewhere in the container.
RUN chmod go+rx /root

# Fix unicode issues in Python3 by setting default text file encoding.
ENV LANG C.UTF-8

# Create local dir for NFS mount.
RUN mkdir -p /nfs/jsalt
# Set environment vars based on gcp/config/jsalt_paths.1.2.sh
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
