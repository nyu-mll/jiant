# Dockerfile for jiant repo. Currently intended to run in our GCP environment.
#
# Usage:
#   docker build -t <image_name> .
#   export JIANT_PATH="/nfs/jsalt/path/to/jiant"
#   docker run --runtime=nvidia --rm -v "/nfs/jsalt:/nfs/jsalt" \
#       python $JIANT_PATH/main.py --config_file $JIANT_PATH/demo.conf \
#       [ ... additional args to main.py ... ]


# FROM ubuntu:16.04
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# Add Tini
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

# Update Ubuntu packages
# RUN apt-get update 
RUN apt-get update && yes | apt-get upgrade

# Add utils
RUN apt-get install -y wget git bzip2

# Install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh \
  && bash Anaconda3-5.2.0-Linux-x86_64.sh -b \
  && rm Anaconda3-5.2.0-Linux-x86_64.sh

# Set path to conda
ENV PATH /root/anaconda3/bin:$PATH

# Fix some package issues
RUN pip install msgpack

# Install TensorFlow 1.8
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
RUN pip install python-Levenshtein

# Install local data files.
RUN python -m nltk.downloader perluniprops nonbreaking_prefixes punkt

##
# Temporary: set up paths so we can run jiant/main.py
RUN mkdir -p /nfs/jsalt
# Set environment vars based on gcp/config/jsalt_paths.1.2.sh
ENV JSALT_SHARE_DIR "/nfs/jsalt/share"
ENV JIANT_DATA_DIR "$JSALT_SHARE_DIR/glue_data"
ENV NFS_PROJECT_PREFIX "/nfs/jsalt/exp/docker"
ENV JIANT_PROJECT_PREFIX "$NFS_PROJECT_PREFIX"
ENV GLOVE_EMBS_FILE "$JSALT_SHARE_DIR/glove/glove.840B.300d.txt"
ENV FASTTEXT_EMBS_FILE "$JSALT_SHARE_DIR/fasttext/crawl-300d-2M.vec"
ENV WORD_EMBS_FILE "$FASTTEXT_EMBS_FILE"
ENV FASTTEXT_MODEL_FILE "."
ENV PATH_TO_COVE "$JSALT_SHARE_DIR/cove"
ENV ELMO_SRC_DIR "$JSALT_SHARE_DIR/elmo"



# ##
# # TEMPORARY: run jupyter so we can inspect the environment
# RUN mkdir /opt/notebooks
# RUN jupyter notebook --generate-config --allow-root
# RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py
# EXPOSE 8888
# CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/opt/notebooks", "--ip='*'", "--port=8888", "--no-browser"]

# CMD ["bash", "-c" , "sleep 100"]
