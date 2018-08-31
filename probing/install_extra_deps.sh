#!/bin/bash

# Install extra dependencies for edge probing scripts (preprocessing).

conda install ftfy=5.4.1
conda install spacy=2.0.11
python -m spacy download en
