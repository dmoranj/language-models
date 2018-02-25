# Language Model Experiments

## Overview

This repository's goal is to contain a set of experiments on language modelling with
different technologies, none of them intended to have any use apart from that of a
didactic resource.

Models will be presented as a set of command line tools for training and evaluation
(in Python) with some associated data analysis resources (in R).

## Character level RNN models

This tool aims to train a set of models to learn a distribution from a corpus of text 
at a character level. In order to accomplish this task, the text is split into lines,
each line representing an example for the algorithm. Each example line is codified with
the following two actions:

* Each character is one-hot encoded with the default alphabet. This alphabet contains
all the lowercase letters, numbers and the common punctuation signs.

* The line is adapted to the maximum line length, by removing the exceed characters
(and padding with the all-zeroes one-hot vector).

Lines are considered to be always separated by dots (some preprocessing is performed
to ensure that, see the Text Cleaning section below).


### RNN Model

The RNN model provided by the tool has some customization level, but is mainly composed
of the following components:

* The input layer (composed of sequences of one-hot encoded characters).
* A sequence of connected RNN layers (either LSTM or GRU). 
* A sequence of time-distributed Dense layers (i.e: the layers share weights for all
sequence steps).
* A time-distributed Softmax activation layer that outputs a one-hot encoded character.

Each example input to the model is a single line, displaced one character to the left, 
the example label being the original line. The model is, thus, a sequence-to-sequence model,
encouraged to generate the original text character by character when presented with a blank.
This makes the model unsupervised in a certain sense, although it is trained with backprop 
anyway. The optimization algorithm is Adam, with configurable learning rate.


Generation process


### Command line tool



### Text cleaning
clean_text


### Training analysis

