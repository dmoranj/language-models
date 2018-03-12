# Language Model Experiments

## Overview

This repository's goal is to contain a set of experiments on language modelling with
different technologies, none of them intended to have any use apart from that of a
didactic resource.

Models will be presented as a set of command line tools for training and evaluation
(in Python) with some associated data analysis resources (in R).

## Generative RNN models (textcommand.py)

This tool aims to train a set of models to learn a distribution from a corpus of text. 
In order to accomplish this task, the text is split into lines, each line representing 
an example for the algorithm. Two types of models were codified in this experiments:
a character-based model, that learns the conditional probability of generating a given
character given the previous ones; and a word-level model, that do the same, but with
the focus on words.


### Character-level Model

#### Topology

The RNN model provided by the tool has some customization level, but is mainly composed
of the following components:

* An input layer (composed of sequences of one-hot encoded characters).
* A sequence of connected RNN layers (either LSTM or GRU). 
* A sequence of time-distributed Dense layers (i.e: the layers share weights for all
sequence steps).
* A time-distributed Softmax activation layer that outputs a one-hot encoded character.

Each example input to the model is a single line, displaced one character to the left, 
the example label being the original line. The model is, thus, a sequence-to-sequence model,
encouraged to generate the original text character by character when presented with an input
that starts with a blank. This makes the model unsupervised in a certain sense, although it 
is trained with backprop anyway. 

The optimization algorithm used is Adam, with configurable learning rate and the rest of the
parameters fixed.

#### Example codification

The tool contains a hardcoded alphabet that will be used to codify the text. It contains the
alphanumerical characters and some punctuation signs. If the text contains characters that are
not found in the vocabulary, they will be codified as a special character (and output as # if
found in generation).

Each example line is codified with the following two actions:

* First, each character is one-hot encoded with the default alphabet. This alphabet contains
all the lowercase letters, numbers and the common punctuation signs.

* Then the line is adapted to the maximum line length, by removing the exceed characters
(and padding with the all-zeroes one-hot vector).

Lines are considered to be always separated by dots (some preprocessing is performed
to ensure that, see the Text Cleaning section below).

#### Generation process

New texts are generated character by character, starting with a blank. The output of the 
corresponding cell at each point in the sequence is considered to be the probability distribution
of the next possible characters given the previous read input, so, in each sequence step, a
character value is drawn from that distribution at random (using *np.random.choice()* method).

The generation process for the character model makes use of the trained weights of the RNN cells,
adding them to a prediction model that has a single cell, whose predictions are chained to next
prediction execution in order to obtain the full predicted sequence (note that for word models
this step is performed differently).

### Word level model

This model's main difference with the character-level model is the use of words in the input
layer instead of characters. 

Each example is transformed with the following steps:
* First, Word ID's are retrieved from an instance of Kera's Tokenizer, by parsing the whole dataset 
in a preprocessing step. 
* The dataset is then transformed from text lines to arrays of integers, that will be fed to the network.
* Each line is then padded using Kera's *pad_sequences()* method, to get a set of equally long sequences.
They will form the rows of the input matrix.
* After the input layer, the model contains an embedding layer, that trains an embedding for the problem. 
This embedding will form the input to the recurrent layers.

The output of the model is a sequence of vectors representing probabilities over words in one-hot encoding. 

### Command line tool

This is is the command line interface for training, evaluating and use of the
generative language models. The tool provide three different command
options one for each one of this tasks. Be aware that some arguments may be
command-specific; those cases are indicated with the capitalized command in
brackets. The *evaluate* command is only a mock, for the moment being. 

The same tool can be used to train different generative models. There are 
currently two of them implemented: character and word. Model-specific options
are indicated in the argument list.

In order to train the model, a folder containing the text corpus in
plain text format files with '.txt' suffix is required. Text will be processed
to remove excess whitespaces. Characters outside the considered alphabet
will be replaced by the # character before training for character models, and
will be removed from the text for training word models.  

The model is trained to generate lines, i.e.: each example consists of a line 
of text with a blank before it and a target that is the line itself. In order to 
allow big datasets to be used in training the model the corpus is loaded in 
batches of lines. 

When the training begins, each batch of lines is trained for the selected
number of epochs before passing to the next batch. One full pass of the text
corpus is an iteration. Be aware that for big enough batches, memory problems
may arise.

Files are loaded as they are needed in a random order (potentially different
for each iteration). The model is saved at the end of each iteration.

The following is the specification of the command line tool with all its options:

```buildoutcfg
  textcommand.py [-h] [--datasetPath DATASETPATH] [--modelPath MODELPATH]
                      [--statsPath STATSPATH] [--alphabetPath ALPHABETPATH]
                      [--vocabularyPath VOCABULARYPATH]
                      [--maxLength MAXLENGTH] [--batchSize BATCHSIZE]
                      [--minibatchSize MINIBATCHSIZE] [--rnnLayers RNNLAYERS]
                      [--learningRate LEARNINGRATE] [--rnnType RNNTYPE]
                      [--hidden HIDDEN] [--decodeOption DECODEOPTION]
                      [--epochs EPOCHS] [--iterations ITERATIONS]
                      [--embedding EMBEDDING] [--load LOAD]
                      command model
```

The following list shows a brief explanation of each option. Positional arguments
are mandatory.

Positional arguments:
  
* **command**: Command to execute: train, evaluate or generate.
* **model**: Kind of model to train or execute: character or word.

Optional arguments:

* `-h, --help`: show this help message and exit.
* `--datasetPath DATASETPATH`: Path to the dataset folder
* `--modelPath MODELPATH`: Path to save the model (for training) or load it (in generation and evaluation)
* `--statsPath STATSPATH`: Path to save the model statistics [TRAIN]
* `--alphabetPath ALPHABETPATH`: Path to save the alphabet (for training) or load it (in generation and evaluation)
* `--vocabularyPath VOCABULARYPATH`: Path to save and load the vocabulary (just for word models).
* `--maxLength MAXLENGTH`: Maximum line length (greater lines will be trimmed to this length).
* `--batchSize BATCHSIZE`: Maximum batch size [TRAIN].
* `--minibatchSize MINIBATCHSIZE`: Maximum minibatch size [TRAIN].
* `--rnnLayers RNNLAYERS`: Number of RNN Layers of the model.
* `--learningRate LEARNINGRATE`: Learning rate [TRAIN].
* `--rnnType RNNTYPE`: Default type of RNN Cell: LSTM or GRU.
* `--hidden HIDDEN`: Number of hidden units per RNN layer
* `--decodeOption DECODEOPTION`: Indicates which method to use for character decoding giving the posterior [GENERATE]
* `--epochs EPOCHS`:  Number of epochs to train the model for [TRAIN].
* `--iterations ITERATIONS`: Number of iterations to train the model for [TRAIN].
* `--embedding EMBEDDING`: Embedding dimension (for word models only) [TRAIN].
* `--load LOAD`: Flat to indicate whether to train a new model or load a new one [TRAIN]


### Training analysis

In order to ease the analysis of the training process, a CSV file with information about the 
executions is saved for each iteration of the algorithm. This CSV file contains one row for each epoch
with information about the Accuracy and Loss in that epoch.

An R Markdown document is provided along with the code to ease the analysis of the training. This file 
currently offers just basic table information and basic plots of the evolution of Accuracy and Loss.
