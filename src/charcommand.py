import argparse
import charactermodel as cmodel
import characterpredict as cpredict
import time

DEFAULT_EPOCHS=10
DEFAULT_ITERATIONS=5
DEFAULT_DECODE_OPTION='choice'
DEFAULT_LEARNING_RATE=0.0001
DEFAULT_RNN_LAYERS=1
DEFAULT_RNN_TYPE='LSTM'
GRU_HIDDEN_UNITS = 128
DEFAULT_BATCH_SIZE = 7000
DEFAULT_MINIBATCH_SIZE = 128
MAX_LENGTH = 750
SAVE_PATH = './results/'
MODEL_FILE = SAVE_PATH + 'charmodelwideanddeep.h5'
STATS_FILE = SAVE_PATH + 'charmodelwideanddeep.csv'
ALPHABET_FILE = SAVE_PATH + 'alphabet.json'

def generateDescription():
    return """
        This is is the command line interface for training, evaluating and use of the generative language character
        model. The tool provide three different command options one for each one of this tasks. Be aware that some 
        arguments may be command-specific; those cases are indicated with the capitalized command in brackets. 
        
        In order to train the model, a folder containing the text corpus in plain text format files with '.txt' suffix
        is required. Text will be processed to remove excess whitespaces and characters outside the considered alphabet
        will be replaced by the # character before training. The model is trained to generate lines, i.e.: each example
        consists of a line of text with a blank before it and a target that is the line itself.
        
        In order to allow big datasets to be used in training the model the corpus is loaded in batches of lines. When
        the training begins, each batch of lines is trained for the selected number of epochs before passing to the next
        batch. One full pass of the text corpus is an iteration. Be aware that for big enough batches, memory problems
        may arise (7000 lines have shown to be a conservative well-behaved number). Files are loaded as they are needed
        in a random order (potentially different for each iteration).
        
        The model is saved at the end of each iteration.
    """

def defineParser():
    parser = argparse.ArgumentParser(description=generateDescription())
    parser.add_argument('command', type=str, help='Command to execute: train, evaluate or generate')
    parser.add_argument('--datasetPath', dest='datasetPath', type=str,
                        help='Path to the dataset folder')
    parser.add_argument('--modelPath', dest='modelPath', type=str, default=MODEL_FILE,
                        help='Path to save the model (for training) or load it (in generation and evaluation)')
    parser.add_argument('--statsPath', dest='statsPath', type=str, default=STATS_FILE,
                        help='Path to save the model statistics [TRAIN]')
    parser.add_argument('--alphabetPath', dest='alphabetPath', type=str, default=ALPHABET_FILE,
                        help='Path to save the alphabet (for training) or load it (in generation and evaluation)')
    parser.add_argument('--maxLength', dest='maxLength', type=int, default=MAX_LENGTH,
                        help='Maximum line length (greater lines will be trimmed to this length).')
    parser.add_argument('--batchSize', dest='batchSize', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Maximum batch size [TRAIN].')
    parser.add_argument('--minibatchSize', dest='minibatchSize', type=int, default=DEFAULT_MINIBATCH_SIZE,
                        help='Maximum minibatch size [TRAIN].')
    parser.add_argument('--rnnLayers', dest='rnnLayers', type=int, default=DEFAULT_RNN_LAYERS,
                        help='Number of RNN Layers of the model.')
    parser.add_argument('--learningRate', dest='learningRate', type=float, default=DEFAULT_LEARNING_RATE,
                        help='Learning rate [TRAIN].')
    parser.add_argument('--rnnType', dest='rnnType', type=str, default=DEFAULT_RNN_TYPE,
                        help='Default type of RNN Cell: LSTM or GRU.')
    parser.add_argument('--hidden', dest='hidden', type=int, default=GRU_HIDDEN_UNITS,
                        help='Number of hidden units per RNN layer')
    parser.add_argument('--decodeOption', dest='decodeOption', type=str, default=DEFAULT_DECODE_OPTION,
                        help='Indicates which method to use for character decoding giving the posterior [GENERATE]')
    parser.add_argument('--epochs', dest='epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of epochs to train the model for [TRAIN].')
    parser.add_argument('--iterations', dest='iterations', type=int, default=DEFAULT_ITERATIONS,
                        help='Number of iterations to train the model for [TRAIN].')
    parser.add_argument('--load', dest='load', type=bool, default=False,
                        help='Flat to indicate whether to train a new model or load a new one [TRAIN]')

    return parser


def start():
    args = defineParser().parse_args()

    start = time.time()

    if args.command == 'train':
        cmodel.train_model(args)
    elif args.command == 'evaluate':
        print('Evaluating')
    elif args.command == 'generate':
        cpredict.generation(args)
    else:
        print('Unknown command.')

    end = time.time()
    print('\n\nCommand execution time: {0:.2f} min'.format((end - start)/60))

start()

