from keras.layers import *
from keras.models import Model, load_model
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from modelutils import save_model
import numpy as np
import tokenizedataset as td
from functools import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from constants import *
import pickle
import json

MAX_WORDS=64

def create_model(input_shape, hidden_units, unit_type, opt, layers, vocab_size, emb_dimension):
    print("* Creating new model")

    m, max_length = input_shape

    char_input = Input(shape=(max_length,), name="X")
    embed_layer = Embedding(vocab_size, emb_dimension)(char_input)

    if unit_type == "LSTM":
        a = LSTM(hidden_units, return_sequences=True)(embed_layer)

        for i in range(layers - 1):
            a = LSTM(hidden_units, return_sequences=True)(a)
    else:
        a = GRU(hidden_units, return_sequences=True)(embed_layer)

        for i in range(layers - 1):
            a = GRU(hidden_units, return_sequences=True)(a)

    x = TimeDistributed(Dense(128), name='Dense1')(a)
    x = TimeDistributed(Dense(vocab_size + 1, activation='softmax'), name='OutputLayer')(x)

    output = x

    model = Model(inputs=[char_input], outputs=[output])
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=["accuracy"])

    return model


def flatten(sequence):
    return reduce(lambda x, y: x + y, sequence)


def create_dataset_generator(inputs, batch_size, vocab_size):
    while True:
        lines = inputs[np.random.randint(0, len(inputs), batch_size)]
        displaced = np.zeros(lines.shape, dtype=np.int32)
        displaced[:, 1:] = lines[:, :-1]

        lines = to_categorical(lines, vocab_size + 1)

        yield displaced, lines


def load_lines(path):
    datasets = td.load_folder(path)
    lines = [reduce((lambda x, y: x + ' ' +y), word_seq) for word_seq in datasets]

    return lines


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)

    return tokenizer


def read_dataset(lines, tokenizer):
    max_words = MAX_WORDS
    matrix_input = tokenizer.texts_to_sequences(lines)
    np.random.shuffle(matrix_input)
    matrix_input = np.array(pad_sequences(matrix_input, maxlen=max_words, padding='post'))

    return matrix_input, max_words


def saveTokenizer(path, tokenizer):
    with open(path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def loadTokenizer(path):
    with open(path, 'rb') as handle:
        tokenizer = pickle.load(handle)

        return tokenizer

def train_model(options):
    opt = Adam(lr=options.learningRate, beta_1=0.9, beta_2=0.999, decay=0.01)

    lines = load_lines(options.datasetPath)

    if options.load:
        tokenizer = loadTokenizer(options.vocabularyPath)
    else:
        tokenizer = create_tokenizer(lines)
        saveTokenizer(options.vocabularyPath, tokenizer)

    inputs, max_words = read_dataset(lines, tokenizer)
    vocab_length = len(tokenizer.word_counts)

    if options.load:
        model = load_model(options.modelPath)
        model.optimizer = opt
    else:
        input_shape = (options.minibatchSize, max_words)
        model = create_model(input_shape, options.hidden, options.rnnType, opt, options.rnnLayers,
                             vocab_length, options.embedding)

        model.summary()

    data_source = create_dataset_generator(inputs, options.minibatchSize, vocab_length)

    for i in range(options.iterations):
        history = model.fit_generator(data_source, steps_per_epoch=options.batchSize,
                                      epochs=options.epochs, verbose=1)
        save_model(model, history, options)


