from keras.layers import *
from keras.models import Model, load_model
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
import pandas as pd
import numpy as np
import tokenizedataset as td
from functools import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from constants import *
import json


HIDDEN_UNITS=256
UNIT_TYPE='LSTM'
LAYERS=2
EMB_DIMENSION=512
BATCH_SIZE=128
MAX_WORDS=129


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
    x = TimeDistributed(Dense(1), name='OutputLayer')(x)

    model = Model(inputs=[char_input], outputs=[x])
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=["accuracy"])

    return model


def flatten(sequence):
    return reduce(lambda x, y: x + y, sequence)

def read_dataset(path):
    # read the texts
    datasets = td.load_folder(path)

    # extract the vocabulary (Tokenizer)
    tokenizer = Tokenizer()
    lines = [reduce((lambda x, y: x + ' ' +y), word_seq) for word_seq in datasets]
    max_words = MAX_WORDS
    tokenizer.fit_on_texts(lines)
    matrix_input = tokenizer.texts_to_sequences(lines)
    np.random.shuffle(matrix_input)
    matrix_input = np.array(pad_sequences(matrix_input, maxlen=max_words, padding='post'))

    return matrix_input, tokenizer, max_words


def create_dataset_generator(inputs, batch_size, vocab_size):
    while True:
        lines = inputs[np.random.randint(0, len(inputs), batch_size)]
        displaced = np.zeros(lines.shape, dtype=np.int32)
        displaced[:, 1:] = lines[:, :-1]

        new_shape = (lines.shape[0], lines.shape[1], 1)
        lines = np.reshape(lines, new_shape)

        yield displaced, lines


def train_model():
    # Read the datasets
    inputs, tokenizer, max_words  = read_dataset('/home/dani/Documentos/Proyectos/MachineLearning/datasets/Tolkien')
    vocab_length = len(tokenizer.word_counts)

    # Create the model
    input_shape = (BATCH_SIZE, max_words)
    opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=0.01)
    model = create_model(input_shape, HIDDEN_UNITS, UNIT_TYPE, opt, LAYERS, vocab_length, EMB_DIMENSION)

    model.summary()
    data_source = create_dataset_generator(inputs, 256, vocab_length)
    model.fit_generator(data_source, steps_per_epoch=150, epochs=10, verbose=2)


train_model()
