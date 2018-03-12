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
from constants import *
import json

MAX_WORDS=128

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

    output = Reshape((input_shape[1],))(x)

    model = Model(inputs=[char_input], outputs=[output])
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


def create_dataset_generator(inputs, batch_size):
    while True:
        lines = inputs[np.random.randint(0, len(inputs), batch_size)]
        displaced = np.zeros(lines.shape, dtype=np.int32)
        displaced[:, 1:] = lines[:, :-1]

        lines = lines /15000

        yield displaced, lines


def saveTokenizer(path, tokenizer):
    with open(path, 'w') as f:
        json.dump(tokenizer.word_index, f)


def train_model(options):
    inputs, tokenizer, max_words = read_dataset(options.datasetPath)
    vocab_length = len(tokenizer.word_counts)
    input_shape = (options.minibatchSize, max_words)
    opt = Adam(lr=options.learningRate, beta_1=0.9, beta_2=0.999, decay=0.01)
    saveTokenizer(options.vocabularyPath, tokenizer)

    if options.load:
        model = load_model(options.modelPath)
        model.optimizer = opt
    else:
        model = create_model(input_shape, options.hidden, options.rnnType, opt, options.rnnLayers,
                             vocab_length, options.embedding)
        model.summary()


    data_source = create_dataset_generator(inputs, options.minibatchSize)

    for i in range(options.iterations):
        history = model.fit_generator(data_source, steps_per_epoch=options.batchSize,
                                      epochs=options.epochs, verbose=1)
        save_model(model, history, options)


