from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import regularizers
from modelutils import save_model
import numpy as np
import tokenizedataset as td
from constants import *
from textutils import load_alphabet, save_alphabet
import characterpredict as cp

ALPHABET_CHARS = "abcdefghijklmnopqrstuvwxyz?!0123456789;:,.()-'\""


def create_model(input_shape, hidden_units, dense_hidden, unit_type, opt, layers, dropout, l1, l2):
    print("* Creating new model")

    _, max_length, char_length = input_shape

    char_input = Input(shape=(max_length, char_length), name="X")

    if unit_type == "LSTM":
        a = LSTM(hidden_units, return_sequences=True, kernel_regularizer=regularizers.l2(l2),
                 activity_regularizer=regularizers.l1(l1), dropout=dropout)(char_input)

        for i in range(layers - 1):
            a = LSTM(hidden_units, return_sequences=True, kernel_regularizer=regularizers.l2(l2),
                     activity_regularizer=regularizers.l1(l1), dropout=dropout)(a)
    else:
        a = GRU(hidden_units, return_sequences=True, kernel_regularizer=regularizers.l2(l2),
                activity_regularizer=regularizers.l1(l1), dropout=dropout)(char_input)

        for i in range(layers - 1):
            a = GRU(hidden_units, return_sequences=True, kernel_regularizer=regularizers.l2(l2),
                    activity_regularizer=regularizers.l1(l1), dropout=dropout)(a)

    x = TimeDistributed(Dense(dense_hidden, activation='relu', kernel_regularizer=regularizers.l2(l2),
                              activity_regularizer=regularizers.l1(l1)))(a)

    x = TimeDistributed(Dense(char_length, kernel_regularizer=regularizers.l2(l2)))(x)
    x = TimeDistributed(Activation('softmax'))(x)

    model = Model(inputs=[char_input], outputs=[x])
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    return model


def prepaire_lines(lines):
    for line in lines:
        if line[-1] != '.':
            line.append('.')

    return lines


def onehot_line(line, alphabet, options):
    full_line = ' '.join(line)

    line = np.zeros((options.maxLength, alphabet['length']))

    for i, c in enumerate(full_line):
        if i >= options.maxLength:
            break

        if c not in alphabet['c2i'].keys():
            c = '#'

        index = alphabet['c2i'][c]
        line[i, index] = 1

    return line


def create_alphabet():
    charset = set()

    [charset.add(c) for c in ALPHABET_CHARS]
    charset.add(' ')
    charset.add('#')
    charlist = list(charset)

    charlist = ['@'] + charlist

    alphabet = {
        "length": len(charlist),
        "i2c": charlist,
        "c2i": {c: i for i, c in enumerate(charlist)}
    }

    return alphabet


def convert_to_onehot(prepaired_lines, alphabet, options):
    oh_lines = [onehot_line(line, alphabet, options) for line in prepaired_lines]
    line_array = np.array(oh_lines)
    np.random.shuffle(line_array)

    return line_array


def initialize_dataset(lines):
    x = np.zeros(lines.shape)
    x[:, 1:, :] = lines[:, :-1, :]

    return {
        'X': x,
        'Y': lines
    }


def generate_datasets(lines, alphabet, options):
    prepaired_lines = prepaire_lines(lines)
    lines_oh = convert_to_onehot(prepaired_lines, alphabet, options)
    final_lines = initialize_dataset(lines_oh)

    return final_lines


def dataset_generator(alphabet, options):
    while True:
        for lines in td.generate_line_batch(options.datasetPath, options.minibatchSize):
            train = generate_datasets(lines, alphabet, options)
            yield (train['X'], train['Y'])


def train_model(options):
    opt = Adam(lr=options.learningRate, beta_1=0.9, beta_2=0.999, decay=0.01, clipnorm=1.0)

    if options.load:
        alphabet = load_alphabet(options.alphabetPath)
        model = load_model(options.modelPath)
        model.optimizer = opt
    else:
        alphabet = create_alphabet()
        model = create_model((options.batchSize, options.maxLength, alphabet['length']), options.hidden,
                             options.dense, options.rnnType, opt, options.rnnLayers, options.dropout,
                             options.l2, options.l1)
        model.summary()

        save_alphabet(options.alphabetPath, alphabet)

    data_source = dataset_generator(alphabet, options)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    for i in range(options.iterations):
        history = model.fit_generator(data_source, steps_per_epoch=options.batchSize,
                                      epochs=options.epochs, verbose=1)
        save_model(model, history, options)

        cp.generation(options)


