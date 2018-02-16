from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import tokenizedataset as td
from constants import *
import json

GRU_HIDDEN_UNITS = 128
MAX_LENGTH = 750
SAVE_PATH = './results/'
MODEL_FILE = SAVE_PATH + 'charmodel.h5'
ALPHABET_FILE = SAVE_PATH + 'alphabet.json'

def create_model(input_shape, hidden_units):
    print("* Creating model")
    
    _, max_length, char_length = input_shape

    char_input = Input(shape=(max_length, char_length), name="X")

    a = LSTM(hidden_units, return_sequences=True)(char_input)
    x = TimeDistributed(Dense(char_length))(a)
    x = TimeDistributed(Activation('softmax'))(x)

    model = Model(inputs=[char_input], outputs=[x])

    return model


def fit_model(model, inputs, Y, batch_size, epochs):
    print('Fitting model with summary: ')
    model.summary()
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    model.fit(inputs, Y, batch_size=batch_size, epochs=epochs)


def prepaire_lines(lines):
    for line in lines:
        if line[-1] != '.':
            line.append('.')

    return lines


def onehot_line(line, alphabet):
    full_line = ' '.join(line)

    line = np.zeros((MAX_LENGTH, alphabet['length']))

    for i, c in enumerate(full_line):
        if i >= MAX_LENGTH:
            break

        index = alphabet['c2i'][c]
        line[i, index] = 1

    return line


def get_alphabet(lines):
    charset = set()

    for line in lines:
        for word in line:
            [charset.add(c) for c in word]

    charset.add(' ')
    charlist = list(charset)

    alphabet = {
        "length": len(charlist),
        "i2c": charlist,
        "c2i": {c: i for i, c in enumerate(charlist)}
    }

    return alphabet


def convert_to_onehot(prepaired_lines):
    alphabet = get_alphabet(prepaired_lines)

    oh_lines = [onehot_line(line, alphabet) for line in prepaired_lines]
    line_array = np.array(oh_lines)
    np.random.shuffle(line_array)

    return line_array, alphabet


def initialize_dataset(lines):
    x = np.zeros(lines.shape)
    x[:, 1:, :] = lines[:, :-1, :]

    return {
        'X': x,
        'Y': lines
    }


def generate_datasets(lines, hidden_units):
    print('* Generating datasets')
    prepaired_lines = prepaire_lines(lines)
    lines_oh, alphabet = convert_to_onehot(prepaired_lines)
    final_lines = initialize_dataset(lines_oh)

    return final_lines, alphabet


def save_alphabet(path, alphabet):
    with open(path, 'w') as f:
        json.dump(alphabet, f)


def train_model(input_path, hidden_units, max_examples, batch_size, epochs):
    lines = td.load_folder(input_path)[0:max_examples]
    train, alphabet = generate_datasets(lines, hidden_units)
    model = create_model(train['X'].shape, hidden_units)
    fit_model(model, {"X": train['X']}, train['Y'], batch_size, epochs)
    model.save(MODEL_FILE)
    save_alphabet(ALPHABET_FILE, alphabet)


def command_train():
    train_model('/home/dani/Documentos/Proyectos/MachineLearning/datasets/Tolkien', GRU_HIDDEN_UNITS, 12000, 256, 50)


def load_alphabet(path):
    with open(path, 'r') as f:
        text = f.read()
        alphabet = json.loads(text)

        return alphabet


def create_prediction_model(model):
    _, seq_length, char_length = model.input.shape

    input_char = Input((1, char_length.value), name='X')

    rnn_cell = model.layers[1].cell
    dense_layer = model.layers[2].layer
    activation_layer = model.layers[3].layer

    a0 = Input((1, rnn_cell.units), name='a0')
    c0 = Input((1, rnn_cell.units), name='c0')

    x, a, c = RNN(rnn_cell, return_state=True, return_sequences=False)(input_char, initial_state=[a0, c0])
    x = dense_layer(x)
    x = activation_layer(x)

    new_model = Model(inputs=[input_char, a0, c0], outputs=[x, a, c])

    return new_model


def decode(x, alphabet):
    draw = np.random.uniform()

    accumulator = 0
    index = -1

    for i, value in enumerate(x.tolist()[0]):
        accumulator += value

        if draw < accumulator:
            index = i
            break

    if index >= 0:
        return index, alphabet['i2c'][index]
    else:
        return -1, '#'


def one_hot_char(index, alphabet):
    ohc = np.zeros((1, alphabet['length']))
    ohc[0, index] = 1
    return ohc


def command_generate():
    model = load_model(MODEL_FILE)

    _, seq_length, char_length = model.input.shape

    prediction_model = create_prediction_model(model)
    alphabet = load_alphabet(ALPHABET_FILE)

    a = np.zeros((1, GRU_HIDDEN_UNITS))
    c = np.zeros((1, GRU_HIDDEN_UNITS))
    x = np.zeros((1, char_length))

    output = []
    maximums = []

    for i in range(MAX_LENGTH):
        a = np.reshape(a, (1, 1, GRU_HIDDEN_UNITS))
        c = np.reshape(c, (1, 1, GRU_HIDDEN_UNITS))
        x = np.reshape(x, (1, 1, char_length))

        x, a, c = prediction_model.predict([x, a, c])

        index, out = decode(x, alphabet)
        output.append(out)
        maximums.append(np.max(x))

        x = one_hot_char(index, alphabet)

    print(''.join(output))
    print(maximums)
    print('Maximum probability= {}'.format(max(maximums)))
    print('Alphabet length= {}'.format(alphabet['length']))
    print('Non-weigthed base probability= {}'.format(1/alphabet['length']))


command_generate()