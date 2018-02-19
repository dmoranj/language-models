from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import Adam
import pandas as pd
import numpy as np
import tokenizedataset as td
from constants import *
import json

ALPHABET_CHARS = "abcdefghijklmnopqrstuvwxyz?!0123456789;:,.()-_'\""

def create_model(input_shape, hidden_units):
    print("* Creating model")

    _, max_length, char_length = input_shape

    char_input = Input(shape=(max_length, char_length), name="X")

    a = LSTM(hidden_units, return_sequences=True)(char_input)
    x = TimeDistributed(Dense(char_length))(a)
    x = TimeDistributed(Activation('softmax'))(x)

    model = Model(inputs=[char_input], outputs=[x])

    return model


def fit_model(model, inputs, Y, batch_size, epochs, learning_rate):
    print('Fitting model with summary: ')
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    history = model.fit(inputs, Y, batch_size=batch_size, epochs=epochs, verbose=2)
    return history

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

        if not c in alphabet['c2i'].keys():
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
    print('* Generating datasets')
    prepaired_lines = prepaire_lines(lines)
    lines_oh = convert_to_onehot(prepaired_lines, alphabet, options)
    final_lines = initialize_dataset(lines_oh)

    return final_lines


def save_alphabet(path, alphabet):
    with open(path, 'w') as f:
        json.dump(alphabet, f)


def save_model(model, history, options):
    model.save(options.modelPath)

    epochs = {
        'accuracy': history.history['acc'],
        'loss': history.history['loss']
    }

    df = pd.DataFrame(epochs)
    df['rate'] = options.learningRate
    df['minibatch'] = options.minibatchSize

    df.to_csv(options.statsPath, mode='a', header=False)


def train_model(options):
    if options.load:
        alphabet = load_alphabet(options.alphabetPath)
        model = load_model(options.modelPath)
    else:
        alphabet = create_alphabet()
        model = create_model((options.batchSize, options.maxLength, alphabet.length), options.hidden)
        model.summary()

    save_alphabet(options.alphabetPath, alphabet)

    for i in range(options.iterations):
        for lines in td.generate_line_batch(options.datasetPath, options.batchSize):
            train = generate_datasets(lines, alphabet, options)
            history = fit_model(model, {"X": train['X']}, train['Y'], options.minibatchSize,
                                options.epochs, options.learningRate)
            save_model(model, history, options)


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
    p = x.tolist()[0]
    index = np.random.choice(range(len(p)), p=p/np.sum(p))

    return index, alphabet['i2c'][index]


def one_hot_char(index, alphabet):
    ohc = np.zeros((1, alphabet['length']))
    ohc[0, index] = 1
    return ohc


def generate(options):
    model = load_model(options.modelPath)

    _, seq_length, char_length = model.input.shape

    prediction_model = create_prediction_model(model)
    alphabet = load_alphabet(options.alphabetPath)

    a0 = np.zeros((1, options.hidden))
    c0 = np.zeros((1, options.hidden))
    x0 = np.zeros((1, char_length))

    output = []
    maximums = []

    for j in range(10):
        sentence = []

        a = a0
        c = c0
        x = x0

        for i in range(options.maxLength):
            a = np.reshape(a, (1, 1, options.hidden))
            c = np.reshape(c, (1, 1, options.hidden))
            x = np.reshape(x, (1, 1, char_length))

            x, a, c = prediction_model.predict([x, a, c])

            index, out = decode(x, alphabet)
            sentence.append(out)

            if out == '.':
                output.append(''.join(sentence))
                break

            maximums.append(np.max(x))

            x = one_hot_char(index, alphabet)

    print('Your author says:\n\n')
    print('\n'.join(output))
    print('\nMaximum probability= {}'.format(max(maximums)))
    print('Alphabet length= {}'.format(alphabet['length']))
    print('Non-weigthed base probability= {}'.format(1/alphabet['length']))


