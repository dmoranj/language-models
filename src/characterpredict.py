from keras.layers import *
from keras.models import Model, load_model
from charactermodel import load_alphabet
from keras.utils import to_categorical
import pandas as pd
import json


def decode(x, alphabet):
    return alphabet['i2c'][x]


def one_hot_char(index, alphabet):
    ohc = np.zeros((1, alphabet['length']))
    ohc[0, index] = 1
    return ohc

def parse_outputs(outputs, alphabet, options):
    new_outputs = {}

    x = outputs[-1]
    index, out = decode(x, alphabet, options.decodeOption)
    new_outputs['X'] = one_hot_char(index, alphabet)

    hidden_state = outputs[:-1]

    if options.rnnType == 'LSTM':
        for key, val in enumerate(hidden_state):
            if key % 2 == 0:
                name = 'a' + str(key//2)
            else:
                name = 'c' + str(key//2)

            new_outputs[name] = val
    else:
        for key, val in enumerate(hidden_state):
            new_outputs['c' + str(key)] = val

    return new_outputs, out, x


def reshape_current_state(char_length, inputs, options):
    for key in inputs.keys():
        if key.startswith('a') or key.startswith('c'):
            inputs[key] = np.reshape(inputs[key], (1, 1, options.hidden))

    inputs['X'] = np.reshape(inputs['X'], (1, 1, char_length))
    return inputs


def create_initial_states(sequence_size, char_length):
    input = np.zeros((1, sequence_size, char_length))

    return input


def generate(options):
    model = load_model(options.modelPath)

    _, seq_length, char_length = model.input.shape

    alphabet = load_alphabet(options.alphabetPath)

    input = create_initial_states(seq_length, char_length)

    maximums = []
    minimums = []
    sentence = []
    final_output = []

    for j in range(2):
        for i in range(1, 100):
            if i % 50 == 0:
                print('Iteration: {}'.format(i))

            output = model.predict(input)

            index = np.random.choice(range(1, char_length + 1), p=output[0, i - 1, :])
            maximums.append(np.amax(output[0, i - 1, :]))
            minimums.append(np.amin(output[0, i - 1, :]))

            last_char = decode(index, alphabet)
            input[0, i, :] = to_categorical(index, char_length)
            sentence.append(last_char)

        final_output.append(''.join(sentence))
        sentence = []

    stats = {
        'maximum': maximums,
        'minimum': minimums
    }

    return alphabet, '\n'.join(final_output), stats


def generation(options):
    alphabet, final_output, stats = generate(options)
    display_result(alphabet, final_output, stats)


def display_result(alphabet, final_output, stats):
    print('Your author says:\n\n')
    print(final_output)

    maximums = stats['maximum']
    minimums = stats['minimum']

    global_max = max(maximums)
    global_min = min(minimums)

    print('\nMaximum probability= {}'.format(global_max))
    print('\nMinimum probability= {}'.format(global_min))
    print('\nProbability interval size: {}'.format(global_max - global_min))
    print('\nAlphabet length= {}'.format(alphabet['length']))
    print('\nNon-weigthed base probability= {}'.format(1 / alphabet['length']))


