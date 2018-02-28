from keras.layers import *
from keras.models import Model, load_model
from charactermodel import load_alphabet
import pandas as pd
import json


def create_prediction_model(model, rnn_type, layers):
    _, seq_length, char_length = model.input.shape

    input_char = Input((1, char_length.value), name='X')

    rnn_cell = model.layers[1].cell
    dense_layer = model.layers[layers + 1].layer
    activation_layer = model.layers[layers + 2].layer

    c0 = Input((1, rnn_cell.units), name='c0')

    inputs = [input_char]
    outputs = []

    if rnn_type == 'LSTM':
        a0 = Input((1, rnn_cell.units), name='a0')
        x, a, c = RNN(rnn_cell, return_state=True, return_sequences=False)(input_char, initial_state=[a0, c0])
        inputs.append(c0)
        outputs.append(a)
        outputs.append(c)

        for i in range(layers - 1):
            rnn_cell = model.layers[2 + i].cell
            x = Reshape((1, x.shape[2].value))(x)
            a0 = Input((1, rnn_cell.units), name='a' + str(i + 1))
            c0 = Input((1, rnn_cell.units), name='c' + str(i + 1))
            inputs.append(a0)
            inputs.append(c0)
            x, a, c = RNN(rnn_cell, return_state=True, return_sequences=False)(x, initial_state=[a0, c0])
            outputs.append(a)
            outputs.append(c)

    else:
        x, c = RNN(rnn_cell, return_state=True, return_sequences=False)(input_char, initial_state=[c0])
        inputs.append(c0)
        outputs.append(c)

        for i in range(layers - 1):
            rnn_cell = model.layers[2 + i].cell
            x = Reshape((1, x.shape[2].value))(x)
            c0 = Input((1, rnn_cell.units), name='c' + str(i + 1))
            inputs.append(c0)
            x, c = RNN(rnn_cell, return_state=True, return_sequences=False)(x, initial_state=[c0])
            outputs.append(c)

    x = dense_layer(x)
    x = activation_layer(x)

    outputs.append(x)

    new_model = Model(inputs=inputs, outputs=outputs)

    return new_model


def decode(x, alphabet, option):
    p = x.tolist()[0]

    if option == 'choice':
        index = np.random.choice(range(len(p)), p=p/np.sum(p))
    else:
        index = np.argmax(p)

    return index, alphabet['i2c'][index]


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


def create_initial_states(options, char_length):
    initial_states = {}
    for i in range(options.rnnLayers):
        if options.rnnType == 'LSTM':
            a0 = np.zeros((1, options.hidden))
            initial_states['a' + str(i)] = a0

        c0 = np.zeros((1, options.hidden))
        initial_states['c' + str(i)] = c0

    initial_states['X'] = np.zeros((1, char_length))

    return initial_states


def generate(options):
    model = load_model(options.modelPath)

    _, seq_length, char_length = model.input.shape

    prediction_model = create_prediction_model(model, options.rnnType, options.rnnLayers)
    alphabet = load_alphabet(options.alphabetPath)

    initial_states = create_initial_states(options, char_length)

    global_sum = initial_states['X']
    output = []
    maximums = []
    minimums = []
    sentence = []

    for j in range(10):
        inputs = initial_states.copy()

        for i in range(options.maxLength):
            inputs = reshape_current_state(char_length, inputs, options)
            outputs = prediction_model.predict(inputs)
            inputs, out, x = parse_outputs(outputs, alphabet, options)

            sentence.append(out)

            if out == '.':
                output.append(''.join(sentence))
                sentence = []
                break

            maximums.append(np.max(x))
            minimums.append(np.min(x))
            global_sum += x

        if len(sentence) > 0:
            output.append(''.join(sentence) + '.')

    final_output = '\n'.join(output)

    stats = {
        'maximum': maximums,
        'minimum': minimums,
        'globalSum': global_sum
    }

    return alphabet, final_output, stats


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

    total_weight = np.sum(stats['globalSum'])

    prob_df = pd.DataFrame({'char': alphabet['i2c'], 'p': stats['globalSum'][0]/total_weight})
    prob_df = prob_df.sort_values(by='p', ascending=False)

    print('\nMaximum probability= {}'.format(global_max))
    print('\nMinimum probability= {}'.format(global_min))
    print('\nProbability interval size: {}'.format(global_max - global_min))
    print('\nAlphabet length= {}'.format(alphabet['length']))
    print('\nNon-weigthed base probability= {}'.format(1 / alphabet['length']))
    print(prob_df)


