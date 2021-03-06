from keras.layers import *
import keras.models as km
from wordmodel import loadTokenizer
import json
import re

MAX_SENTENCE_LENGTH = 30

def extract_model_features(model):
    regex = re.compile(r"\.layers\.recurrent\.(.*)'", re.IGNORECASE)
    rnnLayers = [str(type(layer)) for layer in model.layers if str(type(layer)).find('layers.recurrent') > 0]
    cellType = [regex.findall(layer)[0] for layer in rnnLayers][0]
    _, sequenceSize = model.input.shape

    return len(rnnLayers), cellType, sequenceSize


def load_model(path):
    model = km.load_model(path)

    rnnLayers, cellType, sequenceSize = extract_model_features(model)

    return model, rnnLayers, cellType, sequenceSize


def load_vocabulary(path):
    tokenizer = loadTokenizer(path)

    return tokenizer.word_index


def reverse_vocabulary(vocabulary):
    return {value: key for key, value in vocabulary.items()}


def generate(model, sequenceSize):
    input = np.zeros((1, sequenceSize))
    output_words = []

    stats = []

    for i in range(1, MAX_SENTENCE_LENGTH):
        output = model.predict(input)
        _, _, vocab_size = output.shape
        last_word = np.random.choice(range(1, vocab_size + 1), p=output[0, i - 1])
        stat = {
            'max': max(output[0, i-1]),
            'min': min(output[0, i-1])
        }

        stats.append(stat)

        if last_word <= 0:
            break

        input[0, i] = last_word
        output_words.append(last_word)

    return output_words, stats


def print_full_line():
    full_line = '\n' + 50*'-' + '\n\n'
    print(full_line)


def display_result(vocabulary, final_output, stats):
    reversed_vocabulary = reverse_vocabulary(vocabulary)

    words = [reversed_vocabulary[word_index] for word_index in final_output]
    text = ' '.join(words)

    print('\n\nThe generated text is:\n\n')
    print_full_line()
    print(text)
    print_full_line()

    print('\n\nStats:\n\n')
    print('Max interval: {}\n'.format(max([s['max'] - s['min'] for s in stats])))
    print('Max p: {}\tMin p: {}'.format(max([s['max'] for s in stats]), min([s['min'] for s in stats])))


def generation(options):
    print('Generating...')
    vocabulary = load_vocabulary(options.vocabularyPath)
    model, _, _, sequenceSize =load_model(options.modelPath)
    final_output, stats = generate(model, sequenceSize)
    display_result(vocabulary, final_output, stats)
