from keras.layers import *
import keras.models as km
import json
import re


def extract_model_features(model):
    regex = re.compile(r"\.layers\.recurrent\.(.*)'", re.IGNORECASE)
    rnnLayers = [str(type(layer)) for layer in model.layers if str(type(layer)).find('layers.recurrent') > 0]
    cellType = [regex.findall(layer)[0] for layer in rnnLayers][0]
    _, sequenceSize = model.input

    return len(rnnLayers), cellType, sequenceSize


def load_model(path):
    model = km.load_model(path)

    rnnLayers, cellType, sequenceSize = extract_model_features(model)

    return model, rnnLayers, cellType, sequenceSize


def load_vocabulary(path):
    with open(path, 'r') as f:
        text = f.read()
        vocabulary = json.loads(text)

        return vocabulary


def generate(model, vocabulary, sequenceSize, options):
    print('generating')





def display_result():
    print('displaying')


def generation(options):
    print('Generating...')
    vocabulary = load_vocabulary(options.vocabularyPath)
    model, _, _, sequenceSize =load_model(options.modelPath)
    final_output, stats = generate(model, vocabulary, sequenceSize, options)
    display_result(vocabulary, final_output, stats)
