import numpy as np
import time
from constants import *
import os

import tensorflow as tf

K = 4
PREFIX = 'emb_'
SUFFIX = '.csv'

def get_vocabulary(lines):
    vocabulary = set()

    for line in lines:
        for word in line:
            vocabulary.add(word.lower().strip())

    return list(vocabulary)


def one_hot(word_id, encoding_bits):
    representation = np.zeros((encoding_bits, 1))
    representation[word_id, 0] = 1

    return representation


def process_vocabulary(vocabulary):
    encoding_bits = len(vocabulary)

    vocabulary_map = {
        "i2w": vocabulary,
        "w2i": {value:key for key, value in enumerate(vocabulary)},
    }

    vocabulary_map["w2o"] = {value: one_hot(vocabulary_map['w2i'][value], encoding_bits) for value in vocabulary}

    return vocabulary_map


def create_model():
    return None


def get_neighbour(index):
    if index != 0:
        return np.round(np.random.lognormal(np.log(index), np.log(2))).astype(np.int32)
    else:
        return np.ceil(np.random.exponential()).astype(np.int32)


def draw_word(lines):
    line = lines[np.random.randint(0, len(lines))]
    target_index = np.random.randint(0, len(line))

    return target_index, line


def create_examples(lines, number, vocabulary):
    print('Creating {} samples in {} lines'.format(number, len(lines)))

    start = time.time()
    examples = []

    positive = np.reshape(np.array([1]), (1, 1))
    negative = np.reshape(np.array([0]), (1, 1))

    for i in range(number):
        target_index, line = draw_word(lines)
        context = get_neighbour(target_index)

        while context < 0 or context >= len(line) or context == target_index:
            context = get_neighbour(target_index)

        one_hot_target = vocabulary['w2o'][line[target_index]]

        for i in range(K):
            negative_index, negative_line = draw_word(lines)
            one_hot_negative_context = vocabulary['w2o'][negative_line[negative_index]]
            examples.append(np.concatenate((one_hot_target, one_hot_negative_context, negative)))

        examples.append(np.concatenate((one_hot_target, vocabulary['w2o'][line[context]], positive)))

    end = time.time()

    print('Finished in {}s'.format(end - start))

    return examples


def divide_by_ratios(dataset, ratios):
    dataset = np.array(dataset)
    dataset = np.reshape(dataset, dataset.shape[:-1])
    np.random.shuffle(dataset)
    rows = len(dataset)
    results = []

    lastIndex = 0

    for ratio in ratios:
        nextIndex = int(min(lastIndex + ratio*rows, rows))
        sliced = dataset[lastIndex:nextIndex]
        results.append(sliced)
        lastIndex = nextIndex

    return results


def save_dataset(dataset, names):
    for id, name in enumerate(names):
        file_path = os.path.join(OUTPUT_FOLDER, PREFIX + name + SUFFIX)
        np.savetxt(file_path, dataset[id], delimiter=',')



def create_embedding_dataset(lines):
    num_examples = 1000

    vocabulary = get_vocabulary(lines)
    vocabulary = process_vocabulary(vocabulary)

    examples = create_examples(lines, num_examples, vocabulary)
    dataset = divide_by_ratios(examples, DIVISION_RATES)

    save_dataset(dataset, DIVISION_NAMES)
