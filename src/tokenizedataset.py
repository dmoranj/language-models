import re
import os
import numpy as np
import glob

MIN_TOKEN_LENGTH = 3

def clean_text(text):
    cleaned = text.replace('\n', '')\
        .replace('\xe2\x80', '')\
        .replace('\x94', '') \
        .replace('\x9c', '') \
        .replace('\x99', '')\
        .replace('\x9d', '')\
        .replace('Mr.', 'Mr')\
        .replace('Mrs.', 'Mrs')\
        .replace('?', ' ?')\
        .replace('!', ' !')\
        .replace(' -', '-')

    stripped = re.sub(r'\s+', ' ', cleaned.strip())
    clean_linebreaks = re.sub(r'-\s+', '', stripped)
    without_chapters = re.sub(r'(Chapter|CHAPTER)\s([A-Z]+\s)+', '', clean_linebreaks)

    return without_chapters


def split_sentences(fulltext):
    return [line.split() for line in fulltext.split('.')]


def tokenize(text):
    text = clean_text(text)
    tokens = [token for token in split_sentences(text) if len(token) > MIN_TOKEN_LENGTH]
    np.random.shuffle(tokens)
    word_list = [[word.lower() for word in line] for line in tokens]

    return word_list


def load_and_tokenize(textpath):
    with open(textpath, 'r') as t:
        text = t.read()
        tokens = tokenize(text)
        return tokens


def load_folder(folder):
    files = glob.glob(os.path.join(folder, "*.txt"))

    all_tokens = []

    for file in files:
        print('Loading text from: ' + file)

        textpath = os.path.join(folder, file)
        tokens = load_and_tokenize(textpath)

        all_tokens.extend(tokens)

    return all_tokens


def generate_line_batch(input_folder, max_lines):
    files = glob.glob(os.path.join(input_folder, "*.txt"))
    np.random.shuffle(files)

    current_batch = []

    for file in files:
        textpath = os.path.join(input_folder, file)
        tokens = []
        tokens = load_and_tokenize(textpath)
        token_index = 0

        while token_index < len(tokens):
            current_batch.append(tokens[token_index])

            if len(current_batch) >= max_lines:
                np.random.shuffle(current_batch)
                yield current_batch
                current_batch = []

            token_index += 1


    if len(current_batch) < max_lines:
        yield current_batch
