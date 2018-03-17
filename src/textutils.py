import json


def load_alphabet(path):
    with open(path, 'r') as f:
        text = f.read()
        alphabet = json.loads(text)

        return alphabet



def save_alphabet(path, alphabet):
    with open(path, 'w') as f:
        json.dump(alphabet, f)

