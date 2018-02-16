import tokenizedataset as td
from embeddings import create_embedding_dataset

FEATURE_N=500

def codify_examples(lines, embedding):
    return None


def divide_datasets(examples, division):
    return None


def generatedatasets(lines):
    embedding = create_embedding_dataset(lines)

    #examples = codify_examples(lines, embedding)
    #datasets = divide_datasets(examples, division)

    return embedding


#vocabulary = generatedatasets(td.load_folder('/home/dani/Documentos/Proyectos/MachineLearning/datasets/Tolkien'))



