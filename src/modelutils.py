import pandas as pd


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

