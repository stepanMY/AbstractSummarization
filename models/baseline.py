import numpy as np
from util.tokenizator import tokenize_sentence


class NotFittedError(Exception):
    pass


class FirstWorder:
    """
    Class that uses first n words of the abstract to make a prediction
    """
    def __init__(self, mode, n_words=0):
        """
        @param mode: string, 'mean' or 'median' or 'n'
        @param n_words: int, number of words to extract. Used only if mode == 'n'
        """
        if mode in {'mean', 'median', 'n'}:
            self.mode = mode
        else:
            raise ValueError('mode should be like mean/median/n')
        if isinstance(n_words, int):
            self.n_words = n_words
        else:
            raise TypeError('n_words should be int type')
        if mode == 'n':
            self.optimal_length = n_words
        else:
            self.optimal_length = None

    def fit(self, x, y):
        """
        Fit the model

        @param x: list of strings, abstracts
        @param y: list of strings, titles
        """
        lengths = [len(tokenize_sentence(elem)) for elem in y]
        if self.mode == 'mean':
            self.optimal_length = int(np.mean(lengths))
        elif self.mode == 'median':
            self.optimal_length = int(np.median(lengths))

    def predict(self, x):
        """
        Predict titles of given x abstracts, returns list of lists (tokenized)

        @param x: list of strings, abstracts
        """
        if self.optimal_length is None:
            raise NotFittedError('model was not fitted')
        predictions = []
        for elem in x:
            predictions.append(tokenize_sentence(elem)[:self.optimal_length])
        return predictions
