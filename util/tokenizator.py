from razdel import tokenize


def tokenize_sentence(sentence):
    """
    Tokenize sentence using razdel library

    @param sentence: string, sentence to tokenize
    """
    tokens = list(tokenize(sentence))
    tokenized = [_.text for _ in tokens]
    return tokenized


def tokenize_sentences(sentences):
    """
    Tokenize sentences using razdel library

    @param sentences: list of strings, sentences to tokenize
    """
    tokenized = [tokenize_sentence(elem) for elem in sentences]
    return tokenized
