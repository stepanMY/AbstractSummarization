from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.chrf_score import corpus_chrf


def bleu_score(y_true, y_pred, smoothing=True):
    """
    Calculates corpus bleu score

    @param y_true: list of lists, true tokenized values
    @param y_pred: list of lists, hypothetical tokenized values
    @param smoothing: bool, whether or not use smoothing
    """
    y_true_ = [[elem] for elem in y_true]
    if smoothing:
        smoothie = SmoothingFunction().method1
        score = corpus_bleu(y_true_, y_pred, smoothing_function=smoothie)
    else:
        score = corpus_bleu(y_true_, y_pred)
    return score


def chrf_score(y_true, y_pred, beta=3.0):
    """
    Calculates corpus chrf score

    @param y_true: list of lists, true tokenized values
    @param y_pred: list of lists, hypothetical tokenized values
    @param beta: float, value of beta to use when calculating F-score
    """
    score = corpus_chrf(y_true, y_pred, beta=beta)
    return score


def calc_metrics(y_true, y_pred):
    """
    Calculates corpus metrics

    @param y_true: list of lists, true tokenized values
    @param y_pred: list of lists, hypothetical tokenized values
    """
    metrics = dict()
    metrics['bleu'] = bleu_score(y_true, y_pred)
    metrics['chrf'] = chrf_score(y_true, y_pred)

    return metrics


def print_corp_metrics(y_true, y_pred):
    """
    Prints corpus metrics in percents

    @param y_true: list of lists, true tokenized values
    @param y_pred: list of lists, hypothetical tokenized values
    """
    delim = '-------------METRICS-------------'
    print(delim)
    metrics = calc_metrics(y_true, y_pred)
    print('BLEU:     \t{:3.1f}'.format(metrics['bleu'] * 100.0))
    print('chrF:     \t{:3.1f}'.format(metrics['chrf'] * 100.0))
