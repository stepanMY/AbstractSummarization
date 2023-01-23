from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.chrf_score import corpus_chrf
from rouge import Rouge


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


def rouge_score(y_true, y_pred):
    """
    Calculates corpus rouge scores

    @param y_true: list of lists, true tokenized values
    @param y_pred: list of lists, hypothetical tokenized values
    """
    y_true_ = [' '.join(i) for i in y_true]
    y_pred_ = [' '.join(i) for i in y_pred]
    rouge = Rouge()
    scores_dict = rouge.get_scores(y_pred_, y_true_, avg=True)
    return scores_dict


def calc_metrics(y_true, y_pred):
    """
    Calculates corpus metrics

    @param y_true: list of lists, true tokenized values
    @param y_pred: list of lists, hypothetical tokenized values
    """
    metrics = dict()
    metrics['corpus_size'] = len(y_true)
    metrics['ref_example'] = y_true[0]
    metrics['hyp_example'] = y_pred[0]
    metrics['bleu'] = bleu_score(y_true, y_pred)
    metrics['chrf'] = chrf_score(y_true, y_pred)
    rouge_scores = rouge_score(y_true, y_pred)
    metrics['rouge-1-f'] = rouge_scores['rouge-1']['f']
    metrics['rouge-2-f'] = rouge_scores['rouge-2']['f']
    metrics['rouge-l-f'] = rouge_scores['rouge-l']['f']
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
    print('Corpus Size:   ', metrics['corpus_size'])
    print('Ref Example:   ', metrics['ref_example'])
    print('Hyp Example:   ', metrics['hyp_example'])
    print('BLEU:     \t{:3.1f}'.format(metrics['bleu'] * 100.0))
    print('chrF:     \t{:3.1f}'.format(metrics['chrf'] * 100.0))
    print("ROUGE-1-F:\t{:3.1f}".format(metrics['rouge-1-f'] * 100.0))
    print("ROUGE-2-F:\t{:3.1f}".format(metrics['rouge-2-f'] * 100.0))
    print("ROUGE-L-F:\t{:3.1f}".format(metrics['rouge-l-f'] * 100.0))
