from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.chrf_score import corpus_chrf
from rouge import Rouge
from collections import Counter
from meteor import Meteor


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


def meteor_score(y_true, y_pred, meteor_jar, language='ru'):
    """
    Calculates corpus meteor scores

    @param y_true: list of lists, true tokenized values
    @param y_pred: list of lists, hypothetical tokenized values
    @param meteor_jar: .jar file name, synonyms and stemming for calculation
    @param language: string, language name; 'ru' for russian
    """
    y_true_ = [[' '.join(i)] for i in y_true]
    y_pred_ = [' '.join(i) for i in y_pred]
    meteor = Meteor(meteor_jar, language=language)
    scores = meteor.compute_score(y_pred_, y_true_)
    return scores


def calc_duplicate_n_grams_rate(y_pred):
    """
    Calculates weighted n-grams duplicate rate in references

    @param y_pred: list of lists, true tokenized values
    """
    all_ngrams_count = Counter()
    duplicate_ngrams_count = Counter()
    for doc in y_pred:
        for n in range(1, 5):
            ngrams = [tuple(doc[i:i + n]) for i in range(len(doc) - n + 1)]
            unique_ngrams = set(ngrams)
            all_ngrams_count[n] += len(ngrams)
            duplicate_ngrams_count[n] += len(ngrams) - len(unique_ngrams)
    scores_dict = {n: duplicate_ngrams_count[n] / all_ngrams_count[n]
                   if all_ngrams_count[n] else 0.0 for n in range(1, 5)}
    return scores_dict


def calc_metrics(y_true, y_pred, meteor_jar):
    """
    Calculates corpus metrics

    @param y_true: list of lists, true tokenized values
    @param y_pred: list of lists, hypothetical tokenized values
    @param meteor_jar: .jar file name, synonyms and stemming for calculation
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
    met_score = meteor_score(y_true, y_pred, meteor_jar)
    metrics['meteor'] = met_score
    dup_scores = calc_duplicate_n_grams_rate(y_pred)
    metrics['duplicate_ngrams'] = dup_scores
    lengths = [len(i) for i in y_pred]
    metrics['avg_length'] = sum(lengths) / len(lengths)
    return metrics


def print_corp_metrics(y_true, y_pred, meteor_jar):
    """
    Prints corpus metrics in percents

    @param y_true: list of lists, true tokenized values
    @param y_pred: list of lists, hypothetical tokenized values
    @param meteor_jar: .jar file name, synonyms and stemming for calculation
    """
    delim = '-------------METRICS-------------'
    print(delim)
    metrics = calc_metrics(y_true, y_pred, meteor_jar)
    print('Corpus Size:   ', metrics['corpus_size'])
    print('Ref Example:   ', metrics['ref_example'])
    print('Hyp Example:   ', metrics['hyp_example'])
    print('BLEU:     \t{:3.1f}'.format(metrics['bleu'] * 100.0))
    print('chrF:     \t{:3.1f}'.format(metrics['chrf'] * 100.0))
    print('ROUGE-1-F:\t{:3.1f}'.format(metrics['rouge-1-f'] * 100.0))
    print('ROUGE-2-F:\t{:3.1f}'.format(metrics['rouge-2-f'] * 100.0))
    print('ROUGE-L-F:\t{:3.1f}'.format(metrics['rouge-l-f'] * 100.0))
    print('METEOR:   \t{:3.1f}'.format(metrics['meteor'] * 100.0))
    print('Dup 1-grams:\t{:3.1f}'.format(metrics['duplicate_ngrams'][1] * 100.0))
    print('Dup 2-grams:\t{:3.1f}'.format(metrics['duplicate_ngrams'][2] * 100.0))
    print('Dup 3-grams:\t{:3.1f}'.format(metrics['duplicate_ngrams'][3] * 100.0))
    print('Avg length:\t{:3.1f}'.format(metrics['avg_length']))
