from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.chrf_score import corpus_chrf
from rouge.rouge import rouge_n_sentence_level, rouge_l_sentence_level
from collections import Counter


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
    scores_dict = dict()
    rouge1_corp, rouge2_corp, rouge3_corp, rougel_corp = 0, 0, 0, 0
    for i in range(len(y_true)):
        true, pred = y_true[i], y_pred[i]
        *_, rouge1 = rouge_n_sentence_level(pred, true, 1)
        *_, rouge2 = rouge_n_sentence_level(pred, true, 2)
        *_, rouge3 = rouge_n_sentence_level(pred, true, 3)
        *_, rougel = rouge_l_sentence_level(pred, true)
        rouge1_corp += rouge1
        rouge2_corp += rouge2
        rouge3_corp += rouge3
        rougel_corp += rougel
    scores_dict['rouge-1-f'] = rouge1_corp/len(y_true)
    scores_dict['rouge-2-f'] = rouge2_corp/len(y_true)
    scores_dict['rouge-3-f'] = rouge3_corp/len(y_true)
    scores_dict['rouge-l-f'] = rougel_corp/len(y_true)
    return scores_dict


def calc_duplicate_n_grams_rate(sentences):
    """
    Calculates weighted n-grams duplicate rate in references

    @param sentences: list of lists, tokenized sentences
    """
    all_ngrams_count = Counter()
    duplicate_ngrams_count = Counter()
    for doc in sentences:
        for n in range(1, 5):
            ngrams = [tuple(doc[i:i + n]) for i in range(len(doc) - n + 1)]
            unique_ngrams = set(ngrams)
            all_ngrams_count[n] += len(ngrams)
            duplicate_ngrams_count[n] += len(ngrams) - len(unique_ngrams)
    scores_dict = {n: duplicate_ngrams_count[n] / all_ngrams_count[n]
                   if all_ngrams_count[n] else 0.0 for n in range(1, 5)}
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
    metrics['rouge-1-f'] = rouge_scores['rouge-1-f']
    metrics['rouge-2-f'] = rouge_scores['rouge-2-f']
    metrics['rouge-3-f'] = rouge_scores['rouge-3-f']
    metrics['rouge-l-f'] = rouge_scores['rouge-l-f']
    ref_dup_scores = calc_duplicate_n_grams_rate(y_true)
    metrics['ref_duplicate_ngrams'] = ref_dup_scores
    hyp_dup_scores = calc_duplicate_n_grams_rate(y_pred)
    metrics['hyp_duplicate_ngrams'] = hyp_dup_scores
    ref_lengths = [len(i) for i in y_true]
    metrics['ref_avg_length'] = sum(ref_lengths) / len(ref_lengths)
    hyp_lengths = [len(i) for i in y_pred]
    metrics['hyp_avg_length'] = sum(hyp_lengths) / len(hyp_lengths)
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
    print('ROUGE-1-F:\t{:3.1f}'.format(metrics['rouge-1-f'] * 100.0))
    print('ROUGE-2-F:\t{:3.1f}'.format(metrics['rouge-2-f'] * 100.0))
    print('ROUGE-3-F:\t{:3.1f}'.format(metrics['rouge-3-f'] * 100.0))
    print('ROUGE-L-F:\t{:3.1f}'.format(metrics['rouge-l-f'] * 100.0))
    print('Ref Dup 1-grams:{:3.1f}'.format(metrics['ref_duplicate_ngrams'][1] * 100.0))
    print('Ref Dup 2-grams:{:3.1f}'.format(metrics['ref_duplicate_ngrams'][2] * 100.0))
    print('Ref Dup 3-grams:{:3.1f}'.format(metrics['ref_duplicate_ngrams'][3] * 100.0))
    print('Hyp Dup 1-grams:{:3.1f}'.format(metrics['hyp_duplicate_ngrams'][1] * 100.0))
    print('Hyp Dup 2-grams:{:3.1f}'.format(metrics['hyp_duplicate_ngrams'][2] * 100.0))
    print('Hyp Dup 3-grams:{:3.1f}'.format(metrics['hyp_duplicate_ngrams'][3] * 100.0))
    print('Ref Avg length:\t{:3.1f}'.format(metrics['ref_avg_length']))
    print('Hyp Avg length:\t{:3.1f}'.format(metrics['hyp_avg_length']))
