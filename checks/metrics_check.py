from util.metrics import print_corp_metrics
from util.tokenizator import tokenize_sentences

refs = ['Гуси летят домой и несут Нельса с собой', 'Шла Саша по шоссе', '', 'Стол стул кружка']
refs_ = tokenize_sentences(refs)
hyps = ['Гуси с Нельсом летят домой', 'Саша идет по шоссе', '', '']
hyps_ = tokenize_sentences(hyps)

print_corp_metrics(refs_, hyps_)
