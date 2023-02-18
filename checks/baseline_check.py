from models.baseline import FirstWorder
from util.metrics import print_corp_metrics
from util.tokenizator import tokenize_sentences

X_train = ['В настоящей работе проводится анализ и даются юридические определения понятия и экологической политики.',
           'Научная работа посвящена анализу изменений в социально-экономических процессах прибрежных регионов']
y_train = ['Правовое обеспечение экологической политики в сфере общественного питания',
           'Сравнительный анализ влияния шельфовых проектов на развитие экономики прибрежных регионов']

model = FirstWorder('mean')
model.fit(X_train, y_train)
print(model.optimal_length)
y_pred = model.predict(X_train)
y_true = tokenize_sentences(y_train)
print_corp_metrics(y_true, y_pred)
