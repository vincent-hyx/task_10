import lime
import sklearn
import numpy as np
import sklearn.ensemble
import sklearn.metrics
from sklearn.datasets import fetch_20newsgroups
#读取数据
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)
print(train_vectors[:2])
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)

pred = rf.predict(test_vectors)
print(pred)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')

from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, rf)

print(c.predict_proba([newsgroups_test.data[0]]))

from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)

idx = 83
print(f"\n{newsgroups_test.data[idx]}\n")
print(type(newsgroups_test.data[idx]))
exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=10)
print('Document id: %d' % idx)
print('Probability(christian) =', c.predict_proba([newsgroups_test.data[idx]])[0, 1])
print('True class: %s' % class_names[newsgroups_test.target[idx]])

exp.as_list()

print('Original prediction:', rf.predict_proba(test_vectors[idx])[0, 1])
tmp = test_vectors[idx].copy()
tmp[0, vectorizer.vocabulary_['Posting']] = 0
tmp[0, vectorizer.vocabulary_['Host']] = 0
print('Prediction removing some features:', rf.predict_proba(tmp)[0, 1])
print('Difference:', rf.predict_proba(tmp)[0, 1] - rf.predict_proba(test_vectors[idx])[0, 1])

import matplotlib.pyplot as plt

fig = exp.as_pyplot_figure()

exp.show_in_notebook(text=False)

# exp.save_to_file('/tmp/oi.html')

exp.show_in_notebook(text=True)
plt.show()