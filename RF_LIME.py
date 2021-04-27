import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn import feature_extraction

train = pd.read_csv('LIME_train.csv')
test = pd.read_csv('LIME_test.csv')
class_names = ['non-toxic', 'toxic']

vectorizer = feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(train.text)
test_vectors = vectorizer.transform(test.text)

rf = RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, train.labels)

pred = rf.predict(test_vectors)
print(f1_score(test.labels, pred, average='binary'))

from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=class_names)
c = make_pipeline(vectorizer, rf)

print(c.predict_proba([test.text[0]]))

idx = 83
exp = explainer.explain_instance(test.text[idx], c.predict_proba, num_features=6)
print('Document id: %d' % idx)
print('Probability =', c.predict_proba([test.text[idx]])[0,1])
print('True class: %s' % class_names[test.labels[idx]])

print(exp.as_list(label=1))
exp.save_to_file('oi.html')
