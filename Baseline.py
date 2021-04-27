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

import json
def avg_f1_score(result, sent_label):
    F1 = []
    for i in range(len(result)):
        intersect_num = 0
        result_num = 0
        origin_num = 0
        for result_loc in result[i]:
            result_num += 1
            if result_loc in json.loads(sent_label[i]):
                intersect_num += 1
        for origin_loc in json.loads(sent_label[i]):
            origin_num += 1
        if intersect_num == 0:
            precision = recall = 0
        else:
            precision = intersect_num / result_num
            recall = intersect_num / origin_num
        if precision == 0 or recall == 0:
            F1_ = 0
        else:
            F1_ = 2 * precision * recall / (precision + recall)
        F1.append(F1_)
    return F1


def read_data(file_name):
    '''
    read data and analyze different types of tokens. (bert is better since it consider word and character level)
    '''
    data = pd.read_csv(file_name)
    sent_list = data['text']
    sent_label = data['spans']
    sent_list = [x.lower() for x in sent_list]
    return sent_list, sent_label

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
result = []
for i in range(len(pred)):
    each = []
    num = 0
    if pred[i] == 1:
        for j in range(len(test.text[i])):
            each.append(num)
            num += 1
    result.append(each)

sent_list, sent_label = read_data('~/Documents/UMSI/630-NLP/FinalProject/tsd_test.csv')
f1_list = avg_f1_score(result, sent_label)
print(np.mean(f1_list))