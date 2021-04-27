import pandas as pd
import numpy as np
import json
work_dir = '~/Documents/UMSI/630-NLP/FinalProject/'

train_file = 'tsd_train.csv'
test_file = 'tsd_test.csv'

train = pd.read_csv(work_dir+train_file)
train_text = []
train_label = []
for i in range(7000):
    train_text.append(train.iloc[i]['text'].replace('\n', ' ñ '))
    if json.loads(train.iloc[i]['spans']) == []:
        train_label.append(0)
    else:
        train_label.append(1)

result_train = {'text': train_text, 'labels':train_label}
pd.DataFrame(result_train).to_csv('LIME_train.csv',index=False)

dev_text = []
dev_label = []
for i in range(7000,train.shape[0]):
    dev_text.append(train.iloc[i]['text'].replace('\n', ' ñ '))
    if json.loads(train.iloc[i]['spans']) == []:
        dev_label.append(0)
    else:
        dev_label.append(1)

result_dev = {'text': dev_text, 'labels':dev_label}
pd.DataFrame(result_dev).to_csv('LIME_dev.csv',index=False)


test = pd.read_csv(work_dir+test_file)
test_text = []
test_label = []
for i in range(test.shape[0]):
    test_text.append(test.iloc[i]['text'].replace('\n', ' ñ '))
    if json.loads(test.iloc[i]['spans']) == []:
        test_label.append(0)
    else:
        test_label.append(1)

result_test = {'text': test_text, 'labels':test_label}
pd.DataFrame(result_test).to_csv('LIME_test.csv',index=False)

