#!/usr/bin/env python
# coding: utf-8

# In[41]:


import json
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from nltk.tokenize import RegexpTokenizer
from transformers import BertTokenizer
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[42]:


workDir = '~/Documents/UMSI/630-NLP/FinalProject/'
trainfile = 'tsd_train.csv'
testfile = 'tsd_test.csv'


# In[46]:


def read_data(file_name):
    '''
    read data and analyze different types of tokens. (bert is better since it consider word and character level)
    '''
    data = pd.read_csv(file_name)
    sent_list = data['text']
    sent_label = data['spans']
    sent_list = [x.lower() for x in sent_list]
    print(len(sent_list))
    sns.histplot([len(x) for x in sent_list], bins=[64 * i for i in range(20)])
    plt.show()
    print(sent_list[1])
    
    tokenizer_origin = RegexpTokenizer(r'[^\s]+')
    tokens_list = [tokenizer_origin.tokenize(x) for x in sent_list]
    sns.histplot([len(x) for x in tokens_list], bins=[12 * i for i in range(20)])
    plt.show()
    print(tokens_list[1])
    
    tokens_list = [tokenizer.tokenize(x) for x in sent_list]
    sns.histplot([len(x) for x in tokens_list], bins=[12 * i for i in range(20)])
    plt.show()
    print(tokens_list[1])
    
    return sent_list, sent_label, tokens_list


# In[48]:


sent_list, sent_label, tokens_list = read_data(workDir + trainfile)


# In[102]:


def reberta_labeling(tokens_list, sent_label,istrain=True):
    '''
    label the bert tokens according to the ground truth labels
    '''
    if istrain is True:
        index, tokens, labels = [], [], []
        for i in range(7000):
            num = 0
            for j in range(len(tokens_list[i])):

                istoxic = 0
                for k in range(len(tokens_list[i][j])):
                    if num in json.loads(sent_label[i]):
                        istoxic = 1
                    num += 1
                index.append(i)
                tokens.append(tokens_list[i][j])
                if istoxic == 1:
                    labels.append('t')
                else:
                    labels.append('nt')
        result_dict = {'sentence_id':index, 'words':tokens, 'labels':labels}
        result = pd.DataFrame(result_dict)
        result.to_csv('train.csv',index=False)
        
        index, tokens, labels = [], [], []
        for i in range(7000,len(tokens_list)):
            num = 0
            for j in range(len(tokens_list[i])):
                istoxic = 0
                for k in range(len(tokens_list[i][j])):
                    if num in json.loads(sent_label[i]):
                        istoxic = 1
                    num += 1
                index.append(i-7000)
                tokens.append(tokens_list[i][j])
                if istoxic == 1:
                    labels.append('t')
                else:
                    labels.append('nt')
        result_dict = {'sentence_id':index, 'words':tokens, 'labels':labels}
        result = pd.DataFrame(result_dict)
        result.to_csv('dev.csv',index=False)
    else:
        index, tokens, labels = [], [], []
        for i in range(len(tokens_list)):
            num = 0
            for j in range(len(tokens_list[i])):

                istoxic = 0
                for k in range(len(tokens_list[i][j])):
                    if num in json.loads(sent_label[i]):
                        istoxic = 1
                    num += 1
                index.append(i)
                tokens.append(tokens_list[i][j])
                if istoxic == 1:
                    labels.append('t')
                else:
                    labels.append('nt')
        result_dict = {'sentence_id':index, 'words':tokens, 'labels':labels}
        result = pd.DataFrame(result_dict)
        result.to_csv('test.csv',index=False)


# In[103]:


reberta_labeling(tokens_list, sent_label)


# In[ ]:




