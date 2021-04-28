#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install simpletransformers')


# In[2]:


import logging
import pandas as pd
from simpletransformers.ner import NERModel, NERArgs


# In[3]:


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = NERArgs()
model_args.evaluate_during_training = True
model_args.labels_list = ["t", "nt"]
model_args.overwrite_output_dir = True
model_args.evaluate_during_training = True
model_args.num_train_epochs = 10
# model_args.evaluate_during_training_steps = 10000
# model_args.save_steps = 10000
model_args.best_model_dir = './drive/MyDrive/best_model'


model = NERModel(
    "roberta",
    "roberta-base",
    args=model_args,
)


# In[4]:


train_data = pd.read_csv('train.csv')
dev_data = pd.read_csv('dev.csv')
test_data = pd.read_csv('test.csv')


# In[5]:


train_data.info()


# In[6]:


train_data[train_data.isna().any(axis=1)]


# In[7]:


train_data = train_data.dropna()


# In[8]:


# Train the model
model.train_model(train_data, eval_data=dev_data)


# In[9]:


from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
def read_data(file_name):
    '''
    read data and analyze different types of tokens. (bert is better since it consider word and character level)
    '''
    data = pd.read_csv(file_name)
    sent_list = data['text']
    sent_label = data['spans']
    sent_list = [x.lower() for x in sent_list]
    print(len(sent_list))
    # sns.histplot([len(x) for x in sent_list], bins=[64 * i for i in range(20)])
    # plt.show()
    print(sent_list[1])
    
    # tokenizer_origin = RegexpTokenizer(r'[^\s]+')
    # tokens_list = [tokenizer_origin.tokenize(x) for x in sent_list]
    # sns.histplot([len(x) for x in tokens_list], bins=[12 * i for i in range(20)])
    # plt.show()
    # print(tokens_list[1])
    
    tokens_list = [tokenizer.tokenize(x) for x in sent_list]
    # sns.histplot([len(x) for x in tokens_list], bins=[12 * i for i in range(20)])
    # plt.show()
    print(tokens_list[1])
    
    return sent_list, sent_label, tokens_list


# In[10]:


sent_list_test, sent_label_test, tokens_list_test = read_data('tsd_test.csv')


# In[11]:


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = NERArgs()

model_args.evaluate_during_training = True
model_args.labels_list = ["t", "nt"]
model_args.overwrite_output_dir = True
model_args.evaluate_during_training = True
model_args.num_train_epochs = 5
# model_args.evaluate_during_training_steps = 10000
# model_args.save_steps = 10000
model_args.best_model_dir = './drive/MyDrive/best_model'


model = NERModel(
    "roberta",
    "drive/MyDrive/best_model",
    args=model_args,
)


# In[12]:


predictions, raw_outputs = model.predict(tokens_list_test, split_on_space=False)


# In[12]:


predictions, raw_outputs = model.predict(tokens_list_test, split_on_space=False)


# In[13]:


result = []
for i in range(len(predictions)):
  temp = []
  num = 0
  for j in range(len(predictions[i])):
    for k in range(len(tokens_list_test[i][j])):
      if predictions[i][j][tokens_list_test[i][j]] == 't':
        temp.append(num)
      num += 1
  result.append(temp)


# In[14]:


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


# In[15]:


import numpy as np
f1_list = avg_f1_score(result, sent_label_test)


# In[16]:


np.mean(f1_list)

