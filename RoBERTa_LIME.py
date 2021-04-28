#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install simpletransformers')


# In[2]:


import logging
import pandas as pd
from simpletransformers.ner import NERModel, NERArgs
from simpletransformers.classification import ClassificationModel, ClassificationArgs


# In[23]:


train_file = "LIME_train.csv"
test_file = "LIME_test.csv"
dev_file = "LIME_dev.csv"
train_df = pd.read_csv(train_file)
eval_df = pd.read_csv(dev_file)
test_df = pd.read_csv(test_file)


# In[28]:


args = ClassificationArgs(num_train_epochs=5)
args.best_model_dir = './drive/MyDrive/best_model_classify'
args.evaluate_during_training = True
args.overwrite_output_dir = True
model = ClassificationModel('roberta', 'roberta-base', use_cuda=True, args=args)


# In[29]:


model.train_model(train_df, eval_df=eval_df)


# In[31]:


result, model_outputs, wrong_predictions = model.eval_model(test_df)


# In[33]:


get_ipython().system('pip install lime')


# In[36]:


from lime.lime_text import LimeTextExplainer
import numpy as np


# In[38]:


# define softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# define prediction function
def predict_probs(texts):
    predictions = model.predict(texts)
    x = np.array(list(predictions)[1])
    return np.apply_along_axis(softmax, 1, x)

result = []

# explain instance with LIME
def explain(text):
    class_names = [0,1]
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, predict_probs, num_features=6, labels=[0,1])
    result.append(exp.as_list(label=1))
    exp.save_to_file('lime.html')


# In[39]:


for i in range(test_df.shape[0]):
    explain(test_df.iloc[i]['text'])


# In[40]:


text = test_df.iloc[0]['text']


# In[50]:


text.replace(' ñ ', ' ')


# In[51]:


class_names = [0,1]
explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(text, predict_probs, num_features=20, num_samples=2000)
# exp.show_in_notebook(text)
result = []
result.append(exp.as_list(label=1))
exp.save_to_file('lime.html')


# In[48]:


import json
with open ('result_lime.json', 'w') as fout :
    json.dump(result, fout, indent =4)


# In[ ]:




