#!/usr/bin/env python
# coding: utf-8

# # STEP 1: IMPORT LIBRARIES AND DATASET

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


# import libraries 
import pickle
import seaborn as sns
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import random


# In[3]:


# The pickle module implements binary protocols for serializing and de-serializing a Python object structure.
with open("train.p", mode='rb') as training_data:
    train = pickle.load(training_data)
with open("valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open("test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)


# In[4]:


X_train, y_train = train['features'], train['labels']
X_validation, y_validation = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# In[5]:


X_train.shape


# In[6]:


y_train.shape


# In[7]:


X_validation.shape


# In[8]:


y_validation.shape


# In[9]:


X_test.shape


# In[10]:


y_test.shape


# # STEP 2: IMAGE EXPLORATION

# In[11]:


import matplotlib.pyplot as plt
import numpy as np
fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
fig.set_size_inches(10, 10, forward=True)
fig.tight_layout(pad=1.0)

for i in range(4):
    for j in range(4):
        axs[i, j].imshow(X_train[100*i + 100*j])
        axs[i, j].set_title('Label: {}'.format(y_train[100*i + 100*j]))
plt.show()

