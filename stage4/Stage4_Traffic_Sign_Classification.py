#!/usr/bin/env python
# coding: utf-8

# # STEP 1: IMPORT LIBRARIES AND DATASET

# In[3]:


import warnings
warnings.filterwarnings("ignore")


# In[4]:


# import libraries 
import pickle
import seaborn as sns
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import random


# In[5]:


# The pickle module implements binary protocols for serializing and de-serializing a Python object structure.
with open("train.p", mode='rb') as training_data:
    train = pickle.load(training_data)
with open("valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open("test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)


# In[6]:


X_train, y_train = train['features'], train['labels']
X_validation, y_validation = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# In[7]:


X_train.shape


# In[8]:


y_train.shape


# In[9]:


X_validation.shape


# In[10]:


y_validation.shape


# In[11]:


X_test.shape


# In[12]:


y_test.shape


# In[22]:


type(y_train)


# In[46]:


unique,count = np.unique(y_train,return_counts=True)
data_count = dict(zip(unique,count))
fig, ax = plt.subplots(figsize=(4,3))
plt.bar(data_count.keys(), data_count.values(), color='g')
ax.set_title('Statistics of train labels')
plt.savefig('Statistics of train labels.pdf', dpi=600)


# In[47]:


unique,count = np.unique(y_test,return_counts=True)
data_count = dict(zip(unique,count))
fig, ax = plt.subplots(figsize=(4,3))
plt.bar(data_count.keys(), data_count.values(), color='g')
ax.set_title('Statistics of test labels')
plt.savefig('Statistics of test labels.pdf', dpi=600)


# In[48]:


unique,count = np.unique(y_validation,return_counts=True)
data_count = dict(zip(unique,count))
fig, ax = plt.subplots(figsize=(4,3))
plt.bar(data_count.keys(), data_count.values(), color='g')
ax.set_title('Statistics of validation labels')
plt.savefig('Statistics of validation labels.pdf', dpi=600)


# # STEP 2: IMAGE EXPLORATION

# In[13]:


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


# # STEP 3: DATA PEPARATION

# In[14]:


## Shuffle the dataset 
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)


# In[15]:


X_train.shape


# In[16]:


## transfer to gray images
X_train_gray = np.sum(X_train/3, axis=3, keepdims=True)
X_test_gray  = np.sum(X_test/3, axis=3, keepdims=True)
X_validation_gray  = np.sum(X_validation/3, axis=3, keepdims=True) 


# In[17]:


X_train_gray.shape


# In[18]:


## normalization
X_train_gray_norm = (X_train_gray - 128)/128 
X_test_gray_norm = (X_test_gray - 128)/128
X_validation_gray_norm = (X_validation_gray - 128)/128


# In[19]:


X_train_gray.shape


# In[74]:


i = 620
fig, ax = plt.subplots(1,3,figsize=(6,4))

ax[0].imshow(X_train_gray[i].squeeze(), cmap='gray')
ax[0].set_title('X_train_gray')
ax[1].imshow(X_train[i].squeeze())
ax[1].set_title('X_train')
ax[2].imshow(X_train_gray_norm[i].squeeze(), cmap='gray')
ax[2].set_title('X_train_gray_norm')
fig.tight_layout()
fig.savefig('image_show_styles.pdf', dpi=600)
plt.show()


# # STEP 4: MODEL TRAINING

# The model consists of the following layers:
# 
# STEP 1: THE FIRST CONVOLUTIONAL LAYER #1
# Input = 32x32x1
# Output = 28x28x6
# Output = (Input-filter+1)/Stride* => (32-5+1)/1=28
# Used a 5x5 Filter with input depth of 3 and output depth of 6
# Apply a RELU Activation function to the output
# pooling for input, Input = 28x28x6 and Output = 14x14x6
# * Stride is the amount by which the kernel is shifted when the kernel is passed over the image.
# 
# STEP 2: THE SECOND CONVOLUTIONAL LAYER #2
# Input = 14x14x6
# Output = 10x10x16
# Layer 2: Convolutional layer with Output = 10x10x16
# Output = (Input-filter+1)/strides => 10 = 14-5+1/1
# Apply a RELU Activation function to the output
# Pooling with Input = 10x10x16 and Output = 5x5x16
# 
# STEP 3: FLATTENING THE NETWORK
# Flatten the network with Input = 5x5x16 and Output = 400
# 
# STEP 4: FULLY CONNECTED LAYER
# Layer 3: Fully Connected layer with Input = 400 and Output = 120
# Apply a RELU Activation function to the output
# 
# STEP 5: ANOTHER FULLY CONNECTED LAYER
# Layer 4: Fully Connected Layer with Input = 120 and Output = 84
# Apply a RELU Activation function to the output
# 
# STEP 6: FULLY CONNECTED LAYER
# Layer 5: Fully Connected layer with Input = 84 and Output = 43

# In[57]:


# Import train_test_split from scikit library

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split


# In[58]:


cnn_model = Sequential()

cnn_model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32,32,1)))
cnn_model.add(AveragePooling2D())

cnn_model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
cnn_model.add(AveragePooling2D())

cnn_model.add(Flatten())

cnn_model.add(Dense(units=120, activation='relu'))

cnn_model.add(Dense(units=84, activation='relu'))

cnn_model.add(Dense(units=43, activation = 'softmax'))


# In[59]:


cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])


# In[60]:


# training
history = cnn_model.fit(X_train_gray_norm,
                        y_train,
                        batch_size=500,
                        nb_epoch=10,
                        verbose=1,
                        validation_data = (X_validation_gray_norm,y_validation))


# # STEP 5: MODEL EVALUATION

# In[61]:


score = cnn_model.evaluate(X_test_gray_norm, y_test,verbose=0)
print('Test Accuracy : {:.4f}'.format(score[1]))


# In[62]:


history.history.keys()


# In[76]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))
plt.figure(figsize=(4,3))
plt.plot(epochs, accuracy, 'b', ls = '--', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.savefig('Training and validation accuracy.pdf', dpi=600)
plt.show()


# In[ ]:




