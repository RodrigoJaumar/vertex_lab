#!/usr/bin/env python
# coding: utf-8

# In[12]:


from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
np.random.seed(7)
df = pd.read_csv("gs://prebuilt_container_training_ro/IRIS.csv")


# In[13]:


df


# In[14]:


X = df.drop(["target"],axis=1)
#Y = df['target'].map({'0': 0, '1': 1, '2': 2})
Y = df['target']


# In[15]:


Y


# In[16]:


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

dummy_y = np_utils.to_categorical(encoded_Y)


# In[19]:


model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[20]:


model.fit(X, dummy_y, epochs=150, batch_size=5)


# In[21]:


scores = model.evaluate(X, dummy_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[22]:


predictions = model.predict(X)


# In[23]:


test = np.argmax(predictions,axis=1)
test


# In[24]:


model.save('gs://prebuilt_container_training_ro/model_output')


# In[ ]:




