#!/usr/bin/env python
# coding: utf-8

# # This is Custom training

# In[3]:


import pandas as pd


# In[4]:


data = pd.read_csv("gs://custom_container_training_ro/IRIS.csv")


# In[5]:


type(data)


# In[6]:


data


# In[7]:


from sklearn.model_selection import train_test_split
array = data.values
X = array[:,0:4]
y = array[:,4]
# Split the data to train and test dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=1)


# In[8]:


X_train.shape


# In[9]:


X_test.shape


# In[10]:


y_train.shape


# In[11]:


y_test.shape


# In[12]:


from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, y_train)


# In[13]:


svn


# In[14]:


predictions = svn.predict(X_test)
# Calculate the accuracy 
from sklearn.metrics import accuracy_score 
accuracy_score(y_test, predictions)


# In[15]:


import pickle
import logging
with open('model.pkl', 'wb') as model_file:
  pickle.dump(svn, model_file)


# In[16]:


from google.cloud import storage
storage_path = "gs://custom_container_training_ro/model.pkl"
blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
blob.upload_from_filename('model.pkl')
logging.info("model exported to : {}".format(storage_path))


# In[ ]:




