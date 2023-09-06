#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


diabetes = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Diabetes.csv')


# In[3]:


diabetes.head()
     


# In[4]:


diabetes.info()


# In[5]:


diabetes.columns


# In[6]:


y = diabetes['diabetes']


# In[7]:


X = diabetes.drop(['diabetes'],axis=1)


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)


# In[9]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[10]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=500)
     


# In[11]:


model.fit(X_train,y_train)


# In[13]:


model.intercept_


# In[14]:


model.coef_


# In[15]:


y_pred = model.predict(X_test)


# In[18]:


y_pred


# In[19]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[20]:


confusion_matrix(y_test,y_pred)


# In[21]:


accuracy_score(y_test,y_pred)


# In[22]:


print(classification_report(y_test,y_pred))


# In[ ]:




