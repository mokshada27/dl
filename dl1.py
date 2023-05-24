#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("Boston.csv")
df


# In[3]:


df.dtypes


# In[4]:


df.shape


# In[5]:


# Finding out the correlation between the features
corr=df.corr()
corr.shape


# In[7]:


x=df.drop(['medv'],axis=1)
y=df['medv']


# In[8]:


x


# In[18]:


x=x.drop(['Unnamed: 0'], axis=1)


# In[19]:


y


# In[20]:


y.describe()


# In[21]:


# Splitting to training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3, random_state=4)


# In[22]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)


# In[23]:


x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


# In[24]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# In[25]:


model=Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(54, activation='relu'))
model.add(Dropout(0.18))
model.add(Dense(1))


# In[26]:


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse','mae'])


# In[27]:


model.fit(x_train,y_train, validation_split=0.2, epochs=200)


# In[31]:


score=model.evaluate(x_test,y_test, verbose=0)
print('Mean Square Error: ',score[1])
print('Mean Absolute Error: ',score[2])


# In[32]:


y_pred=model.predict(x_test)
y_pred


# In[33]:


from sklearn.metrics import r2_score


# In[34]:


print('r2 score: ',r2_score(y_test,y_pred))


# In[36]:


print(y_pred[:5])
print(y_test[:5])
y_test.head()


# In[37]:


#Using ML Linear Model

from sklearn.linear_model import LinearRegression


# In[38]:


LR=LinearRegression()
LR.fit(x_train,y_train)


# In[39]:


LR.intercept_


# In[40]:


Y_pred=LR.predict(x_train)


# In[44]:


# Model Evaluation
print('R^2:',metrics.r2_score(y_train, Y_pred))
print('MAE:',metrics.mean_absolute_error(y_train, Y_pred))
print('MSE:',metrics.mean_squared_error(y_train, Y_pred))


# In[ ]:




