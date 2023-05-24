#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import warnings
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# In[2]:


df=pd.read_csv('Google_Stock_Price_Train.csv')
df


# In[3]:


df['Close']=df['Close'].str.replace(',','').astype('float32')
df['Volume']=df['Volume'].str.replace(',','').astype('float32')


# In[4]:


df.isnull().sum()


# In[5]:


df[['Open','High','Low','Close','Volume']].plot(kind= 'box' ,layout=(1,5),subplots=True, sharex=False, sharey=False, figsize=(20,6),color='blue')
plt.show()


# In[6]:


df


# In[7]:


scaler=MinMaxScaler()
df_without_date=df[['Open','High','Low','Close','Volume']]
data_scaled = pd.DataFrame(scaler.fit_transform(df_without_date))


# In[8]:


data_scaled


# In[9]:


data_scaled=data_scaled.drop([1,2,4], axis=1)
data_scaled


# In[10]:


def split_seq_multivariate(sequence, n_past, n_future):
    x, y = [], [] 
    for window_start in range(len(sequence)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(sequence):
            break
        # slicing the past and future parts of the window
        past   = sequence[window_start:past_end, :]
        future = sequence[past_end:future_end, -1]
        x.append(past)
        y.append(future)
    
    return np.array(x), np.array(y)


# In[11]:


n_steps=60
data_scaled=data_scaled.to_numpy()
data_scaled.shape


# In[12]:


x,y=split_seq_multivariate(data_scaled, n_steps,1)


# In[13]:



# X is in the shape of [samples, timesteps, features]
print(x.shape)
print(y.shape)

# make y to the shape of [samples]
y=y[:,0]
y.shape


# In[14]:


x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=50)


# In[15]:


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[16]:


# further dividing the training set into train and validation data
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.2,random_state=30)

print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)


# In[17]:


model=Sequential()
model.add(LSTM(612, input_shape=(n_steps,2)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))


# In[18]:


model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# In[19]:


history=model.fit(x_train, y_train, verbose=2, batch_size=32, epochs=250, validation_data=(x_val,y_val))


# In[20]:


from matplotlib import pyplot
# plot learning curves
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Root Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.legend()
pyplot.show()


# In[21]:



# evaluate the model
mse, mae = model.evaluate(x_test, y_test, verbose=0)
print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, np.sqrt(mse), mae))


# In[22]:


# predicting y_test values
print(x_test.shape)
predicted_values = model.predict(x_test)
print(predicted_values.shape)
# print(predicted_values)


# In[23]:


# evaluating using R squared
R_square = r2_score(y_test, predicted_values) 
 
print(R_square)


# In[24]:


plt.plot(y_test,c = 'r')
plt.plot(predicted_values,c = 'y')
plt.xlabel('Day')
plt.ylabel('Stock Price Volume')
plt.title('Stock Price Volume Prediction Graph using RNN (LSTM)')
plt.legend(['Actual','Predicted'],loc = 'lower right')
plt.figure(figsize=(10,6))


# In[25]:


from matplotlib import pyplot
# plot learning curves
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Root Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.legend()
pyplot.show()


# In[ ]:




