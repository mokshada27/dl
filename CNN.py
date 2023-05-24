#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.datasets import fashion_mnist
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()


# In[3]:


import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

print('Training Data Shape: ',x_train.shape, y_train.shape)

print('Testing Data Shape: ',x_test.shape, y_test.shape)


# In[4]:


classes=np.unique(y_train)
print('Total number of outputs : ', len(classes))
print('Output classes : ', classes)


# In[6]:


x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)
x_train.shape, x_test.shape


# In[9]:


x_train.dtype, x_test.dtype


# In[10]:


x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train=x_train/255
x_test=x_test/255


# In[12]:


# Change the labels from categorical to one-hot encoding
y_train_ohe=to_categorical(y_train)
y_test_ohe=to_categorical(y_test)

# Display the change for category label using one-hot encoding
print('Original label:', y_train[0])
print('After conversion to one-hot:', y_train_ohe[0])


# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train_ohe, test_size=0.2, random_state=13)


# In[14]:


x_train.shape, x_val.shape, y_train.shape, y_val.shape


# In[16]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D, Dense, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU


# In[17]:


batch_size=64
epochs=20
num_classes=10


# In[21]:


fashion_model=Sequential()
fashion_model.add(Conv2D(32,kernel_size=(3,3), activation='linear',padding='same', input_shape=(28,28,1)))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64, kernel_size=(3,3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128,kernel_size=(3,3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
fashion_model.add(Dropout(0.40))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(Dropout(0.30))
fashion_model.add(Dense(num_classes, activation='softmax'))


# In[22]:


fashion_model.summary()


# In[23]:


fashion_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[24]:


fashion_model.fit(x_train,y_train,batch_size=64,epochs=20,verbose=1,validation_data=(x_val,y_val))


# In[25]:


score=fashion_model.evaluate(x_test,y_test_ohe,verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:




