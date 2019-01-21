#!/usr/bin/env python
# coding: utf-8

from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding, add
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.preprocessing import text
import pickle
import numpy as np

from keras.utils.np_utils import to_categorical
from sklearn import metrics
from keras.callbacks import TensorBoard

#load date
train_1v1 = pickle.load(open('...','r'))
train_1v1_label = pickle.load(open('...','r'))

test_1v1 = pickle.load(open('...','r'))
test_1v1_label = pickle.load(open('...','r'))


categories_train_1v1 = to_categorical(train_1v1_label[0])
categories_test_1v1 = to_categorical(test_1v1_label[0])



# Embedding
vocabulary = 32127 #  32126 index+1
max_len = 200
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

#LSTM
lstm_output_size  = 70

#Training
batch_size = 30
epochs = 3


# In[5]:


train = sequence.pad_sequences(train_1v1, maxlen=max_len)
test = sequence.pad_sequences(test_1v1, maxlen=max_len)


model = Sequential()
model.add(Embedding(vocabulary, embedding_size, input_length=max_len))
model.add(Dropout(0.5))
model.add(Conv1D(filters, kernel_size, padding='valid', 
                 activation='relu',strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(9,kernel_initializer='RandomUniform')) # RandomNormal, RandomUniform, TruncatedNormal 
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='adam',metrics=['accuracy'])
# tensorboard = TensorBoard(log_dir='./logs_1v1')
#history=model.fit(train,categories_train_1v1, batch_size=batch_size, 
#          epochs=50,validation_data=(test, categories_test_1v1), verbose=0) 
model.fit(train,categories_train_1v1, batch_size=batch_size,
          epochs=epochs,validation_data=(test, categories_test_1v1), callbacks=[tensorboard]) 

# loss, acc = model.evaluate(test, categories_test_1v1,batch_size=30)
# print 'loss:',loss
# print 'accuracy:',acc


# In[15]:


# pickle.dump(history.history,open('....','w')) #save acc,loss


# In[ ]:


model.save('...') #save model

test_results = model.predict_classes(test)

#macro

p=metrics.precision_score(test_1v1_label[0],test_results,average='macro')
print p

r = metrics.recall_score(test_1v1_label[0],test_results,average='macro')
print r

f1 = 2*p*r/(p+r)
print f1


# micro 
'''
p=metrics.precision_score(test_1v1_label[0],test_results,average='micro')
print p
r = metrics.recall_score(test_1v1_label[0],test_results,average='micro')
print r

f1 = 2*p*r/(p+r)
'''

model.summary()

