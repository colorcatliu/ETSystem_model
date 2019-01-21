#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


train_1v1 = pickle.load(open('train_1v1_one-hot','r'))
train_1v1_label = pickle.load(open('train_1v1_label','r'))

test_1v1 = pickle.load(open('test_1v1_one-hot','r'))
test_1v1_label = pickle.load(open('test_1v1_label','r'))


# In[3]:


categories_train_1v1 = to_categorical(train_1v1_label[0])
categories_test_1v1 = to_categorical(test_1v1_label[0])


# In[ ]:


categories_test_1v1[0]


# In[ ]:


t = pickle.load(open('train_1v1_one-hot','r'))


# In[ ]:


len(test_1v1)


# In[ ]:


''' the length of sentences gets 422 for the longest,
most of the sentences have the length of below 200,
only nine sentences get lengths of over 200, 
and only two sentences get length of over 400;
'''
n=0
for item in train_1v1+test_1v1:
    if len(item)>n:
        n = len(item)
print n


# In[4]:


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


# In[ ]:


# model = load_model('./results_models/cnn_lstm_1v1_data_trained_model')


# In[12]:


model = Sequential()
model.add(Embedding(vocabulary, embedding_size, input_length=max_len))
model.add(Dropout(0.5))
model.add(Conv1D(filters, kernel_size, padding='valid', 
                 activation='relu',strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
# model.add(LSTM(lstm_output_size))
model.add(Flatten())
model.add(Dense(9,kernel_initializer='RandomUniform')) # RandomNormal, RandomUniform, TruncatedNormal 
# model.add(Dense(9))
# model.add(Activation('sigmoid'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='adam',metrics=['accuracy'])
# tensorboard = TensorBoard(log_dir='./logs_1v1')
history=model.fit(train,categories_train_1v1, batch_size=batch_size, 
          epochs=50,validation_data=(test, categories_test_1v1), verbose=0) 
# model.fit(train,categories_train_1v1, batch_size=batch_size,
#           epochs=epochs,validation_data=(test, categories_test_1v1), callbacks=[tensorboard]) 

# loss, acc = model.evaluate(test, categories_test_1v1,batch_size=30)
# print 'loss:',loss
# print 'accuracy:',acc


# In[15]:


pickle.dump(history.history,open('./results_models/1v1_cnn_50s_iterate','w'))


# In[ ]:


model.save('./results_models/cnn_lstm_1v1_data_trained_model_0.32_macro_0.24')


# In[ ]:


test_results = model.predict_classes(test)


# In[ ]:


test_results[:100]


# In[ ]:


p=metrics.precision_score(test_1v1_label[0],test_results,average='macro')
print p


# In[ ]:


r = metrics.recall_score(test_1v1_label[0],test_results,average='macro')
print r


# In[ ]:


2*p*r/(p+r)


# In[ ]:


p=metrics.precision_score(test_1v1_label[0],test_results,average='micro')
print p
r = metrics.recall_score(test_1v1_label[0],test_results,average='micro')
print r

2*p*r/(p+r)


# In[ ]:


labels=['anger','anxiety','expect','hate','joy','love','noemo','sorrow','surprise']


# In[ ]:


metrics.confusion_matrix(test_1v1_label[0],test_results)


# In[ ]:


test_4v1 = pickle.load(open('test_4v1_one-hot','r'))
test_4v1_label = pickle.load(open('test_4v1_label','r'))

categories_test_4v1 = to_categorical(test_4v1_label[0])
test_4v1 = sequence.pad_sequences(test_4v1, maxlen=max_len)


# In[ ]:


test_results_4v1 = model.predict_classes(test_4v1)
p=metrics.precision_score(test_4v1_label[0],test_results_4v1,average='macro')
print p
r = metrics.recall_score(test_4v1_label[0],test_results_4v1,average='macro')
print r
2*p*r/(p+r)


# In[ ]:


p=metrics.precision_score(test_4v1_label[0],test_results_4v1,average='micro')
print p
r = metrics.recall_score(test_4v1_label[0],test_results_4v1,average='micro')
print r
2*p*r/(p+r)


# In[ ]:


model.summary()


# In[ ]:


from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
plot_model(model, to_file='cnn_lstm.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))

