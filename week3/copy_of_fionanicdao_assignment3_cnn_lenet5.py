# -*- coding: utf-8 -*-
"""Copy of fionaNicdao_assignment3_CNN_LeNet5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zLM9zY2CU9FNx8YBIHFzGiIqStluNDmf

# Task 1: implement your own LeNet-5 for MNIST dataset.


*   Change all architecture choices
  * number of filters
  * size of filters
  * activation (use RELU)
  * keep the pooling (size and stride) and the padding
  * use dropout (dropout rate = 0.5)
  * use L2 regularization for dense layers (except last one)

# Fiona Nicdao Assigment 3: CNN LeNet-5
"""

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# %matplotlib inline
import time

"""## processing the MNIST Dataset
* normalize the data
* 70% training and 30% testing sets
"""

# load mnist dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# normalize the data
x_train = np.expand_dims((x_train/255.0),axis=3) # add channel for color
x_test = np.expand_dims((x_test/255.0),axis=3)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Change the data to be split into 70% training set and 30% testing set
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
train_size = 0.7
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size,
                                                    random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

""" ## CNN -LeNet-5
 * Adam optimizer with 2e-4 learning rate
 * batch size = 32
 * number of training epochs = 50
 * optain the trainging and testing accuracy
"""

class LeNet(tf.keras.Model):
  def __init__(self,conv_layers, pooling_layers, fc_layers,
               input_shape, output_shape, activation, loss,
               output_activation, w_init,reg_lambda, dropout):
    super(LeNet, self).__init__()
    self.conv_layers = conv_layers
    self.pooling_layers = pooling_layers
    self.fc_layers = fc_layers
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.activation = activation
    self.optimizer = tf.keras.optimizers.Adam # set and don't change
    self.loss = loss
    self.output_activation = output_activation
    self.w_init = w_init
    self.reg_lambda = reg_lambda
    self.dropout = dropout
    self.learning_rate = 2e-4 # set and don't change
    self.batch_size = 32 # set and don't change
    self.epochs = 50 # set and don't change
    self.regularizer = tf.keras.regularizers.l2(self.reg_lambda)

    self.model = self.create_model()

  def create_model(self):
      model = tf.keras.Sequential()
      model.add(tf.keras.layers.Input(shape=self.input_shape))

      for conv, pool in zip(self.conv_layers, self.pooling_layers):
        model.add(tf.keras.layers.Conv2D(filters=conv[0], kernel_size=conv[1],
                                         padding='same',
                                         activation=self.activation,
                                         kernel_regularizer=self.regularizer,
                                         kernel_initializer=self.w_init,))
        model.add(tf.keras.layers.AveragePooling2D(pool_size=pool[0], strides=pool[1]))

      model.add(tf.keras.layers.Flatten()) # create model makes

      for fc in self.fc_layers:
        model.add(tf.keras.layers.Dense( fc, activation= self.activation,
                                         kernel_regularizer=self.regularizer,
                                         kernel_initializer=self.w_init))
      # output layer
      model.add(tf.keras.layers.Dense(self.output_shape,
                                      activation=self.output_activation,
                                      kernel_regularizer=self.regularizer,
                                      kernel_initializer=self.w_init))

      model.compile(optimizer=self.optimizer(learning_rate=self.learning_rate),
                    loss=self.loss,
                    metrics=['accuracy'])

      return model

  def train(self, x_train, y_train, x_test, y_test):
    t0 = time.time()
    history = self.model.fit(x_train, y_train,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             validation_data=(x_test, y_test))
    print("training time ", time.time()-t0)
    end_time = time.time()-t0
    return history, end_time

  def evaluate(self, x_test, y_test):
    test_loss, test_acc = self.model.evaluate(x_test, y_test)
    return test_loss, test_acc
    # return self.model.evaluate(x_test, y_test)

"""## Testing the Model with different Architecture choices
* change number of filters
* change size of filters

Keep the same
* keep the pooling(size and stride) and the padding
* use dropout rate = 0.5
* use L2 regularizatio for dense layers (except last one)
"""

# make a dataframe to compare the accuracy and loss for all the different architecture choices
df = pd.DataFrame(columns=['cov_layers','pooling_layers', 'fc_layers', 'Loss', 'Accuracy','Time'])
print(df)

"""### Test: Model 1"""

# filter_size, num_filters, input_shape, out_Shape, stride, padding
# pooling and window_size, stride, activation, optim, loss, learning rate , batch size
# epochs, regularizer, reg_lambda - strength of regularization, dropout, weight initializer

conv_layers = [[6,5], [16,5]] # number of filters , 5x5  filers
pooling_layers = [[2, 2],[2,2]] # 2x2 and stride by 2 / we have two pooling layers with 2x2 window  and 2 stride
fc_layers = [120, 84] # dense layer
input_shape = (28,28,1)
output_shape = 10
activation = 'relu'
loss = 'categorical_crossentropy'
output_activation = 'softmax'
w_init = 'random_normal' # tf.keras.initializers.HeNormal()
reg_lambda = 5e-4
dropout = 0.5

model_1 = LeNet(conv_layers, pooling_layers, fc_layers,
               input_shape, output_shape, activation, loss,
               output_activation, w_init,reg_lambda, dropout=dropout)

history, end_time = model_1.train(x_train, y_train, x_test, y_test)
test_loss, test_acc = model_1.evaluate(x_test, y_test)
df.loc[len(df)] = [conv_layers, pooling_layers, fc_layers, test_loss, test_acc, end_time]
print(df)
model_1.model.summary()

"""### Test: Model 2"""

conv_layers = [[1,6], [10,6]] # number of filters , 5x5  filers
pooling_layers = [[2, 2],[2,2]] # 2x2 and stride by 2 / we have two pooling layers with 2x2 window  and 2 stride
fc_layers = [10, 10,10] # dense layer
input_shape = (28,28,1)
output_shape = 10
activation = 'relu'
loss = 'categorical_crossentropy'
output_activation = 'softmax'
w_init = 'random_normal' # tf.keras.initializers.HeNormal()
reg_lambda = 5e-4
dropout = 0.5

model_2 = LeNet(conv_layers, pooling_layers, fc_layers,
               input_shape, output_shape, activation, loss,
               output_activation, w_init,reg_lambda, dropout=dropout)

history, end_time = model_2.train(x_train, y_train, x_test, y_test)
test_loss, test_acc = model_2.evaluate(x_test, y_test)
df.loc[len(df)] = [conv_layers, pooling_layers, fc_layers, test_loss, test_acc, end_time]
print(df)
model_2.model.summary()

"""### Test: Model 3"""

conv_layers = [[50,5], [100,5]] # number of filters , 5x5  filers
pooling_layers = [[2, 2],[2,2]] # 2x2 and stride by 2 / we have two pooling layers with 2x2 window  and 2 stride
fc_layers = [200, 100,50] # dense layer
input_shape = (28,28,1)
output_shape = 10
activation = 'relu'
loss = 'categorical_crossentropy'
output_activation = 'softmax'
w_init = 'random_normal' # tf.keras.initializers.HeNormal()
reg_lambda = 5e-4
dropout = 0.5

model_3 = LeNet(conv_layers, pooling_layers, fc_layers,
               input_shape, output_shape, activation, loss,
               output_activation, w_init,reg_lambda, dropout=dropout)

history, end_time = model_3.train(x_train, y_train, x_test, y_test)
test_loss, test_acc = model_3.evaluate(x_test, y_test)
df.loc[len(df)] = [conv_layers, pooling_layers, fc_layers, test_loss, test_acc, end_time]
print(df)
model_3.model.summary()

"""### Summary of 3 different models"""

print(df)

"""#### Results:

The first model is the one from the class / slides from week 3 module, this is my baseline model to compare my other test.

The second model I made the convolutional layers only 1 filter of 6x6 filter and then 10 filters of 6x6 filter. The same pooling layers and lowered the dense layers to [10, 10, 10]. The results is a greater loss value and a significally lower accuracy of only 11.2% so you need more filters for the model to have a better accuracy and loss value.

The third model I made the convolutional layers with 50 filters of 5x5 filter and then 100 filters of 5x5. The same pooling layers and increased the dense layers to be [200,100, 50]. The results is an loss value and accuracy similar to the first model. So adding a lot more convolutional layers and dense layers does not make the loss value and accuracy significally better.

"""