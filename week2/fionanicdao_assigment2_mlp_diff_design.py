# -*- coding: utf-8 -*-
"""fionaNicdao_assigment2_mlp_diff_design.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gsGyOMtXAmUaqE6rRN1NAaSghANkszRi

# Fiona Nicdao's Assignment 2
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import  keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

"""## Processing the MNIST Dataset"""

#build the model based on the data

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Change the data to be split into 70% training set and 30% testing set
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
train_size = 0.7
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size,
                                                    random_state=42)

dev_size = 0.8 * x_train.shape[0]
dev_size = int(dev_size)

#shuffle the x_train (good practice)
#seed for reproducibility
indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

# plot the image
plt.imshow(x_train[0], cmap='gray')
plt.show()

#dividing the training dataset into 80/20 : training set/ validation set
x_val = x_train[dev_size:] #validation sets
y_val = y_train[dev_size:]

x_train = x_train[:dev_size] #training sets
y_train = y_train[:dev_size]

#preparing training data
#dividing them by max pixel value as a float to get all values btw 0 and 1
x_train = (x_train/255.0).reshape(-1, 28*28)
x_val = (x_val/255.0).reshape(-1, 28*28)
x_test = (x_test/255.0).reshape(-1, 28*28)

#make the classes one-hot encodings
y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)
y_test = tf.keras.utils.to_categorical(y_test)

print(x_train.shape) #6000 training samples, image is 28x28 size
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#check step that the data is normalized between [1.0, 0.0]
x_train[0].max(), x_train[0].min()
# better to have it float values /

"""## Building the Model : MLP"""

#model
class MLP(tf.keras.Model):
  def __init__(self, num_classes, input_shape, n_layers, n_units, activation,
               optim, loss, initializer,reg):
      super(MLP, self).__init__()
      self.num_classes = num_classes
      self.input_shape = input_shape
      self.n_layers = n_layers
      self.n_units = n_units
      self.activation = activation
      self.optimizer = optim
      self.loss = loss
      self.initializer = initializer
      self.regularizer = reg

      self.model = self.create_model()

  #build the structure of the model
  def create_model(self):
    model = tf.keras.Sequential() # Sequential model is just a placeholder
    model.add(tf.keras.layers.Input(shape=self.input_shape))

    for i in range(self.n_layers):
      model.add(tf.keras.layers.Dense(self.n_units,
                                      input_shape=self.input_shape,
                                      activation=self.activation,
                                      kernel_initializer = self.initializer,
                                      kernel_regularizer= self.regularizer))

    model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

    return model

  def compile_model(self):
    self.model.compile(optimizer=self.optimizer, loss=self.loss,
                       metrics=['accuracy'])

  def train_model(self, x_train, y_train, x_val, y_val, epochs=10,
                  batch_size=32):
    self.model.fit(x_train, y_train, epochs=epochs, batch_size=64,
                   validation_data=(x_val, y_val))

  def evaluate_model(self, x_test, y_test):
    test_loss, test_acc = self.model.evaluate(x_test, y_test)
    return test_loss, test_acc

"""# Task 1
## compare the performance of 2-layer vs 3-layer vs 4-layer MLPs on MNIST dataset

"""

# make a dataframe to compare the accuracy and loss for all the different activation
df = pd.DataFrame(columns=['Number of Layers', 'Loss', 'Accuracy','Time'])
df['Number of Layers'] = df['Number of Layers'].astype(np.int32)

layers = [2, 3, 4]
for layer in layers:
  mlp_layer = MLP(num_classes=10,
            input_shape=(28*28,),
            n_layers=layer,
            n_units=100,
            activation='relu',
            optim= tf.keras.optimizers.Adam(learning_rate=0.0002),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            initializer=tf.keras.initializers.RandomNormal(),
            reg = tf.keras.regularizers.l2(0.001))
  mlp_layer.compile_model()
  start = time.time()
  mlp_layer.train_model(x_train, y_train, x_val, y_val, epochs=50, batch_size=32)
  end = time.time()
  print(f"Training time for MLP {layer}-layer : {end - start} seconds")
  test_loss, test_acc = mlp_layer.evaluate_model(x_test, y_test)
  df.loc[len(df)] = [layer, test_loss, test_acc, end - start]
  mlp_layer.summary()

df['Number of Layers'] = df['Number of Layers'].astype(np.int32)
print(df)

"""Results:  Minimum difference with the loss and accuracy for all three different layers. 2-layer has the fastest time and 4-layer has the slowest time.

# TASK 2
## Compare the performance of 2-layer MLP when using different settings

### different weight initialization (RandomNormal, zeros, ones, GlorotNormal)
### different regularization (l1, l2, l1_l2)
### different optimizers (SDG , ADAM, Ftrl)

### Default weight initialization: RandomNormal
### Default regularization: l2
### Default optimizer: SDG
"""

# make a dataframe to compare the accuracy and loss for all the different activation
df_task2 = pd.DataFrame(columns=['weight init','regularization',
                           'optimizer', 'Loss', 'Accuracy','Time'])

"""### Different Weight Initialization (RandomNormal, zeros, ones, GlorotNormal)"""

# try : RandomNormal, zeros, ones, GlorotNormal

intialweights = [{'name': 'RandomNormal','obj':tf.keras.initializers.RandomNormal()},
                 {'name': 'zeros','obj':tf.keras.initializers.zeros()},
                 {'name': 'ones','obj':tf.keras.initializers.ones()},
                 {'name': 'GlorotNormal','obj':tf.keras.initializers.GlorotNormal()}]
for idx, initweight in enumerate(intialweights):
  mlp_idx = MLP(num_classes=10,
            input_shape=(28*28,),
            n_layers=2,
            n_units=100,
            activation='relu',
            optim= tf.keras.optimizers.SGD(learning_rate=0.0002),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            initializer=initweight['obj'],
            reg = tf.keras.regularizers.l2(0.001))

  mlp_idx.compile_model()

  start = time.time()
  mlp_idx.train_model(x_train, y_train, x_val, y_val, epochs=50, batch_size=32)
  end = time.time()

  print(f"Training time for MLP_{idx} task 2 {initweight['name']} is the weight intialization: {end - start} seconds")
  print("\n")

  test_loss, test_acc = mlp_idx.evaluate_model(x_test, y_test)

  df_task2.loc[len(df_task2)] =[initweight['name'],'l2', 'SGD',test_loss, test_acc, end - start]

  mlp_idx.summary()

"""### Different regularizations (l1, l2, l1_l2)"""

# try : l1, l2, l1_l2
reguleriers = [{'name': 'l1', 'obj': tf.keras.regularizers.l1(0.001)},
               {'name': 'l2', 'obj': tf.keras.regularizers.l2(0.001)},
               {'name': 'l1_l2', 'obj': tf.keras.regularizers.l1_l2(0.001)}]
for idx, reg in enumerate(reguleriers):
  mlp_idx = MLP(num_classes=10,
            input_shape=(28*28,),
            n_layers=2,
            n_units=100,
            activation='relu',
            optim= tf.keras.optimizers.SGD(learning_rate=0.0002),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            initializer=tf.keras.initializers.RandomNormal(),
            reg = reg['obj'])

  mlp_idx.compile_model()

  start = time.time()
  mlp_idx.train_model(x_train, y_train, x_val, y_val, epochs=50, batch_size=32)
  end = time.time()

  print(f"Training time for MLP task 2 {reg['name']} regularization: {end - start} seconds")
  print("\n")

  test_loss, test_acc = mlp_idx.evaluate_model(x_test, y_test)

  df_task2.loc[len(df_task2)] =['RandomNormal',reg['name'], 'SGD',test_loss, test_acc, end - start]

  mlp_idx.summary()

"""### Different optimizers (SDG , ADAM, Ftrl)"""

#try : SDG , ADAM, Ftrl
optimizers = [{'name':'SGD','obj':tf.keras.optimizers.SGD(learning_rate=0.0001)},
              {'name':'Adam','obj':tf.keras.optimizers.Adam(learning_rate=0.0001)},
              {'name':'Ftrl','obj':tf.keras.optimizers.Ftrl(learning_rate=0.0001)}]
for idx, optimizer in enumerate(optimizers):
  mlp_idx = MLP(num_classes=10,
            input_shape=(28*28,),
            n_layers=2,
            n_units=100,
            activation='relu',
            optim= optimizer['obj'],
            loss=tf.keras.losses.CategoricalCrossentropy(),
            initializer=tf.keras.initializers.RandomNormal(),
            reg = tf.keras.regularizers.l2(0.001))

  mlp_idx.compile_model()

  start = time.time()
  mlp_idx.train_model(x_train, y_train, x_val, y_val, epochs=50, batch_size=32)
  end = time.time()

  print(f"Training time for MLP task 2 {optimizer['name']} is the optimizer: {end - start} seconds")
  print("\n")

  test_loss, test_acc = mlp_idx.evaluate_model(x_test, y_test)

  df_task2.loc[len(df_task2)] =['RandomNormal', 'l2', optimizer['name'],test_loss, test_acc,end - start]

  mlp_idx.summary()

"""## Compare the accuracy, loss, time for all the different designs"""

print(df_task2)

"""# Results:
The best for this dataset is ...

Best weight initalization by max accuracy and min loss is **GlorotNormal**.

Regularization accuracy is about the same for all (l1, l2, l3) with **l2** with the lowest loss and the fastest time is **l1_l2**.

Best optimizer is **Adam** with the highest accuracy at 97% and lowest loss.  

"""