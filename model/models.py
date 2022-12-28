from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.optimizers import adam_v2
import numpy as np

def find_accuracy(y_test, y_pred):
  scoreboard = (y_test.argmax(axis=1) == y_pred.argmax(axis=1))
  return np.mean(scoreboard)

def my_convnet_model(input_shape):
  inputs = layers.Input(shape=input_shape)  # The actual shape is going to be [batch_size, input_shape]

  # I'm not going to "overflow" the namespace, lets just stick with x all the way but the last levels.
  x = layers.Conv2D(32, (15, 15), padding='same', name="Conv1", data_format="channels_last")(inputs)
  x = layers.Activation("relu")(x)
  x = layers.Conv2D(32, (15, 15), padding='same', name="Conv2", data_format="channels_last")(x)
  x = layers.Activation("relu")(x)
  x = layers.MaxPooling2D(pool_size=(2, 2), name="Pool1", data_format='channels_last')(x)
  x = layers.Dropout(0.25)(x)

  x = layers.Conv2D(64, (15, 15), padding='same', name="Conv3", data_format="channels_last")(x)
  x = layers.Activation("relu")(x)
  x = layers.Conv2D(64, (15, 15), padding='same', name="Conv4", data_format="channels_last")(x)
  x = layers.Activation("relu")(x)
  x = layers.MaxPooling2D(pool_size=(2, 2), name="Pool2", data_format='channels_last')(x)
  x = layers.Dropout(0.25)(x)
  x = layers.Flatten()(x)
  x = layers.Dense(units=64, name="FC1")(x)
  x = layers.Activation("relu")(x)
  x = layers.Dense(2, name="FC2")(x)

  predictions = layers.Activation("sigmoid", name="outputs")(x)

  # Here we "wrap" the entire flow, the inputs are "inputs" variable and the outputs are "predictions" variable
  model = models.Model(inputs=inputs, outputs=predictions)
  opt = adam_v2.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999)
  model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

  return model