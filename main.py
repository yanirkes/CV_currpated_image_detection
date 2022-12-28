import pandas as pd
import numpy as np
sys.path.append("src")
from pathlib import Path
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.optimizers import adam_v2
from sklearn import model_selection as splt
from tensorflow.python.keras.utils.np_utils import to_categorical
import tensorflow  as tf
tf.config.list_physical_devices(device_type=None)
tf.config.run_functions_eagerly(True)
__dir__ = Path(__file__).absolute().parent
data_dir_ex = __dir__ /'assets/BAD/10 examples'
data_dir_large_b = __dir__ /'assets/BAD/More Examples'
data_dir_large_g = __dir__ /'assets/GOOD/More Examples'
data_save = __dir__ /'data'
num_classes = 2

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


# data = np.load(data_save+"/data.npy", allow_pickle=True)
data_2 = np.load(data_save/"data_resized.npy", allow_pickle=True)
y_ = np.load(data_save/"y.npy", allow_pickle=True)


convnet_model = my_convnet_model((data_2.shape[1], data_2.shape[2], data_2.shape[3]))
# convnet_model.summary()

y_cat = to_categorical(y_,  num_classes = num_classes)
X_train, X_test, y_train, y_test = splt.train_test_split(data_2, y_cat, test_size=0.33, random_state=42)

hist = convnet_model.fit(X_train, y_train, batch_size =128, epochs=50)
pd.DataFrame(hist.history).plot()
ypred = convnet_model.predict(X_test)
find_accuracy(y_test,ypred)