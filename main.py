import pandas as pd
import numpy as np
sys.path.append("src")
from pathlib import Path
from model.models import my_convnet_model, find_accuracy
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


data_2 = np.load(data_save/"data_resized.npy", allow_pickle=True)
y_ = np.load(data_save/"y.npy", allow_pickle=True)

convnet_model = my_convnet_model((data_2.shape[1], data_2.shape[2], data_2.shape[3]))
# convnet_model.summary()

y_cat = to_categorical(y_,  num_classes = num_classes)
X_train, X_test, y_train, y_test = splt.train_test_split(data_2, y_cat, test_size=0.33, random_state=42)

# FIT
hist = convnet_model.fit(X_train, y_train, batch_size =128, epochs=50)

# Plot loss and accuracy history
pd.DataFrame(hist.history).plot()

ypred = convnet_model.predict(X_test)
find_accuracy(y_test,ypred)

# Save model
convnet_model.save_weights('./checkpoints/my_model_weights')