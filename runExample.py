from model.models import my_convnet_model, find_accuracy
import numpy as np
sys.path.append("src")
from tensorflow.python.keras.utils.np_utils import to_categorical
from pathlib import Path
__dir__ = Path(__file__).absolute().parent
data_save = __dir__ /'data'
num_classes = 2

# CHECK INPUT DATA SHAPE AND CHANGE ACCORDINGLY
dim1 = 32
dim2 = 56
dim3 = 3

# CREATE MODEL
model = my_convnet_model((dim1, dim2, dim3))

# RESTORE THE WEIGHTS
model.load_weights('./checkpoints/my_model_weights')

# LOAD DATA
data = np.load(data_save/"data_example.npy", allow_pickle=True)
y_ = np.load(data_save/"y_example.npy", allow_pickle=True)
y_cat = to_categorical(y_ ,  num_classes = num_classes)

# PREDICT AND CALC ACCURACY
ypred = model.predict(data)
find_accuracy(y_cat,ypred)