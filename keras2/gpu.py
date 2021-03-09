import tensorflow as tf
tf.config.list_physical_devices ()

from tensorflow.python.client import device_lib

device_lib.list_local_devices()

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


