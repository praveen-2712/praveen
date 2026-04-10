import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
devices = tf.config.list_physical_devices()
print("Devices: ", devices)
