import numpy as np
from keras.models import load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import Model
from keras.layers import Dense, Input, MaxPooling2D, Conv2D, UpSampling2D
from keras.callbacks import TensorBoard
import warnings
import tensorflow as tf
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)
warnings.filterwarnings("ignore")

model = load_model("../model/卷积自编码器编码器部分.h5")
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()
x_test4D = x_test_image.reshape(x_test_image.shape[0], 28, 28, 1).astype('float32')
x_test_normalize = x_test4D / 255.
image = x_test_normalize[0]  # 28 28 1
imageX = image.reshape(-1, 28, 28, 1)
encode = model.predict(imageX)
image = image.reshape(1, 28, 28)
encode = encode.reshape(9,8)

plt.subplot(1, 2, 1)
plt.imshow(image[0])
plt.subplot(1, 2, 2)
plt.imshow(encode)
plt.show()
