import numpy as np
from keras.models import load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import Model
from keras.layers import Dense,Input,MaxPooling2D,Conv2D,UpSampling2D
from keras.callbacks import TensorBoard
import warnings
import tensorflow as tf
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess =tf.compat.v1.Session(config=config)
K.set_session(sess)
warnings.filterwarnings("ignore")



np.random.seed(11)
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()
plt.imshow(x_test_image[0], cmap='binary')
print('x_train_image', len(x_train_image))
print('x_test_image', len(x_test_image))

print('y_train_label', x_train_image.shape)
print('y_train_label', y_train_label.shape)

x_train4D = x_train_image.reshape(x_train_image.shape[0], 28, 28, 1).astype('float32')
x_test4D = x_test_image.reshape(x_test_image.shape[0], 28, 28, 1).astype('float32')
print(x_train4D.shape)
print(x_test4D.shape)

x_train_normalize = x_train4D / 255.
x_test_normalize = x_test4D / 255.

factor = 0.3
x_train_noisy = x_train_normalize + factor * np.random.normal(0, 1, x_train_normalize.shape)
x_test_noisy = x_test_normalize + factor * np.random.normal(0, 1, x_test_normalize.shape)

x_train_noisy = np.clip(x_train_noisy, 0, 1)
x_test_noisy = np.clip(x_test_noisy, 0, 1)

model = load_model("../model/去噪卷积自编码器.h5")

# # 评估模型
# loss, accuracy = model.evaluate(x_test_noisy, x_test_normalize)

# print('test loss', loss)
# print('test accuracy', accuracy)

decoded_imgs = model.predict(x_test4D)
n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
