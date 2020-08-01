import numpy as np
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

x_train_normalize = x_train4D / 255
x_test_normalize = x_test4D / 255

factor = 0.3
x_train_noisy = x_train_normalize + factor * np.random.normal(0, 1, x_train_normalize.shape)
x_test_noisy = x_test_normalize + factor * np.random.normal(0, 1, x_test_normalize.shape)

x_train_noisy = np.clip(x_train_noisy, 0, 1)
x_test_noisy = np.clip(x_test_noisy, 0, 1)


def show_noisy_images(start=0, end=5):
    plt.figure(figsize=(20, 4))
    for i in range(start, end):
        ax = plt.subplot(2, end, i + 1)
        plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='binary')
    plt.show()


show_noisy_images(0, 10)

# 函数式要定义输入
input_image = Input((28, 28, 1))
# 编码器 2个简单的卷积，2个最大池化，缩小为原来的1/4了
encoder = Conv2D(16, (3, 3), padding='same', activation='relu')(input_image)
encoder = MaxPooling2D((2, 2))(encoder)
encoder = Conv2D(8, (3, 3), padding='same', activation='relu')(encoder)
encoder_out = MaxPooling2D((2, 2))(encoder)

# 构建编码模型，可以提取特征图片，展开了就是特征向量
encoder_model = Model(inputs=input_image, outputs=encoder_out)

# 解码器，反过来
decoder = UpSampling2D((2, 2))(encoder_out)
decoder = Conv2D(8, (3, 3), padding='same', activation='relu')(decoder)
decoder = UpSampling2D((2, 2))(decoder)
decoder = Conv2D(16, (3, 3), padding='same', activation='relu')(decoder)
# 转成原始图片尺寸
decoder_out = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(decoder)

autoencoder = Model(input_image, decoder_out)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

autoencoder.fit(x_train_noisy, x_train_normalize, epochs=20, batch_size=256, shuffle=True, verbose=1,
                validation_data=(x_test_noisy, x_test_normalize))

autoencoder.save('./model/去噪卷积自编码器.h5')
