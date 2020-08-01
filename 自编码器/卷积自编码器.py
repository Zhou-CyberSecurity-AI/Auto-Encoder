import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, add
from keras.layers import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils
import warnings
import tensorflow as tf
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)
warnings.filterwarnings("ignore")


class CNN_VAE():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.latent_dim = 1024
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.features = (3, 3, 8)
        # 定义优化器
        self.adam = Adam(lr=1e-4)
        self.loss = 'mse'
        self.Encode = self.encoder()
        self.Decode = self.decoder()

        img = Input(shape=self.img_shape)
        images_encode = self.Encode(img)

        validity = self.Decode(images_encode)

        self.epochs = 20
        self.batch_size = 128
        self.nb_classes = 10

        self.combined = Model(img, validity)
        history = self.train()
        self.showlossAcc(history)

    def encoder(self):
        images = Input(shape=self.img_shape)
        model = Sequential()
        # 解码器
        model.add(Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=self.img_shape,
            padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(Conv2D(
            filters=8,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=self.img_shape,
            padding='same'))
        model.add(Conv2D(
            filters=8,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(
            filters=8,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.save('./model/卷积自编码器编码器部分.h5')
        model.summary()
        outputs = model(inputs=images)
        return Model(images, outputs)

    def decoder(self):
        # 解码器
        model = Sequential()
        model.add(Conv2D(
            filters=8,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=self.features,
            padding='same'))
        model.add(UpSampling2D((2, 2)))  # 6 6 8
        model.add(Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation='relu'))
        model.add(UpSampling2D((2, 2)))  # 12 12 16
        model.add(Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(
            filters=1,
            kernel_size=(3, 3),
            activation='sigmoid',
            ))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(
            filters=1,
            kernel_size=(3, 3),
            activation='sigmoid',
            padding='same'
        ))
        model.summary()
        encode_out = Input(shape=self.features)
        validty = model(encode_out)
        return Model(encode_out, validty)

    def train(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        # 归一化
        X_train = X_train.astype("float32") / 255.
        X_test = X_test.astype("float32") / 255.

        # 标签one-hot编码
        y_train = np_utils.to_categorical(y_train, self.nb_classes)
        y_test = np_utils.to_categorical(y_test, self.nb_classes)

        self.combined.compile(optimizer=self.adam, loss=self.loss, metrics=['accuracy'])
        # self.combined.summary()
        history = self.combined.fit(X_train, X_train, epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                                    validation_data=(X_test, X_test))

        return history

    def showlossAcc(self, history):
        print(history.history.keys())
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()


def main():
    cnnVAE = CNN_VAE()
    cnnVAE.train()


if __name__ == '__main__':
    main()
