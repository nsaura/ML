from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

import sys

import matplotlib.pyplot as plt

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
#print "Conv2D 16 shape : {}".format(x.get_shape())
x = MaxPooling2D((2, 2), padding='same')(x)
#print "MaxPooling2D 1 shape : {}".format(x.get_shape())
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#print "Conv2D 8 shape : {}".format(x.get_shape())
x = MaxPooling2D((2, 2), padding='same')(x)
#print "MaxPooling2D 2 shape : {}".format(x.get_shape())
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#print "Conv2D (2) 8 shape : {}".format(x.get_shape())
encoded = MaxPooling2D((2, 2), padding='same')(x)
#print "MaxPooling2D encoded 2 shape : {}".format(encoded.get_shape())
#sys.exit()

#Conv2D 16 shape : (?, 28, 28, 16)
#MaxPooling2D 1 shape : (?, 14, 14, 16)
#Conv2D 8 shape : (?, 14, 14, 8)
#MaxPooling2D 2 shape : (?, 7, 7, 8)
#Conv2D (2) 8 shape : (?, 7, 7, 8)
#MaxPooling2D encoded 2 shape : (?, 4, 4, 8)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional <--------- OKAY

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#print "Conv2D 8 shape : {}".format(x.get_shape())
x = UpSampling2D((2, 2))(x)
#print "Upsampling shape : {}".format(x.get_shape())
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#print "Conv2D (2) 8 shape : {}".format(x.get_shape())
x = UpSampling2D((2, 2))(x)
#print "Upsampling (2) shape : {}".format(x.get_shape())
x = Conv2D(16, (3, 3), activation='relu')(x)
#print "Conv2D 16 shape : {}".format(x.get_shape())
x = UpSampling2D((2, 2))(x)
#print "Upsampling (last) shape : {}".format(x.get_shape())
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#print "Conv2D 1 decode shape : {}".format(decoded.get_shape())
#sys.exit()

#Conv2D 8 shape : (?, 4, 4, 8)
#Upsampling shape : (?, 8, 8, 8)
#Conv2D (2) 8 shape : (?, 8, 8, 8)
#Upsampling (2) shape : (?, 16, 16, 8)
#Conv2D 16 shape : (?, 14, 14, 16)
#Upsampling (last) shape : (?, 28, 28, 16)
#Conv2D 1 decode shape : (?, 28, 28, 1)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format


from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
                
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n +1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

plt.ion()

