import tensorflow
from tensorflow import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

(trainx,trainy),(testx,testy) = tensorflow.keras.datasets.fashion_mnist.load_data()

width, height = trainx.shape[1], trainx.shape[2]
batch_size = 128
epochs = 1
val_split = 0.2
latent_dim = 2
channels = 1
input_shape = (height, width, channels)
print('shape in', input_shape)
trainx = np.expand_dims(trainx, axis=-1)
testx = np.expand_dims(testx, axis=-1)
trainx = trainx.astype('float32')
testx = testx.astype('float32')
trainx = trainx/255
testx = testx/255

####
#Make the encoder
####

inp = Input(shape =input_shape, name = 'encoder_input')
cx = Conv2D(8,(3,3), (2,2), padding = 'same', activation='relu')(inp)
cx = BatchNormalization()(cx)
cx = Conv2D(16,(3,3),(2,2), padding='same', activation='relu')(cx)
cx = BatchNormalization()(cx)
x = Flatten()(cx)
x = Dense(20, activation = 'relu')(x)
x = BatchNormalization()(x)
mu = Dense(latent_dim, name = 'latent_mu')(x)
sigma = Dense(latent_dim, name = 'latent_sigma')(x)

#get the shape of the final convolutional layer
conv_shape = K.int_shape(cx)
print(conv_shape)

#sample with reparam trick
def sample_z(pair):
  mu, sigma = pair
  batch = K.shape(mu)[0]
  dim = K.int_shape(mu)[1]
  eps = K.random_normal(shape=(batch, dim))
  return mu + K.exp(sigma / 2) * eps

z = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([mu, sigma])
encoder = Model(inp, [mu, sigma, z], name = 'encoder')
#encoder.summary()


#########
#Define the decoder now
#########

d_inp = Input(shape=(latent_dim,), name = 'decoder_input')
x = Dense(conv_shape[1] * conv_shape[2]*conv_shape[3], activation='relu')(d_inp)
x = BatchNormalization()(x)
x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
cx = Conv2DTranspose(filters = 16, kernel_size = 3, strides = 2, padding = 'same', activation='relu' )(x)
cx = BatchNormalization()(cx)
cx = Conv2DTranspose(filters = 8, kernel_size = 3, strides = 2, padding = 'same', activation='relu' )(cx)
cx = BatchNormalization()(cx)
out = Conv2DTranspose(filters = channels, kernel_size=3, activation='sigmoid', padding = 'same', name = 'deocder_output')(cx)

decoder = Model(d_inp, out, name = 'decoder')
#decoder.summary()


#whole model:


VAE_output = decoder(encoder(inp)[2])
VAE = Model(inp, VAE_output, name = 'vae')

#define loss function:

def kl_reconstruction_loss(true, pred):
  # Reconstruction loss
  reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * width * height
  # KL divergence loss
  kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  # Total loss = 50% rec + 50% KL divergence loss
  return K.mean(reconstruction_loss + kl_loss)

(VAE.summary())
# Compile VAE
VAE.compile(optimizer='adam', loss=kl_reconstruction_loss)


print('compiled')
# Train autoencoder
VAE.fit(trainx, trainx, epochs = epochs, batch_size = batch_size, validation_split = val_split)


print('fitted')

# Plot results
data = (testx, testy)
#viz_latent_space(encoder, data)
#viz_decoded(encoder, decoder, data)
encoder.save('VAE_fashionmnist_encoder.h5')
decoder.save('VAE_fashionmnist_decoder.h5')
VAE.save('VAE_fashionmnist.h5')

b = encoder.predict([[data[0][0]]])
a = decoder.predict([[[.21,.34],[0,0]]])

plt.imshow(a[1], cmap='gray_r')
plt.show()


#latents = encoder.predict(trainx[:10])

#print(latents)
