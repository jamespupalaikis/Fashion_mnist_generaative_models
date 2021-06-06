import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as k

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten,LeakyReLU, Conv2D,Embedding,MaxPool2D,Flatten,Dropout,Reshape,Conv2DTranspose,Input,Concatenate
from tensorflow.keras.optimizers import Adam



def define_discriminator(in_shape = (28,28,1), num_classes = 10):
    in_label = Input(shape=(1,))
    li = Embedding(num_classes,50)(in_label)
    #scale up dimensions
    n_nodes = in_shape[0]*in_shape[1]
    li = Dense(n_nodes)(li)
    #reshape to additional channel
    li = Reshape((in_shape[0], in_shape[1], 1))(li)
    #image input
    in_image = Input(shape=in_shape)
    #concatenate the label as a channel
    merge = Concatenate()([in_image, li])
    #downsample that shittt
    #########
    fe = Conv2D(128,(3,3),(2,2),padding='same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv2D(128, (3, 3), (2, 2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    #flatten it
    ############
    fe = Flatten()(fe)
    fe = Dropout(.4)(fe)
    #output
    out_layer = Dense(1,activation='sigmoid')(fe)
    #define model
    ###############
    model = Model([in_image,in_label],out_layer)

    model.compile(loss = 'binary_crossentropy', optimizer=Adam(lr = 0.0002,beta_1=0.5),
                  metrics = ['accuracy'])
    model.summary()
    return model

def define_generator(latent_dim,n_classes = 10):
    #n_nodes = 128 * 7 * 7 #for creating 128 copies of 7x7 res output images for upsampling
    in_label = Input(shape=(1,))
    #embedding for categories
    li = Embedding(n_classes,50)(in_label)
    n_nodes = 7*7
    li = Dense(n_nodes)(li)
    #reshape to additional channel
    li = Reshape((7,7,1))(li)
    #image gen input
    in_lat = Input(shape=(latent_dim,))
    #foundation for 7x7 image to be upscaled
    n_nodes = 128 * 7 * 7
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha = 0.2)(gen)
    gen = Reshape((7,7,128))(gen)
    #merge now
    merge = Concatenate()([gen,li])
    #up to 14x14, then 28x28
    gen = Conv2DTranspose(128,(4,4),(2,2),padding='same')(merge)
    gen = LeakyReLU(alpha = 0.2)(gen)
    gen = Conv2DTranspose(128, (4, 4), (2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    out_layer = Conv2D(1,(7,7),activation='tanh',padding = 'same')(gen)

    model = Model([in_lat,in_label],out_layer)
    model.summary()
    return model




def define_gan(generator, discriminator):
    discriminator.trainable = False
    #get noise and label inputs from generator model
    gen_noise,gen_label = generator.input
    #get image output from gen model
    gen_output = generator.output
    #connect the image and label as disc inputs
    gen_output = discriminator([gen_output,gen_label])
    #define our model as taking laten points and labels,
    #outputting a proibability
    model = Model([gen_noise, gen_label],gen_output)


    model.compile(loss = 'binary_crossentropy', optimizer =
                  Adam(lr = 0.0002, beta_1=0.5))
    model.summary()
    return model


def load_real_samples():
    (trainx,trainy),(_,_) = k.datasets.fashion_mnist.load_data()
    #expand to 3d
    x = np.expand_dims(trainx,axis = -1)
    x = x.astype('float32')
    x = (x-127.5)/127.5
    return [x,trainy]

def gen_real_samples(dataset, n_samples):
    images, label = dataset

    ix = np.random.randint(0,images.shape[0],n_samples)

    X,labels = images[ix],label[ix]
    y = np.ones((n_samples,1))
    return [X,labels],y

def gen_latent_points(latent_dim, n_samples,n_classes = 10):

    xinput = np.random.randn(latent_dim * n_samples)
    xinput = xinput.reshape(n_samples,latent_dim)

    labels = np.random.randint(0,n_classes,n_samples)
    return [xinput,labels]

def gen_fake_samples(generator,latent_dim,n_samples):
    xinput,labels_input = gen_latent_points(latent_dim,n_samples)
    images = generator.predict([xinput,labels_input])
    y = np.zeros((n_samples,1))
    return [images, labels_input],y

def train(gen, disc, gan, dataset, latent_dim, n_epochs, n_batch):
    bat_per_epo = int(dataset[0].shape[0]/n_batch) #IDK what this is
    halfbatch = int(n_batch/2)

    for i in range(n_epochs):
        #enum batches over training set
        for j in range(bat_per_epo):
            [xreal,labelsreal],yreal = gen_real_samples(dataset,halfbatch)
            [xfake,labels],yfake = gen_fake_samples(gen,latent_dim,halfbatch)
            disc_loss_1,_ = disc.train_on_batch([xreal,labelsreal],yreal)
            disc_loss_2,_ = disc.train_on_batch([xfake,labels], yfake)

            [xgan,labels_input] = gen_latent_points(latent_dim,n_batch)
            ygan = np.ones((n_batch,1))

            gan_loss = gan.train_on_batch([xgan,labels_input],ygan)

            print('>%d,%d/%d, d1 = %.3f, d2 = %.3f g = %.3f'%
                  (i + 1, j + 1, bat_per_epo,disc_loss_1,disc_loss_2,gan_loss))

    gen.save('fashionmnistgen.h5')


latent_dim = 100
discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan = define_gan(generator, discriminator)

dataset = load_real_samples()

epochs = 100

train(generator,discriminator,gan,dataset,latent_dim,epochs,100)




