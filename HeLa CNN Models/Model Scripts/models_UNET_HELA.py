# This module creates the VAE-GAN model. It is made up of 3 smaller models, the Enocder which takes in
# input images and through convolutions returns a probability distrubtion and the actual values (z) of the
# latent distribution of variables describing the predicitve qualities of the image, the Generator which
# uses the z values and sampled (pz) values from the distribution to generate images as predictions of the
# nuclear shape in 3D as a mask, and the Discriminator which takes in the original input image and a real
# or generated nucleus to determine if the image is real (1) or fake/generated (0).

# Importing packages

import numpy as np
from keras import Model
from keras import layers

# Generator model

def UNET():
    '''
    Generator model
    '''

    def encoder_step(layer, Nf, ks, norm=True): # Function defining each convolutional step of the encoder, including the 3D convolutional layer, instance normalization, and a leaky ReLU activation function
        x = layers.Conv3D(Nf, kernel_size=ks, strides=2, kernel_initializer='he_normal', padding='same')(layer)
        if norm:
            x = layers.GroupNormalization(groups=-1)(x)
        x = layers.LeakyReLU()(x)

        return x

    # Parameters required for model creation

    inputs = layers.Input((75,512,512,1), name='input_image') # Input layer of the model
    Nfilter_start = 32 # Number of filters to use initially
    depth = 6          # Depth of the model's convolutions (how many convolutional layers to make)
    ks = 4             # Kernel size of the convolution filters
    x = inputs         # Defining x as the input to use in the next layer
    
    # Calling the convolutional layers for each step of the depth
    for d in range(depth):
        if d==0:
            e0 = encoder_step(x, Nfilter_start*np.power(2,d), ks, False)
        elif d==1:
            e1 = encoder_step(e0, Nfilter_start*np.power(2,d), ks)
        elif d==2:
            e2 = encoder_step(e1, Nfilter_start*np.power(2,d), ks)
        elif d==3:
            e3 = encoder_step(e2, Nfilter_start*np.power(2,d), ks)
        elif d==4:
            e4 = encoder_step(e3, Nfilter_start*np.power(2,d), ks)
        elif d==5:
            x = encoder_step(e4, Nfilter_start*np.power(2,d), ks)

    def decoder_step(d, layer, Nf, ks): # Function defining the 3D transpose convolutional layers to rebuild an image from the input vector. Some steps require small changes in stride to achieve a close output image dimension 
        if d == 4:
            x = layers.ZeroPadding3D(padding=((1,0),(0,0),(0,0)))(layer)
            x = layers.Conv3DTranspose(Nf, kernel_size=ks, strides=[1,2,2], padding='same', kernel_initializer='he_normal')(x)
        elif d == 3:
            x = layers.ZeroPadding3D(padding=((2,0),(0,0),(0,0)))(layer)
            x = layers.Conv3DTranspose(Nf, kernel_size=ks, strides=[1,2,2], padding='same', kernel_initializer='he_normal')(x)
        elif d == 2:
            x = layers.Conv3DTranspose(Nf, kernel_size=ks, strides=[2,2,2], padding='same', kernel_initializer='he_normal')(layer)
        elif d == 1:
            x = layers.Conv3DTranspose(Nf, kernel_size=ks, strides=[2,2,2], padding='same', kernel_initializer='he_normal')(layer)
            x = layers.Cropping3D(cropping=((1,0),(0,0),(0,0)))(x)
        elif d == 0:
            x = layers.Conv3DTranspose(Nf, kernel_size=ks, strides=[2,2,2], padding='same', kernel_initializer='he_normal')(layer)
        x = layers.GroupNormalization(groups=-1)(x)
        x = layers.LeakyReLU()(x)
        if d==4: 
            x = layers.Concatenate()([e4, x])
        elif d==3: 
            x = layers.Concatenate()([e3, x])
        elif d==2: 
            x = layers.Concatenate()([e2, x])
        elif d==1:
            x = layers.Concatenate()([e1, x])
        elif d==0: 
            x = layers.Concatenate()([e0, x])
        return x
    
    # 3D transpose convolutional layers are called
    
    for d in range(depth-2, -1, -1):
        x = decoder_step(d, x, Nfilter_start*np.power(2,d), ks)

    # Final image prediction layer is created with one last 3D tranpose convolution with a sigmid activation function and cropping down to the right image dimensions
    x = layers.Conv3DTranspose(1, kernel_size=ks, strides=2, padding='same', kernel_initializer='he_normal', activation='sigmoid', name='output_generator')(x)
    last = layers.Cropping3D(cropping=((1,0),(0,0),(0,0)))(x)

    return Model(inputs=[inputs], outputs=last, name='Generator')
