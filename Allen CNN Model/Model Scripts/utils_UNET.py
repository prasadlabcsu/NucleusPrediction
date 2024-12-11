# This module defines the utility functions like num_sort used in the main script and the DataGenerator
# class used to load, augment, and prepare the image data for training and validating the model.

# Importing packages and modules

import numpy as np
import imageio as iio
import tensorflow as tf
from augmentation_UNET import *

def num_sort(s): # Function used to sort the image file path names by their image number
    for i in range(len(s)):
        if s[i] == "g":
            if s[i+1] == "e":
                if s[i+2] != "s":
                    numstr = s[i+2]
                    for n in range(3):
                        if s[i+3+n] != ".":
                            numstr += s[i+3+n]
                        else:
                            break
    return int(numstr)

def load_img(img_files): # Function which loads the image data from their file path names into numpy arrays
    ''' Load one image and its target from file
    '''
    vol_x = iio.volread(img_files[0])
    x = np.array(vol_x)

    vol_y = iio.volread(img_files[1])
    y = np.array(vol_y)
    
    del(vol_x, vol_y)
    return x, y
    
class DataGenerator(tf.keras.utils.Sequence): # This class, inherited from the Sequence class in tensorflow, is used to generate the training and validation data set batches for training and validating the model. One training generator and one validation generator should be defined with the paths for all their image files as input.
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(75,512,512), shuffle=True, augmentation=True):
        'Initialization'
        self.list_IDs = list_IDs         # List of image path names to load data from
        self.batch_size = batch_size     # Size of the batches to use in model training
        self.dim = dim                   # Dimensions of the images
        self.shuffle = shuffle           # A flag which determines whether or not to randomly shuffle the image indices after a training epoch finishes
        self.augmentation = augmentation # A flag which determines whether or not to randomly augment the input and target images
        self.on_epoch_end()              # A function which is called at the end of a training epoch to potentially randomly shuffle the image indices if desired

    def __len__(self): # Method returning the size of the generator class = the number of batches to generate from the entire dataset
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs)/self.batch_size))

    def __getitem__(self, index): # Method for generating one batch of data to use in training or validation where 'index' is the batch number and 'indexes' is the indices of the images in the batch
        'Generate one batch of data'
        # Generate indices of the batch
        if index == self.__len__()-1:
            indexes = self.indexes[index*self.batch_size:]
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of image file path names 
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data in the batch
        x, y = self.__data_generation(list_IDs_temp) # Loads images into numpy arrays
        if self.augmentation == True:
            x, y = self.__data_augmentation(x, y) # Augments the data if the flag is TRUE
        
        if index == self.__len__()-1:
            self.on_epoch_end() # Calls the on_epoch_end method when the final batch is created
        
        return x, y

    def on_epoch_end(self): # Method for randomly shuffling the indices of the images after each epoch
        'Updates indices after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
  
    def __data_generation(self, list_IDs_temp): # Method loading the entire batch of images into numpy arrays
        'Generates data containing batch_size samples'
        # Initialization
        x = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, IDs in enumerate(list_IDs_temp):
            # Store sample
            x[i], y[i] = load_img(IDs)
            
        return x, y

    def __data_augmentation(self, x, y): # Method for augmenting the image data if the augment flad is TRUE
        'Apply augmentation'
        x_aug, y_aug = aug_batch(x, y) # This function is imported from the augmentation module
                
        return x_aug, y_aug
