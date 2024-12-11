# This module defines the 'fit' funtion used to train and validate the entire model.

# Importing packages and modules

import numpy as np
import tensorflow as tf
from models_UNET import *
from losses_UNET import *
from datetime import datetime as dt

# Calling and creating the models
U = UNET()

# Defining the optimizers (gradient calculators for updating the model parameters)
unet_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Training step function

@tf.function # Defines the train step function for tensorflow to optimize and run faster
def train_step(image, target, batch_size):
    with tf.GradientTape() as unet_tape:

        image = tf.convert_to_tensor(image) # Converts the input and nuclear images to tensorflow tensors from numpy arrays
        image = image/255
        image = tf.cast(image, dtype=tf.float32)
        target = tf.convert_to_tensor(target)
        target = tf.cast(target, dtype=tf.float32)
        
        unet_output = U(image, training=True) # Calls the Generator model with the input image
        unet_output = tf.cast(unet_output, dtype=tf.float32)
        unet_output = tf.reshape(unet_output, shape=[batch_size,75,512,512])

        reconstruction_loss = ReconstructionLoss(target,unet_output)

    unet_gradients = unet_tape.gradient(reconstruction_loss, U.trainable_variables)
    unet_optimizer.apply_gradients(zip(unet_gradients, U.trainable_variables))
      
    return reconstruction_loss
        
# Validation step function

@tf.function
def test_step(image, target, batch_size): # Very similar steps as the training, but with the training flag set to false in calling the models on the validation images
    
    image = tf.convert_to_tensor(image)
    image = image/255
    image = tf.cast(image, dtype=tf.float32)
    target = tf.convert_to_tensor(target)
    target = tf.cast(target, dtype=tf.float32)
    
    unet_output = U(image, training=False)
    unet_output = tf.cast(unet_output, dtype=tf.float32)
    unet_output = tf.reshape(unet_output, shape=[batch_size,75,512,512])

    reconstruction_loss = ReconstructionLoss(target,unet_output)

    return reconstruction_loss

# Fit training and validation function

def fit(train_gen, valid_gen, epochs, model_num):
    
    path = "/nfs/home/slawto/RESULTS_UNET"
    history_path = path + "/History_Model_" + str(model_num+1) + ".txt" # Path used for the 'History_Model_#' file
        
    with open(history_path, 'w') as f: # Create the model history file in the RESULTS folder
        pass
        
    Nt = len(train_gen)                  # Denotes the number of batches
    history = {'train': [], 'valid': []} # Allocates the history set for training and validation steps as output for the fit function
    prev_loss = np.inf                   # Defines a previous loss of infinity so that the model will be saved initially on epoch 1
    prev_e_save = 0
    epoch_reconstruction_loss_val = tf.keras.metrics.Mean()
    
    for e in range(epochs): # For each epoch, training will be done on each batch and the validation step will decide at the end if the model has improved and whether to save it
        now = str(dt.now())
        with open(history_path, 'a') as f: # Write the epoch number into the history file
            f.write('Epoch: {' + str(e+1) + '}/{' + str(epochs) + '} | Time: ' + now + '\n')
            f.write('============================================= \n')
            f.write(' \n')
        b = 0 # Tracks the current batch number
        for k in range(Nt): # For each batch of images in the training set: ...
            b += 1
            Xb, yb = train_gen.__getitem__(k) # Generate the image batch from the train_gen class
            b_size = np.shape(Xb)
            train_loss = train_step(Xb, yb, b_size[0]) # Call the training step function to calculate the losses and update the model parameters
            reconstruction_loss = train_loss.numpy()

            now = str(dt.now())
            with open(history_path, 'a') as f: # Write the losses from training into the history file
                f.write('Batch: {' + str(b) + '}/{' + str(Nt) + '} | Time: ' + now + '\n')
                f.write('Reconstruction Loss: ' + str(reconstruction_loss) + '\n')
                f.write(' \n')

        history['train'].append([reconstruction_loss]) # Add the losses from training into the history output for the fit function
        
        Nv = len(valid_gen) # Number of batches in the validation set
        for k in range(Nv): # For each validation batch: ...
            Xb, yb = valid_gen.__getitem__(k) # Generate the batch of validation images
            b_size = np.shape(Xb)
            losses_val = test_step(Xb, yb, b_size[0]) # Calculate the losses from the validation step function
            epoch_reconstruction_loss_val.update_state(losses_val)
            
        now = str(dt.now())
        with open(history_path, 'a') as f: # Write the means of the validation losses into the history file
            f.write('Validation of Epoch ' + str(e+1) + ': | Time: ' + now + '\n')
            f.write('Reconstruction Loss: ' + str(epoch_reconstruction_loss_val.result().numpy()) + '\n')
            f.write(' \n')
            
        history['valid'].append([epoch_reconstruction_loss_val.result()]) # Add the validation losses into the history output of the fit function
        
        # Save the models
        
        if epoch_reconstruction_loss_val.result().numpy() < prev_loss: # Model weights are saved into the RESULTS folder if the validation dice loss has improved
            U.save_weights(path + '/UNET_Weights_Model_' + str(model_num+1) + '_Epoch_' + str(e+1) + '.weights.h5')

            with open(history_path, 'a') as f:
                f.write('Validation reconstruction loss decreased from ' + str(prev_loss) + ' to ' + str(epoch_reconstruction_loss_val.result().numpy()) + '. Model weights are now saved. \n')
                f.write(' \n')
                
            prev_loss = epoch_reconstruction_loss_val.result().numpy()
            prev_e_save = e
        elif e > prev_e_save+3:
            U.save_weights(path + '/UNET_Weights_Model_' + str(model_num+1) + '_Epoch_' + str(e+1) + '_No_Improvement.weights.h5')

            with open(history_path, 'a') as f:
                f.write('Third epoch without decrease in validation reconstruction loss reached. Model weights have been saved as precaution. \n')
                f.write(' \n')
                
            prev_e_save = e
        elif e == epochs-1:
            U.save_weights(path + '/UNET_Weights_Model_' + str(model_num+1) + '_Epoch_' + str(e+1) + '.weights.h5')

            with open(history_path, 'a') as f:
                f.write('Last epoch reached. Model weights are now saved. \n')
                f.write(' \n')
        else: # Models aren't saved if the validation loss is worse and instead a message is written into the history file
            with open(history_path, 'a') as f:
                f.write('Validation reconstruction loss did not decrease from ' + str(prev_loss) + '. \n')
                f.write(' \n')
        
        # Reset the loss states
        epoch_reconstruction_loss_val.reset_state()
        
    return history
