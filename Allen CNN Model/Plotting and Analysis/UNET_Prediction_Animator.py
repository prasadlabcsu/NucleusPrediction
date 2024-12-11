# Importing packages

import tensorflow as tf
from keras import Model
from keras import layers
import imageio as iio
import matplotlib.pyplot as plt
import matplotlib.animation as manim
import numpy as np
from scipy.ndimage import median_filter

# Params
epoch_num = 20
cutoff = 0.42

# mn_list = [6,2,8,1,1,3,2,4,1,4,10,6,3,10,9,9,1,1,1,3,8,6,1,3,10,9,9,10,2]
# in_list = [43,3079,422,10,2930,3256,2495,2555,1485,2432,2698,398,2951,1239,2796,3434,2088,2374,2732,1015,2363,179,895,2079,3584,2192,3636,1814,2985]

mn_list = [9]
in_list = [2796]

for i in range(len(mn_list)):
    model_num = mn_list[i]
    im_num = in_list[i]
    no_imp = False

    if model_num == 3 or model_num == 7 or model_num == 8:
        no_imp = True

    # Load Validation Set Sample Numbers
    s_file = open("C:\\Users\\sebas\\OneDrive\\Desktop\\HPC UNET Files\\Sets_Model_" + str(model_num) + ".txt")
    s_text = s_file.read()
    valid_set = []

    for k in range(len(s_text)):
        cursor = s_text[k]
        counter = k
        if cursor == 'T':
            if s_text[k+1] == 'r':
                break
        if cursor == '.':
            samp_num = ''
            counter -= 1
            cursor = s_text[counter]
            while cursor != 'e':
                samp_num += cursor
                counter -= 1
                cursor = s_text[counter]
            valid_set.append(int(samp_num[::-1]))

    valid_set = set(valid_set)
    valid_set = list(valid_set)

    if im_num == 0:
        im_num = valid_set[idx_num]

    # Define Model
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

    # Create Model
    U = UNET()

    # Load Weights
    if no_imp==False:
        U.load_weights("C:\\Users\\sebas\\OneDrive\\Desktop\\HPC UNET Files\\UNET_Weights_Model_" + str(model_num) + "_Epoch_" + str(epoch_num) + ".weights.h5")
    else:
        U.load_weights("C:\\Users\\sebas\\OneDrive\\Desktop\\HPC UNET Files\\UNET_Weights_Model_" + str(model_num) + "_Epoch_" + str(epoch_num) + "_No_Improvement.weights.h5")

    # Load input image
    vol_in = iio.volread("C:\\Users\\sebas\\OneDrive\\Desktop\\Actin_Images_TIFF\\Actin_Images_Interphase_512_Membrane_Mask_Standard_Size_Image" + str(im_num) + ".tiff")
    t_in = tf.convert_to_tensor(vol_in)
    t_in = t_in/255
    t_in = tf.cast(t_in, dtype=tf.float32)
    t_in = tf.reshape(t_in, shape=[1,75,512,512,1])

    # Predict Nuclear Shape
    t_out = U(t_in, training=False)
    t_out = tf.reshape(t_out, shape=[75,512,512])

    # Plot Predicted Nuclear Shape
    vol_out = t_out.numpy()
    vol_out_pred = np.round(vol_out*255)

    # Plot Predicted Binary Nucleus
    vol_out_bin = np.where(vol_out > cutoff, 255, 0)
    vol_out_bin = median_filter(vol_out_bin, size=3)

    # Plot True Nucleus
    vol_true = iio.volread("C:\\Users\\sebas\\OneDrive\\Desktop\\Actin_Images_TIFF\\Actin_Images_Interphase_512_Nucleus_Standard_Size_Image" + str(im_num) + ".tiff")
    vol_true_255 = vol_true*255

    # Plot Shifted Nucleus
    z, y, x = np.nonzero(vol_true)
    xyz_center_true = np.round(np.array([np.mean(z), np.mean(y), np.mean(x)]))
    z, y, x = np.nonzero(vol_out_bin)
    zyx = np.nonzero(vol_out_bin)
    if np.size(zyx) == 0:
        dice_loss_shift = tf.constant(0,shape=(),dtype=tf.float32)
    else:
        xyz_center_bin = np.round(np.array([np.mean(z), np.mean(y), np.mean(x)]))
        dev = xyz_center_true - xyz_center_bin
        vol_out_shift = np.zeros_like(vol_out_bin)
        while int(np.max(zyx[0])+dev[0]) > 74:
            dev[0] -= 1
        for m in range(np.size(z)):
            vol_out_shift[int(zyx[0][m]+dev[0]),int(zyx[1][m]+dev[1]),int(zyx[2][m]+dev[2])] = 255

    # Combined Prediction Plot 
    vol_in_outline = iio.volread("C:\\Users\\sebas\\OneDrive\\Desktop\\Actin_Images_TIFF\\Actin_Images_Interphase_512_Membrane_Standard_Size_Image" + str(im_num) + ".tiff")
    vol_in_outline = vol_in_outline*255
    #vol_combined = vol_in_outline + vol_true_255
    vol_combined = vol_true_255
    vol_combined = np.reshape(vol_combined, newshape=[75,512,512,1])
    vol_combined_RGB = np.concatenate((vol_combined,vol_combined,vol_combined), axis=3)
    vol_out_pred = np.reshape(vol_out_pred, newshape=[75,512,512,1])
    vol_out_pred_RGB = np.concatenate((np.zeros_like(vol_out_pred),vol_out_pred,np.zeros_like(vol_out_pred)), axis=3)
    vol_combined_RGB = vol_combined_RGB - vol_out_pred_RGB
    vol_combined_RGB = np.absolute(vol_combined_RGB)
    vol_combined_RGB = vol_combined_RGB.astype(int)

    # Combined Binary Prediction Plot
    vol_combined_bin_RGB = np.concatenate((vol_combined,vol_combined,vol_combined), axis=3)
    vol_out_bin_r = np.reshape(vol_out_bin, newshape=[75,512,512,1])
    vol_out_bin_RGB = np.concatenate((np.zeros_like(vol_out_bin_r),vol_out_bin_r,np.zeros_like(vol_out_bin_r)), axis=3)
    vol_combined_bin_RGB = vol_combined_bin_RGB - vol_out_bin_RGB
    vol_combined_bin_RGB = np.absolute(vol_combined_bin_RGB)
    vol_combined_bin_RGB = vol_combined_bin_RGB.astype(int)

    # Combined Shifted Binary Prediction Plot
    vol_combined_shift_RGB = np.concatenate((vol_combined,vol_combined,vol_combined), axis=3)
    vol_out_shift_r = np.reshape(vol_out_shift, newshape=[75,512,512,1])
    vol_out_shift_RGB = np.concatenate((np.zeros_like(vol_out_shift_r),vol_out_shift_r,np.zeros_like(vol_out_shift_r)), axis=3)
    vol_combined_shift_RGB = vol_combined_shift_RGB - vol_out_shift_RGB
    vol_combined_shift_RGB = np.absolute(vol_combined_shift_RGB)
    vol_combined_shift_RGB = vol_combined_shift_RGB.astype(int)

    # Define Dice Loss
    def DiceLoss(target, gen_output):

        num = tf.reduce_sum(target*gen_output)
        den = tf.reduce_sum(target+gen_output)
        loss = 2*num/den

        return loss

    # Calculate Dice Score of Predicted Nucleus
    t_true = tf.convert_to_tensor(vol_true)
    t_true = tf.cast(t_true, dtype=tf.float32)
    dice_loss_pred = DiceLoss(t_true,t_out)

    # Calculate Dice Score of Predicted Binary Nucleus
    t_out_bin = tf.convert_to_tensor(vol_out_bin/255)
    t_out_bin = tf.cast(t_out_bin, dtype=tf.float32)
    dice_loss_bin = DiceLoss(t_true,t_out_bin)

    # Calculate Dice Score of Shifted Nucleus
    t_out_shift = tf.convert_to_tensor(vol_out_shift/255)
    t_out_shift = tf.cast(t_out_shift, dtype=tf.float32)
    dice_loss_shift = DiceLoss(t_true,t_out_shift)

    # Print Dice Score Results
    print("Predicted Nucleus Dice Score:", dice_loss_pred.numpy())
    print("Predicted Binary Nucleus Dice Score:", dice_loss_bin.numpy())
    print("Predicted Shifted Binary Nucleus Dice Score:", dice_loss_shift.numpy())

    # Animation
    FFMpegWriter = manim.writers['ffmpeg']
    writer = FFMpegWriter(fps=6)
    fig = plt.figure()
    dim_z, _, _ = np.nonzero(np.array(vol_in_outline))
    z_range = range(np.min(dim_z),np.max(dim_z)+1)

    plt.title("Dice Score: " + str(dice_loss_pred.numpy()))
    with writer.saving(fig, "C:\\Users\\sebas\\OneDrive\\Desktop\\UNET Animations\\Image_" + str(im_num) + "_Model_" + str(model_num) + "_Combined_Prediction.mp4", 200):
        for z_frame in z_range:
            plt.imshow(vol_combined_RGB[z_frame,:,:,:])
            writer.grab_frame()

    plt.title("Dice Score: " + str(dice_loss_bin.numpy()))
    with writer.saving(fig, "C:\\Users\\sebas\\OneDrive\\Desktop\\UNET Animations\\Image_" + str(im_num) + "_Model_" + str(model_num) + "_Combined_Binary_Prediction.mp4", 200):
        for z_frame in z_range:
            plt.imshow(vol_combined_bin_RGB[z_frame,:,:,:])
            writer.grab_frame()

    plt.title("Dice Score: " + str(dice_loss_shift.numpy()))
    with writer.saving(fig, "C:\\Users\\sebas\\OneDrive\\Desktop\\UNET Animations\\Image_" + str(im_num) + "_Model_" + str(model_num) + "_Combined_Shifted_Prediction.mp4", 200):
        for z_frame in z_range:
            plt.imshow(vol_combined_shift_RGB[z_frame,:,:,:])
            writer.grab_frame()
