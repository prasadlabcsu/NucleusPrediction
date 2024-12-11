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
group = "Y27"
no_imp = False
epoch_num = 10
cutoff = 0.55

# CTL
# mn_list = [9,9,3,3,3,9,7,10,8,2,1,2,2,7,6,1,3,9,8,8,1,5,8,1,2,1,1,2,1,9,9,4,5,9,5]
# in_list = [318,276,423,977,823,1120,1266,1246,747,873,1034,637,264,496,625,1275,862,751,1272,905,422,650,7,62,632,841,1337,1090,1222,206,734,880,1270,183,943]

# Ble
# mn_list = [2,7,2,9,2,7,8,10,3,2,8,5,3,1,1,3,1,5,6,3,6,9,2,1,9,3,9,1,1,1,9,10,2,8,3,9,9,4,2]
# in_list = [524,2700,1753,1322,589,394,1870,1601,647,762,2309,1294,2608,18,1560,340,262,2033,2642,2252,959,2271,1654,988,1807,1531,2378,521,23,1477,2709,210,1948,678,1248,2649,628,1340,1245]

# H11
# mn_list = [8,6,9,1,4,9,1,4,8,3,7,5,2,4,5,1,4,6,3,3,4,7,7,1,4,1,9,4,5,1,1,8,5,2,3,7,6,3,9,9,6]
# in_list = [1145,1429,290,1056,701,1003,1805,577,466,1199,2051,284,663,1544,465,517,640,1440,1764,1926,1256,1001,994,1985,1423,1553,693,325,1206,14,2043,375,1438,134,1536,1920,447,286,808,274,1504]

# Y27
# mn_list = [8,8,9,8,10,10,5,6,4,9,8,4,3,1,1,5,9,4,10,1,1,6,6,3,3,9,8,3,1,2,10,8,2,8,3,2,3,5,8]
# in_list = [1590,1095,390,414,409,158,319,1314,1054,1257,1191,1312,1261,1200,43,1317,699,357,715,33,1215,1538,1349,1419,596,174,384,1116,1246,187,63,1128,1188,230,1420,423,500,517,298]

mn_list = [8,8,9,8,10,10,5,6,4,9,8,4,3,1,1,5,9,4,10,1,1,6,6,3,3,9,8,3,1,2,10,8,2,8,3,2,3,5,8]
in_list = [1590,1095,390,414,409,158,319,1314,1054,1257,1191,1312,1261,1200,43,1317,699,357,715,33,1215,1538,1349,1419,596,174,384,1116,1246,187,63,1128,1188,230,1420,423,500,517,298]

for i in range(len(mn_list)):
    model_num = mn_list[i]
    im_num = in_list[i]

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
        U.load_weights("C:\\Users\\sebas\\OneDrive\\Desktop\\HeLa_Images_TIFF\\HeLa UNET Files\\UNET_Weights_Group_" + group + "_Model_" + str(model_num) + "_Epoch_" + str(epoch_num) + ".weights.h5")
    else:
        U.load_weights("C:\\Users\\sebas\\OneDrive\\Desktop\\HeLa_Images_TIFF\\HeLa UNET Files\\UNET_Weights_Group_" + group + "_Model_" + str(model_num) + "_Epoch_" + str(epoch_num) + "_No_Improvement.weights.h5")

    # Load input image
    vol_in = iio.volread("C:\\Users\\sebas\\OneDrive\\Desktop\\HeLa_Images_TIFF\\Prepared_HeLa_Images\\HeLa_Images_" + group + "_Membrane_Mask_Standard_Size_Image" + str(im_num) + ".tiff")
    vol_in = np.array(vol_in)
    t_in = tf.convert_to_tensor(vol_in)
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
    vol_true = iio.volread("C:\\Users\\sebas\\OneDrive\\Desktop\\HeLa_Images_TIFF\\Prepared_HeLa_Images\\HeLa_Images_" + group + "_Nucleus_Mask_Standard_Size_Image" + str(im_num) + ".tiff")
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
    vol_in_255s = np.reshape(vol_in*255, newshape=[75,512,512,1])
    vol_true_255s = np.reshape(vol_true_255, newshape=[75,512,512,1])
    vol_ones = np.ones_like(vol_in_255s)*255
    vol_combined_RGB = np.concatenate((vol_ones - vol_in_255s,vol_ones - vol_in_255s + vol_true_255s,vol_ones - vol_in_255s + vol_true_255s), axis=3)
    #vol_combined_RGB = np.concatenate((vol_ones - vol_in_255s + vol_true_255s,vol_ones - vol_in_255s + vol_true_255s,vol_ones - vol_in_255s + vol_true_255s), axis=3)
    #vol_combined_RGB = np.concatenate((vol_true_255s,vol_true_255s,vol_true_255s), axis=3)

    # Combined Raw Prediction Plot
    vol_out_pred = np.reshape(vol_out_pred, newshape=[75,512,512,1])
    vol_out_pred_RGB = np.concatenate((np.zeros_like(vol_out_pred),vol_out_pred,np.zeros_like(vol_out_pred)), axis=3)
    vol_combined_pred_RGB = vol_combined_RGB - vol_out_pred_RGB
    vol_combined_pred_RGB = np.absolute(vol_combined_pred_RGB)
    vol_combined_pred_RGB = vol_combined_pred_RGB.astype(int)

    # Combined Binary Prediction Plot
    vol_combined_bin_RGB = vol_combined_RGB
    vol_out_bin_r = np.reshape(vol_out_bin, newshape=[75,512,512,1])
    vol_out_bin_RGB = np.concatenate((np.zeros_like(vol_out_bin_r),vol_out_bin_r,np.zeros_like(vol_out_bin_r)), axis=3)
    vol_combined_bin_RGB = vol_combined_bin_RGB - vol_out_bin_RGB
    vol_combined_bin_RGB = np.absolute(vol_combined_bin_RGB)
    vol_combined_bin_RGB = vol_combined_bin_RGB.astype(int)

    # Combined Shifted Binary Prediction Plot
    vol_combined_shift_RGB = vol_combined_RGB
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
    dim_z, _, _ = np.nonzero(np.array(vol_in))
    z_range = range(np.min(dim_z),np.max(dim_z)+1)

    plt.title("Dice Score: " + str(dice_loss_pred.numpy()))
    with writer.saving(fig, "C:\\Users\\sebas\\OneDrive\\Desktop\\HeLa_Images_TIFF\\" + group + "_Animations\\Image_" + str(im_num) + "_Model_" + str(model_num) + "_Combined_Prediction.mp4", 300):
        for z_frame in z_range:
            plt.imshow(vol_combined_pred_RGB[z_frame,:,:,:])
            writer.grab_frame()

    plt.title("Dice Score: " + str(dice_loss_bin.numpy()))
    with writer.saving(fig, "C:\\Users\\sebas\\OneDrive\\Desktop\\HeLa_Images_TIFF\\" + group + "_Animations\\Image_" + str(im_num) + "_Model_" + str(model_num) + "_Combined_Binary_Prediction.mp4", 300):
        for z_frame in z_range:
            plt.imshow(vol_combined_bin_RGB[z_frame,:,:,:])
            writer.grab_frame()

    plt.title("Dice Score: " + str(dice_loss_shift.numpy()))
    with writer.saving(fig, "C:\\Users\\sebas\\OneDrive\\Desktop\\HeLa_Images_TIFF\\" + group + "_Animations\\Image_" + str(im_num) + "_Model_" + str(model_num) + "_Combined_Shifted_Prediction.mp4", 300):
        for z_frame in z_range:
            plt.imshow(vol_combined_shift_RGB[z_frame,:,:,:])
            writer.grab_frame()
