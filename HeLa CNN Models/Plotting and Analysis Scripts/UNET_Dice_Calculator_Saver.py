# Importing packages

import tensorflow as tf
from keras import Model
from keras import layers
import imageio as iio
import numpy as np
from scipy.ndimage import median_filter

for group in ["CTL","Ble","H11","Y27"]:

    save_list = []

    for model_num in [1,2,3,4,5,6,7,8,9,10]:

        # Params
        no_imp = False
        epoch_num = 10
        cutoff = 0.55

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

        # Define Dice Loss
        def DiceLoss(target, gen_output):

            num = tf.reduce_sum(target*gen_output)
            den = tf.reduce_sum(target+gen_output)
            loss = 2*num/den

            return loss

        # Load Validation Set Sample Numbers
        valid_set = []
        s_file = open("C:\\Users\\sebas\\OneDrive\\Desktop\\HeLa_Images_TIFF\\HeLa UNET Files\\Sets_Group_" + group + "_Model_" + str(model_num) + ".txt")
        s_text = s_file.read()

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

        # Create the Single Kernel Area Calculator
        kernel = tf.constant([
            [[0, 0, 0], [0, 0.04293184, 0], [0, 0, 0]],
            [[0, 0.084952, 0], [0.084952, 0, 0.084952], [0, 0.084952, 0]],
            [[0, 0, 0], [0, 0.04293184, 0], [0, 0, 0]] 
        ], dtype=tf.float32)

        class MyInitializer(tf.keras.initializers.Initializer):
            def __init__(self, kernel):
                self.kernel = kernel
            def __call__(self, shape, dtype=None):
                return tf.cast(tf.reshape(self.kernel, shape=shape), dtype=dtype)

        inputs = tf.keras.layers.Input(shape=[75,512,512,1])
        conv_layer = tf.keras.layers.Conv3D(filters=1,kernel_size=3,strides=1,padding='same',use_bias=False,data_format='channels_last',kernel_initializer=MyInitializer(kernel))(inputs)

        model = Model(inputs=inputs, outputs=conv_layer, name='Conv_Layer')

        # Define the H, A, V Calculator Function
        def HAV_HeLa(vol_out_bin, t_out_bin):

            dim_z, _, _ = np.nonzero(vol_out_bin)
            if np.size(dim_z) == 0:
                h_bin_samp = 0
                a_bin_samp = 0
                v_bin_samp = 0
            else:

                h_bin_samp = 0.41*(np.max(dim_z) - np.min(dim_z))
                
                a_bin_av = model.predict(tf.reshape(t_out_bin, shape=[1,75,512,512,1]))
                a_bin_av = np.reshape(a_bin_av, newshape=[75,512,512])
                a_bin_im = np.where(vol_out_bin == 0, a_bin_av, 0)
                a_bin_samp = np.sum(a_bin_im)

                v_bin_samp = 0.41*(0.2072**2)*np.shape(vol_out_bin[np.nonzero(vol_out_bin)])[0]

            return h_bin_samp, a_bin_samp, v_bin_samp

        # Calculate the Dice Loss over the Validation Set
        dice_scores_pred = []
        dice_scores_bin = []
        dice_scores_shift = []
        h_bin = []
        a_bin = []
        v_bin = []
        tilt_bin = []
        h_true = []
        a_true = []
        v_true = []
        tilt_true = []

        for k in range(len(valid_set)):

            # Load input image
            img_num = str(valid_set[k])
            vol_in = iio.volread("C:\\Users\\sebas\\OneDrive\\Desktop\\HeLa_Images_TIFF\\Prepared_HeLa_Images\\HeLa_Images_" + group + "_Membrane_Mask_Standard_Size_Image" + img_num + ".tiff")
            t_in = tf.convert_to_tensor(vol_in)
            t_in = tf.cast(t_in, dtype=tf.float32)
            t_in = tf.reshape(t_in, shape=[1,75,512,512,1])

            # Predict Nuclear Shape
            t_out = U(t_in, training=False)
            t_out = tf.reshape(t_out, shape=[75,512,512])
            vol_out = t_out.numpy()
            vol_out_bin = np.where(vol_out > cutoff, 255, 0)
            vol_out_bin = median_filter(vol_out_bin, size=3)

            # Load True Nuclear Shape
            vol_true = iio.volread("C:\\Users\\sebas\\OneDrive\\Desktop\\HeLa_Images_TIFF\\Prepared_HeLa_Images\\HeLa_Images_" + group + "_Nucleus_Mask_Standard_Size_Image" + img_num + ".tiff")
            vol_true = np.array(vol_true)
            vol_true = median_filter(vol_true, size=3)

            # Calculate Dice Score of Predicted Nucleus
            t_true = tf.convert_to_tensor(vol_true)
            t_true = tf.cast(t_true, dtype=tf.float32)
            dice_loss_pred = DiceLoss(t_true,t_out)

            # Calculate Dice Score of Binary Nucleus
            t_out_bin = tf.convert_to_tensor(vol_out_bin/255)
            t_out_bin = tf.cast(t_out_bin, dtype=tf.float32)
            dice_loss_bin = DiceLoss(t_true,t_out_bin)

            # Calculate Dice Score of Shifted Nucleus
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
                    vol_out_shift[int(zyx[0][m]+dev[0]),int(zyx[1][m]+dev[1]),int(zyx[2][m]+dev[2])] = 1
                t_out_shift = tf.convert_to_tensor(vol_out_shift)
                t_out_shift = tf.cast(t_out_shift, dtype=tf.float32)
                dice_loss_shift = DiceLoss(t_true,t_out_shift)

            dice_scores_pred.append(dice_loss_pred.numpy())
            dice_scores_bin.append(dice_loss_bin.numpy())
            dice_scores_shift.append(dice_loss_shift.numpy())

            # Calculate the Predicted (Binary) Nuclear Height, Area, and Volume
            h_bin_samp, a_bin_samp, v_bin_samp = HAV_HeLa(vol_out_bin, t_out_bin)
            h_bin.append(h_bin_samp)
            a_bin.append(a_bin_samp)
            v_bin.append(v_bin_samp)

            # Calculate the Actual Nuclear Height, Area, and Volume
            h_true_samp, a_true_samp, v_true_samp = HAV_HeLa(np.array(vol_true), t_true)
            h_true.append(h_true_samp)
            a_true.append(a_true_samp)
            v_true.append(v_true_samp)

            # Calculate the Predicted (Binary) Nuclear Tilt
            z, y, x = np.nonzero(vol_out_bin)
            if np.size(z) == 0:
                tilt_bin_samp = 1
                tilt_bin.append(tilt_bin_samp)
            else:
                max_z = np.max(z)
                min_z = np.min(z)
                max_y_list = []
                min_y_list = []
                max_x_list = []
                min_x_list = []
                for m in range(np.size(z)):
                    if z[m] == max_z:
                        max_y_list.append(y[m])
                        max_x_list.append(x[m])
                    elif z[m] == min_z:
                        min_y_list.append(y[m])
                        min_x_list.append(x[m])
                max_yx = [np.mean(max_y_list), np.mean(max_x_list)]
                min_yx = [np.mean(min_y_list), np.mean(min_x_list)]
                tilt_bin_samp = np.sqrt(((0.2072*(max_yx[0] - min_yx[0]))**2) + ((0.2072*(max_yx[1] - min_yx[1]))**2) + ((0.41*(max_z - min_z))**2))/h_bin_samp
                tilt_bin.append(tilt_bin_samp)

            # Calculate the True Nuclear Tilt
            z, y, x = np.nonzero(vol_true)
            max_z = np.max(z)
            min_z = np.min(z)
            max_y_list = []
            min_y_list = []
            max_x_list = []
            min_x_list = []
            for m in range(np.size(z)):
                if z[m] == max_z:
                    max_y_list.append(y[m])
                    max_x_list.append(x[m])
                elif z[m] == min_z:
                    min_y_list.append(y[m])
                    min_x_list.append(x[m])
            max_yx = [np.mean(max_y_list), np.mean(max_x_list)]
            min_yx = [np.mean(min_y_list), np.mean(min_x_list)]
            tilt_true_samp = np.sqrt(((0.2072*(max_yx[0] - min_yx[0]))**2) + ((0.2072*(max_yx[1] - min_yx[1]))**2) + ((0.41*(max_z - min_z))**2))/h_true_samp
            tilt_true.append(tilt_true_samp)

            # Print Confirmation Message
            print("--------------------------------------------------------------------------")
            print("Group: ",group,", Model:",model_num)
            print("Completed sample number:",k+1,"out of",len(valid_set))
            print("Dice score of prediction:",dice_loss_pred.numpy())
            print("Dice score of binary:",dice_loss_bin.numpy())
            print("Dice score of shifted:",dice_loss_shift.numpy())
            print("Height of prediction:",h_bin_samp)
            print("Height of actual:",h_true_samp)
            print("Area of prediction:",a_bin_samp)
            print("Area of actual:",a_true_samp)
            print("Volume of prediction:",v_bin_samp)
            print("Volume of actual:",v_true_samp)
            print("Tilt of prediction:",tilt_bin_samp)
            print("Tilt of actual:",tilt_true_samp)
            print("Center deviation in z:",dev[0])
            print("Center deviation in y:",dev[1])
            print("Center deviation in x:",dev[2])
            print("--------------------------------------------------------------------------")
            print("\n")

            # Save the results to the list
            save_list.append([img_num, dice_loss_pred.numpy(), dice_loss_bin.numpy(), dice_loss_shift.numpy(), h_bin_samp, a_bin_samp, v_bin_samp, h_true_samp, a_true_samp, v_true_samp, tilt_bin_samp, tilt_true_samp, dev[0], dev[1], dev[2]])

    # Save the results
    save_array = np.array(save_list, dtype=float)
    save_path = "C:\\Users\\sebas\\OneDrive\\Desktop\\HeLa_Images_TIFF\\HeLa_Results_Cutoff_" + str(cutoff) + "_Group_" + group + ".csv"
    np.savetxt(save_path, save_array, delimiter=",")
