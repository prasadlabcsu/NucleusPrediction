# Importing packages

import tensorflow as tf
from keras import Model
from keras import layers
import imageio as iio
import numpy as np
from scipy import stats
from scipy.ndimage import median_filter

# Params
no_imp = False
model_num = 1
epoch_num = 20
cutoff_range = np.array(range(25,46))/100
losses = []

# Best cutoff = 0.42 !!!

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

# Define Dice Loss
def DiceLoss(target, gen_output):

    num = tf.reduce_sum(target*gen_output)
    den = tf.reduce_sum(target+gen_output)
    loss = 2*num/den

    return loss

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

# Create the Single Kernel Area Calculator
kernel = tf.constant([
    [[0, 0, 0], [0, 0.011736038889, 0], [0, 0, 0]],
    [[0, 0.03141657, 0], [0.03141657, 0, 0.03141657], [0, 0.03141657, 0]],
    [[0, 0, 0], [0, 0.011736038889, 0], [0, 0, 0]]
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
def HAV(vol_out, vol_out_bin, t_out_bin):

    dim_z, _, _ = np.nonzero(vol_out_bin)
    h_bin_samp = 0.29*(np.max(dim_z) - np.min(dim_z))
    
    a_bin_av = model.predict(tf.reshape(t_out_bin, shape=[1,75,512,512,1]))
    a_bin_av = np.reshape(a_bin_av, newshape=[75,512,512])
    a_bin_im = np.where(vol_out_bin == 0, a_bin_av, 0)
    a_bin_samp = np.sum(a_bin_im)

    v_bin_samp = 0.29*(0.108333**2)*np.shape(vol_out_bin[np.nonzero(vol_out_bin)])[0]
    v_pred_samp = 0.29*(0.108333**2)*np.sum(vol_out)

    return h_bin_samp, a_bin_samp, v_bin_samp, v_pred_samp

# Calculate the Dice Loss over the Validation Set
dice_scores_pred = []
dice_scores_bin = []
h_bin = []
a_bin = []
v_bin = []
v_pred = []
h_true = []
a_true = []
v_true = []

for cutoff in cutoff_range:
    for k in range(20):

        # Load input image
        im_num = valid_set[k]
        vol_in = iio.volread("C:\\Users\\sebas\\OneDrive\\Desktop\\Actin_Images_TIFF\\Actin_Images_Interphase_512_Membrane_Mask_Standard_Size_Image" + str(im_num) + ".tiff")
        t_in = tf.convert_to_tensor(vol_in)
        t_in = t_in/255
        t_in = tf.cast(t_in, dtype=tf.float32)
        t_in = tf.reshape(t_in, shape=[1,75,512,512,1])

        # Predict Nuclear Shape
        t_out = U(t_in, training=False)
        t_out = tf.reshape(t_out, shape=[75,512,512])
        vol_out = t_out.numpy()
        vol_out_bin = np.where(vol_out > cutoff, 255, 0)
        vol_out_bin = median_filter(vol_out_bin, size=3)

        # Load True Nuclear Shape
        vol_true = iio.volread("C:\\Users\\sebas\\OneDrive\\Desktop\\Actin_Images_TIFF\\Actin_Images_Interphase_512_Nucleus_Standard_Size_Image" + str(im_num) + ".tiff")

        # Calculate Dice Score of Predicted Nucleus
        t_true = tf.convert_to_tensor(vol_true)
        t_true = tf.cast(t_true, dtype=tf.float32)
        dice_loss_pred = DiceLoss(t_true,t_out)

        # Calculate Dice Score of Predicted Binary Nucleus
        t_out_bin = tf.convert_to_tensor(vol_out_bin/255)
        t_out_bin = tf.cast(t_out_bin, dtype=tf.float32)
        dice_loss_bin = DiceLoss(t_true,t_out_bin)

        dice_scores_pred.append(dice_loss_pred.numpy())
        dice_scores_bin.append(dice_loss_bin.numpy())

        # Calculate the Predicted (Binary) Nuclear Height, Area, and Volume
        h_bin_samp, a_bin_samp, v_bin_samp, v_pred_samp = HAV(vol_out, vol_out_bin, t_out_bin)
        h_bin.append(h_bin_samp)
        a_bin.append(a_bin_samp)
        v_bin.append(v_bin_samp)
        v_pred.append(v_pred_samp)

        # Calculate the Actual Nuclear Height, Area, and Volume
        h_true_samp, a_true_samp, v_true_samp, _ = HAV(vol_out, np.array(vol_true), t_true)
        h_true.append(h_true_samp)
        a_true.append(a_true_samp)
        v_true.append(v_true_samp)

        # Print Confirmation Message
        print("--------------------------------------------------------------------------")
        print("Cutoff =",cutoff)
        print("Completed sample number:",k+1,"out of",len(valid_set))
        print("Dice score of prediction:",dice_loss_pred.numpy())
        print("Dice score of binary:",dice_loss_bin.numpy())
        print("Height of prediction:",h_bin_samp)
        print("Height of actual:",h_true_samp)
        print("Area of prediction:",a_bin_samp)
        print("Area of actual:",a_true_samp)
        print("Volume of prediction:",v_bin_samp)
        print("Volume of prediction (no threshold):",v_pred_samp)
        print("Volume of actual:",v_true_samp)
        print("--------------------------------------------------------------------------")
        print("\n")

    # Mean Dice Scores
    mean_dice_pred = np.mean(dice_scores_pred)
    mean_dice_bin = np.mean(dice_scores_bin)
    mean_h_bin = np.mean(h_bin)
    mean_a_bin = np.mean(a_bin)
    mean_v_bin = np.mean(v_bin)
    mean_v_pred = np.mean(v_pred)
    mean_h_true = np.mean(h_true)
    mean_a_true = np.mean(a_true)
    mean_v_true = np.mean(v_true)
    mad_h = np.mean(np.abs(np.array(h_bin) - np.array(h_true)))
    mad_a = np.mean(np.abs(np.array(a_bin) - np.array(a_true)))
    mad_v = np.mean(np.abs(np.array(v_bin) - np.array(v_true)))
    mad_v_pred = np.mean(np.abs(np.array(v_pred) - np.array(v_true)))
    rmse_h = np.sqrt(np.mean((np.array(h_bin) - np.array(h_true))**2))
    rmse_a = np.sqrt(np.mean((np.array(a_bin) - np.array(a_true))**2))
    rmse_v = np.sqrt(np.mean((np.array(v_bin) - np.array(v_true))**2))
    rmse_v_pred = np.sqrt(np.mean((np.array(v_pred) - np.array(v_true))**2))

    # t Test
    t_test_result_dice = stats.ttest_ind(dice_scores_pred, dice_scores_bin, equal_var=False, alternative='less')
    t_test_result_h = stats.ttest_ind(h_bin, h_true, equal_var=False)
    t_test_result_a = stats.ttest_ind(a_bin, a_true, equal_var=False)
    t_test_result_v = stats.ttest_ind(v_bin, v_true, equal_var=False)
    t_test_result_v_pred = stats.ttest_ind(v_pred, v_true, equal_var=False)

    # Print results
    print("Mean Dice Score from Prediction:" + str(mean_dice_pred))
    print("Mean Dice Score from Binary:" + str(mean_dice_bin))
    print("T Test Result:" + str(t_test_result_dice))
    print("Mean Height from Binary:" + str(mean_h_bin))
    print("Mean Height from Actual:" + str(mean_h_true))
    print("MAD of Height:" + str(mad_h))
    print("RMSE of Height:" + str(rmse_h))
    print("T Test Result:" + str(t_test_result_h))
    print("Mean Area from Binary:" + str(mean_a_bin))
    print("Mean Area from Actual:" + str(mean_a_true))
    print("MAD of Area:" + str(mad_a))
    print("RMSE of Area:" + str(rmse_a))
    print("T Test Result:" + str(t_test_result_a))
    print("Mean Volume from Binary:" + str(mean_v_bin))
    print("Mean Volume from Prediction:" + str(mean_v_pred))
    print("Mean Volume from Actual:" + str(mean_v_true))
    print("MAD of Volume from Binary:" + str(mad_v))
    print("MAD of Volume from Prediction:" + str(mad_v_pred))
    print("RMSE of Volume from Binary:" + str(rmse_v))
    print("RMSE of Volume from Prediction:" + str(rmse_v_pred))
    print("T Test Result from Binary:" + str(t_test_result_v))
    print("T Test Result from Prediction:" + str(t_test_result_v_pred))

    # Calculate the cutoff loss
    h_loss = mad_h/0.4914
    a_loss = mad_a/41.2913
    v_loss = mad_v/52.8601
    dice_loss = (1 - mean_dice_bin)/(1 - 0.809173)

    loss = h_loss + a_loss + v_loss + dice_loss
    losses.append([cutoff, loss, mad_h, mad_a, mad_v, mean_dice_bin])
    print("Cutoff:",cutoff)
    print("Losses:",losses[-1])
