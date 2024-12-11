# This module defines the 3 augmentation operations to possibly apply to the image data, including a
# 3D flip or inversion and 3D rotation. 

# Importing packages

import numpy as np
from scipy.ndimage.interpolation import affine_transform

def flip3D(X, y): # Function for flipping the image along a specified axis in 3D
    """
    Flip the 3D image with respect one of the 3 axes chosen randomly
    """
    choice = np.random.randint(3)
    if choice == 0: # flip on z
        X_flip, y_flip = X[::-1, :, :], y[::-1, :, :]
    if choice == 1: # flip on y
        X_flip, y_flip = X[:, ::-1, :], y[:, ::-1, :]
    if choice == 2: # flip on x
        X_flip, y_flip = X[:, :, ::-1], y[:, :, ::-1]
        
    return X_flip, y_flip

def rotation3D(X, y): # Function for rotating an image up to 30 degrees about any of the 3D axes
    """
    Rotate a 3D image with alfa, beta and gamma degree respect the axis x, y and z respectively.
    The three angles are chosen randomly between 0-30 degrees
    """
    alpha, beta, gamma = np.pi*np.random.random_sample(3,)/2
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    
    R = np.dot(np.dot(Rx, Ry), Rz)
    R_inv = np.linalg.inv(R)
    
    X_rot = affine_transform(X, R_inv, offset=0, order=5, mode='constant')
    y_rot = affine_transform(y, R_inv, offset=0, order=5, mode='constant')
    
    return X_rot, y_rot

def random_decisions(N): # Function which randomly generates a table deciding which augmentations to apply
    """
    Generate N random decisions for augmentation
    N should be equal to the batch size
    """
    
    decisions = np.zeros((N, 2)) # 2 is the number of augmentation techniques to combine (patch extraction and elastic deformation excluded)
    for n in range(N):
        decisions[n] = np.random.randint(2, size=2)
        
    return decisions

def combine_aug(X, y, do): # Function which combines all the augmentations chosen by the random decisions
    """
    Combine randomly the different augmentation techniques written above
    """
    Xnew, ynew = X, y
    
    if np.random.random_sample()>0.75: # Make sure to use at least 25% of the original images
        return Xnew, ynew
    else:   
        if do[0] == 1:
            Xnew, ynew = flip3D(Xnew, ynew)

        if do[1] == 1:
            Xnew, ynew = rotation3D(Xnew, ynew)

        return Xnew, ynew

def aug_batch(Xb, Yb): # Function which uses the above functions to augment a batch of image data
    """
    Generate an augmented image batch 
    """

    batch_size = len(Xb)
    newXb, newYb = np.empty_like(Xb), np.empty_like(Yb)
    decisions = random_decisions(batch_size)
    
    for i in range(batch_size):
        newXb[i], newYb[i] = combine_aug(Xb[i], Yb[i], decisions[i])
        
    return newXb, newYb 
