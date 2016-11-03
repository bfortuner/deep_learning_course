from __future__ import division,print_function
import os, json
from glob import glob
import numpy as np
from matplotlib import pyplot as plt

from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
import bcolz
import itertools

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder

from vgg16 import Vgg16
np.set_printoptions(precision=4, linewidth=100)


def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        if titles is not None:
            sp.set_title(titles[i], fontsize=18)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4):
    return gen.flow_from_directory(dirname, target_size=(224,224),
            class_mode='categorical', shuffle=shuffle, batch_size=batch_size)


def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())


def get_data(batches):
    return np.concatenate([batches.next()[0] for i in range(batches.nb_sample // batches.batch_size)])


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def save_array(fname, arr):
    bcolz.carray(arr, rootdir=fname, mode='w')


def load_array(fname):
    return bcolz.open(fname)[:]


def vgg_cats():
    vgg = Vgg16()
    model = vgg.model
    model.pop()
    model.add(Dense(2, activation='softmax'))
    return model


def get_classes(path):
    batches = get_batches(path+'train', shuffle=False, batch_size=1)
    val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
    return val_batches.classes, batches.classes, onehot(val_batches.classes), onehot(batches.classes)


def split_at(model, layer_type):
    layers = model.layers
    layer_idx = [index for index,layer in enumerate(layers) if type(layer) is layer_type][0]
    return layers[:layer_idx+1], layers[layer_idx+1:]

