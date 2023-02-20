# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:44:36 2022

@author: ZESHAN KAHN
"""

import numpy as np
import os
import cv2, numpy as np, os, h5py, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

data_path_test_2017="/kaggle/input/kvasirv1/val/val/"
data_path_train_2017="/kaggle/input/kvasirv1/dev/dev/"
data_path_test_2018="/kaggle/input/kvasirv2/val/val/"
data_path_train_2018="/kaggle/input/kvasirv2/dev/dev/"
data_path_test_2020="/kaggle/input/hyperkvasir/val/val/"
data_path_train_2020="/kaggle/input/hyperkvasir/dev/dev/"

data_path_dow="/kaggle/input/dowtest/dow/"


batch_size = 25
nb_epoch = 500
learning_rate=0.01
#optimiser="adagard"
optimiser="sgd"


label_map36=['barretts', 'barretts-short-segment', 'bbps-0-1', 'bbps-2-3','cecum', 'normal-cecum', 'dyed-lifted-polyps', 'dyed-resection-margins',
         'esophagitis','esophagitis-a','esophagitis-b-d','hemorrhoids', 'ileum', 'impacted-stool','normal-z-line','polyps','pylorus','normal-pylorus',
         'retroflex-rectum','retroflex-stomach','ulcerative-colitis','ulcerative-colitis-0-1','ulcerative-colitis-1-2','ulcerative-colitis-2-3',
         'ulcerative-colitis-grade-1','ulcerative-colitis-grade-2','ulcerative-colitis-grade-3',
         'lesion', 'dysplasia', 'cancer', 'blurry-nothing', 'colon-clear', 'stool-inclusions', 'stool-plenty', 'instruments', 'out-of-patient']

label_map23=['barretts', 'barretts-short-segment', 'bbps-0-1', 'bbps-2-3', 'cecum', 'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis-a', 'esophagitis-b-d',
             'hemorrhoids', 'ileum', 'impacted-stool', 'normal-z-line', 'polyps', 'pylorus', 'retroflex-rectum', 'retroflex-stomach',
             'ulcerative-colitis-0-1', 'ulcerative-colitis-1-2', 'ulcerative-colitis-2-3', 'ulcerative-colitis-grade-1', 'ulcerative-colitis-grade-2', 'ulcerative-colitis-grade-3'] 
label_map16=['retroflex-rectum', 'out-of-patient', 'ulcerative-colitis', 'normal-cecum', 'normal-z-line', 'dyed-lifted-polyps', 'blurry-nothing', 'retroflex-stomach', 'instruments', 'dyed-resection-margins', 'stool-plenty', 'esophagitis', 'normal-pylorus', 'polyps', 'stool-inclusions', 'colon-clear']
label_map8=['ulcerative-colitis', 'normal-cecum', 'normal-z-line', 'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-pylorus', 'polyps']
from keras.layers import Input, merge, ZeroPadding2D, Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import Sequential, backend as K, optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adagrad
from keras.applications.densenet import DenseNet169
from keras.applications.resnet import ResNet152, ResNet50
from keras.applications.vgg19 import VGG19
from tensorflow.keras.applications import NASNetLarge, MobileNetV2, InceptionV3
import cv2, numpy as np, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')
base_path="/kaggle/input/kvasirv2/"
