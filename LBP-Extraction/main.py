# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:32:35 2022

@author: ZESHAN KAHN
"""
from lbp_extraction import get_lbps


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
data_path_base='/kaggle/input/'
features_path_base='/kaggle/working/'
data_name="me2018"

train="dev/dev"

import cv2, numpy as np, os, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
import h5py
from skimage import feature

img_cols=img_rows=224
img_channels=3
         
label_map=['retroflex-rectum', 'out-of-patient', 'ulcerative-colitis', 'normal-cecum', 'normal-z-line', 'dyed-lifted-polyps', 'blurry-nothing', 'retroflex-stomach', 'instruments', 'dyed-resection-margins', 'stool-plenty', 'esophagitis', 'normal-pylorus', 'polyps', 'stool-inclusions', 'colon-clear']



dataset="kvasirv2"
train_count=5293
test_count=8740
num_classes=16

data_path=data_path_base+dataset+"/"
data_path_train=data_path+"dev/dev/"
data_path_test=data_path+"val/val/"
lbp_train=features_path_base+data_name+"_lbp_train"
lbp_test=features_path_base+data_name+"_lbp_test"

get_lbps(data_path=data_path_train,storage_path=lbp_train,radii=[1,2,3,4,5],num_classes=num_classes,batch_size=2000)
get_lbps(data_path=data_path_test,storage_path=lbp_test,radii=[1,2,3,4,5],num_classes=num_classes,batch_size=2000)