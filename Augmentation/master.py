# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:26:25 2022

@author: Zeshan Khan
"""



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, io

import random
from wand.image import Image as wimage
from PIL import Image as pimage
from PIL import ImageOps as pimageOps
from PIL import ImageEnhance
import cv2


import cv2, numpy as np, os, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
import h5py
from skimage import feature

import cv2, numpy as np, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

from data_reading import gather_paths_all

paths,_,_=gather_paths_all(data_path_train,num_classes=16)

clas=list(set([p.split('/')[-2] for p in paths]))
count=np.zeros(len(clas),np.int16)
for i in range(len(clas)):
    pathsc=[p for p in paths if p.split('/')[-2]==clas[i]]
    count[i]=len(pathsc)
upto=int(1*max(count))
print(upto)

from pprint import pprint
from zipfile import ZipFile


for i in range(len(clas)):
    pathsc=[p for p in paths if p.split('/')[-2]==clas[i]]
    images=augment(pathsx=pathsc,outputx=data_path_aug,upto=upto)
    
    images = file_process_in_memory(images)
    zip_file_bytes_io = io.BytesIO()
    with ZipFile("/kaggle/working/"+clas[i]+".zip", 'w') as zip_file:
        for image_name, bytes_stream in images:
            zip_file.writestr(image_name.split('/')[-1], bytes_stream.getvalue())
        pprint(zip_file.infolist())  # Print final contents of in memory zip file.



