# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:15:12 2022

@author: Zeshan Khan
"""

from reflection_removal import compute_telea_ns

import numpy as np
import pandas as pd
import os
import cv2
import numpy as np
#import specularity as spc
#base_path='/path to kvasirv1/'



trainXp=base_path+"dev/dev/"
testXp=base_path+"val/val/"

teleap = '/output path for telea/'
nsp = '/output path for ns/'

compute_telea_ns(trainXp,teleap,nsp,istelea=False,isns=True)

#compute_telea_ns(testXp,teleap,nsp,istelea=True,isns=True)