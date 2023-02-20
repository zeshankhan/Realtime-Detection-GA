# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:32:24 2022

@author: Zeshan Khan
"""

def set_all_paths():
    data_path_base='/kaggle/input/'
    features_path_base='/kaggle/working/'
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
    data_path_aug=features_path_base+"augmented/"
    if(not os.path.exists(data_path_aug)):
        os.mkdir(data_path_aug)
    config=[('img_cols',img_cols),('img_rows',img_rows),('img_channels',img_channels),('dataset',dataset),
            ('train_count',train_count),('test_count',test_count),('num_classes',num_classes),
            ('data_path_base',data_path_base),('features_path_base',features_path_base),('data_path',data_path),
            ('data_path_train',data_path_train),('data_path_test',data_path_test),('data_path_aug',data_path_aug)]
    return label_map,config
