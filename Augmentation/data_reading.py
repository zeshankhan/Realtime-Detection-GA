# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:27:33 2022

@author: Zeshan Khan
"""

def gather_paths_all(jpg_path,num_classes=16,label_map=None):
    i=0  
    folder=os.listdir(jpg_path)
    count=0
    if (os.path.isfile(jpg_path+folder[0])):
        count=len(os.listdir(jpg_path))
    else:
        count=sum([len(os.listdir(jpg_path+f)) for f in os.listdir(jpg_path)])
    ima=['' for x in range(count)]
    labels=np.zeros((count,num_classes),dtype=float)
    label=[0 for x in range(count)]
    if (os.path.isfile(jpg_path+folder[0])):
        for f in folder:
            im=jpg_path+f
            ima[i]=im
            label[i]=0
            i+=1
            if(count<i):
                break
    else:
        for fldr in folder:
            for f in os.listdir(jpg_path+fldr+"/"):
                im=jpg_path+fldr+"/"+f
                ima[i]=im
                label[i]=label_map.index(fldr)+1
                i+=1
            if(count<=i):
                break
    for i in range(count):
        labels[i][label[i]-1]=1
    return ima,label,labels