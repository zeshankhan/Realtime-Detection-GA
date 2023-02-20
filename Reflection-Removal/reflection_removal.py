# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:18:20 2022

@author: Zeshan Khan
"""

from reflection_detection import compute_Omega

def compute_telea_ns(trainXp,teleap,nsp,istelea=True,isns=True):
    if(not os.path.isdir(teleap)):
        os.mkdir(teleap)
    if(not os.path.isdir(nsp)):
        os.mkdir(nsp)
    imdirs=[trainXp+f+"/" for f in os.listdir(trainXp)]
    impaths=[imdir+f for imdir in imdirs for f in os.listdir(imdir)][:100]
    for impath in impaths:
        I = cv2.imread(impath)
        #gray_img=cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
        gray_img = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        omega=compute_Omega(orignal=I)
        radius = 12
        #I = cv2.imread(impath)
        #print(impath.split("\\"))
        img=impath.split("/")[-1]
        cls1=impath.split("/")[-2]
        if(istelea):
            telea = cv2.inpaint(I, omega, radius, cv2.INPAINT_TELEA)
            if(not os.path.isdir(teleap+cls1)):
                os.mkdir(teleap+cls1)
            cv2.imwrite(teleap+cls1+"/"+img,telea)
        if(isns):
            ns = cv2.inpaint(I, omega, radius, cv2.INPAINT_NS)
            if(not os.path.isdir(nsp+cls1)):
                os.mkdir(nsp+cls1)
            cv2.imwrite(nsp+cls1+"/"+img,ns)
        #break
    return
