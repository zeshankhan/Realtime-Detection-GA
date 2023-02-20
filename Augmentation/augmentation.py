# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:30:01 2022

@author: Zeshan Khan
"""

def augment_noise(imagex=None):
    noise_types=['gaussian','laplacian','multiplicative_gaussian','uniform','poisson','impulse']#random=useless
    nt=noise_types[random.randint(0,5)]
    atn=random.randint(5,10)
    imx1=wimage.from_array(imagex)
    imx1.noise(noise_type=nt, attenuate = atn*0.1)
    img_buffer = np.asarray(bytearray(imx1.make_blob(format='png')), dtype='uint8')
    bytesio = io.BytesIO(img_buffer)
    imx1 = pimage.open(bytesio)
    return imx1
def augment_crop(imx=None):
    wo,ho=imx.size
    wn=random.randint(330,wo)
    hn=random.randint(330,ho)
    left = (wn-wo)/2
    top = (hn-ho)/2
    right = wn+left
    bottom = hn+top
    
    imxn = imx.crop((left, top, right, bottom))
    imyn = imy.crop((left, top, right, bottom))
    
    return imxn,imyn

def augment_fliph(imx=None):
    imxn = imx.transpose(pimage.FLIP_LEFT_RIGHT)
    return imxn

def augment_flipv(imx=None):
    imxn = imx.transpose(pimage.FLIP_TOP_BOTTOM)
    return imxn

def augment_rotate(imx=None):
    rot=random.randint(0,360)
    imxn = imx.rotate(rot)
    return imxn

def augment_mirror(imx=None):
    imxn=pimageOps.mirror(imx)
    return imxn


def augment_scale(imx=None):
    wo,ho=imx.size
    wo=max(wo,224)
    ho=max(ho,224)
    wn=random.randint(224,wo)
    hn=random.randint(224,ho)
    sn = (wn,hn)
    if(imx.mode!="RGB"):
        print('gray',imx.mode)
        imx=imx.convert('RGB')
    imxn=imx.resize(sn)
    return imxn


def augment_bright(imx=None):
    factor=random.randint(5,15)
    imxn = ImageEnhance.Brightness(imx).enhance(factor*0.1)
    return imxn


def augment_contrast(imx=None):
    factor=random.randint(5,15)
    imxn = ImageEnhance.Contrast(imx).enhance(factor*0.1)
    return imxn

def augment_sharp(imx=None):
    factor=random.randint(5,15)
    imxn = ImageEnhance.Sharpness(imx).enhance(factor*0.1)
    return imxn

def augment(pathsx=None,outputx="",upto=0):
    images = []
    for i in pathsx:
        images.append([i,pimage.open(i)])
    curr=len(pathsx)
    dr=pathsx[0].split("/")[-2]
    if(not os.path.exists(outputx+dr+"/")):
        os.mkdir(outputx+dr+"/")
    print(dr,"\tAvailable\t",curr,"\tRemaining\t",upto-curr)
    for i in range(curr,upto):
        x=random.choice(pathsx)
        x1=pimage.open(x)
        for j in range(8):
            rand=random.randint(0,12)
            if(rand==1):
                x1=augment_noise(x1)
            if(rand==10):
                x1=augment_noise(x1)
            if(rand==11):
                x1=augment_noise(x1)
            if(rand==12):
                x1=augment_noise(x1)
            elif(rand==2):
                x1=augment_fliph(x1)
            elif(rand==3):
                x1=augment_flipv(x1)
            elif(rand==4):
                x1=augment_mirror(x1)
            elif(rand==5):
                x1=augment_rotate(x1)
            elif(rand==6):
                x1=augment_scale(x1)
            elif(rand==7):
                x1=augment_bright(x1)
            elif(rand==8):
                x1=augment_contrast(x1)
            elif(rand==9):
                x1=augment_sharp(x1)
            else:
                break
        if(x1.mode!="RGB"):
            print('gray',x1.mode)
            x1=x1.convert('RGB')
        x1=x1.resize((224,224))
        images.append([outputx+dr+"/"+x.split("/")[-1].split(".")[0]+"_"+str(i)+".jpg",x1])
        #x1.save(outputx+dr+"/"+x.split("/")[-1].split(".")[0]+"_"+str(i)+".jpg")
    return images
