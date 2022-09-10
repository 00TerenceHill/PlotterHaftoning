# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 19:30:48 2022

@author: Terence
"""

import numpy as np
import cv2
from Functions import imadjust,ColorQunatization,ColorQunatizationRGB,ColorQunatizationCMYK,readimageLAB,readimageCMYK,linewidth_from_data_units, readimage,SaveImagec,readimageRGB, readimageHSV, ImageByLineChirp
import matplotlib.pyplot as plt
Folder='Images/'
Name='ML-01-01'
Name='m1-01'
Ending='.jpg'

Filename=Folder+Name+Ending


ResizedPX=1024*5

Im = cv2.imread(Filename)
Im = cv2.cvtColor(Im, cv2.COLOR_BGR2RGB)
    
# Im=(Im-np.min(Im))/(np.max(Im)-np.min(Im))*256
width = int( ResizedPX)
height = int(Im.shape[0]/Im.shape[1]*ResizedPX)
dim = (width, height)
print('Start Resize')
img = cv2.resize(Im,dim , interpolation=cv2.INTER_CUBIC )

mask=(img[:,:,1]<1) & (img[:,:,2]<1) & (img[:,:,0]<1)
# mask = np.all(img == 0, axis=2).astype(np.uint8)


mask = mask.astype(np.uint8)
# mask[:,0]=np.min(mask)
# mask[0,:]=np.min(mask)
# mask[:,-1]=np.min(mask)
# mask[-1,:]=np.min(mask)

kernel = np.ones((25,25),np.uint8)
mask = cv2.dilate(mask,kernel,iterations = 1)
# mask = cv2.imread('mask2.png',0)



# dst =( cv2.inpaint(img,mask,3,cv2.INPAINT_NS)*.7+cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)*.3)
dst=cv2.inpaint(img,mask,10,cv2.INPAINT_NS)
kernel = np.ones((25,25),np.float32)/625
dst2 = cv2.filter2D(dst,-1,kernel)


mask=img<1
dst[mask]=dst2[mask]
plt.imshow(mask[:,:,1])
plt.show()

plt.imshow(dst/256)
plt.axis('off')

plt.savefig('./ResultsJPG/' + Name +   '.jpg', dpi=1500, bbox_inches='tight', pad_inches=0)
plt.show()
