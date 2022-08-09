# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 11:01:11 2022

@author: schmi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 13:50:28 2022

@author: schmi
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import math as m
import scipy.ndimage as nd
# %matplotlib qt5
import matplotlib as mpl
mpl.rc("figure", dpi=120)
# Read image
Folder='Images/'
Name='d1'
Ending='.jpg'
Im = cv2.imread(Folder+Name+Ending)
Im = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)

Val1=2048*2
width = int( Val1)
height = int(Im.shape[0]/Im.shape[1]*Val1)
dim = (width, height)
print('Start Resize')
Im = cv2.resize(Im,dim , interpolation=cv2.INTER_LANCZOS4 )
Im=Im.astype('float')
ImK=nd.gaussian_gradient_magnitude(Im,3)
Im[Im<40]=40
# Im[Im>230]=230

x=np.linspace(0,2,Im.shape[0])*np.pi
y=np.linspace(0,2*Im.shape[1]/Im.shape[0],Im.shape[1])*np.pi
X,Y = np.meshgrid(y, x)


x0=7
y0=7
Phi=np.arctan2(Y-y0,X-x0)
R=np.sqrt((X-x0)**2+(Y-y0)**2)

# R=X+Y
val2=64*2
R=R+np.sin(Phi/2*R**2)*15/val2*0
# R=X+Y
# R=R+np.sin(Phi*5+R*2)*10/val2
x0=np.pi
y0=-np.pi*2
R2=np.sqrt((X-x0)**2+(Y-y0)**2)
t=20*np.pi
u=R2*val2
R3=u % t
Conf=R3>6.283185307179586 *1

ImR=Im
try:
    for lauf in range(0,Im.shape[2]):
        ImC=ImR[:,:,lauf]
        R=np.sqrt((X-r0*m.cos(np.pi/4++np.pi*4/3*lauf*0))**2+(Y-r0*m.sin(np.pi/4++np.pi*4/3*lauf*0))**2)

        Im2=np.sin(R*val2+lauf*2*np.pi/3)+ImC/256*2-1
        Im2[Im2<0]=0
        Im2[Im2>0]=1
        Im[:,:,lauf]=Im2
        
 
    R=np.stack([Im2*0+1,Im2*0,Im2*0],axis=2)
    G=np.stack([Im2*0,Im2*0+1,Im2*0],axis=2)
    B=np.stack([Im2*0,Im2*0,Im2*0+1],axis=2)
    # Im = cv2.cvtColor(Im, cv2.COLOR_BGR2RGB)

    cv2.imwrite(Name + 'R' + '.jpg', Im*R*256)
    cv2.imwrite(Name + 'G' + '.jpg', Im*G*256)
    cv2.imwrite(Name + 'B' + '.jpg', Im*B*256)
    
    conf=np.sum(Im,2)
    conf=conf!=1
    conf=np.stack([conf,conf,conf],axis=2)
    # Im=Im*conf
    Im[conf]=1
    

except:
        # Im=np.sin(X*val2)*np.sin(Y*val2)+Im/256-.1
        Rr=R*val2/2/np.pi
        Ind2=0*Rr
        Minx=np.floor(np.min(Rr*np.cos(Phi)))
        Maxx=np.ceil(np.max(Rr*np.cos(Phi)))
        Miny=np.floor(np.min(Rr*np.sin(Phi)))
        Maxy=np.ceil(np.max(Rr*np.sin(Phi)))
        
        wid=(np.max(Phi)-np.min(Phi))
        PhiM=np.min(Phi)
        # for laufr in range(int(np.floor(np.min(Rr))),int(np.ceil(np.max(Rr)))):
        #         print(laufr)

        #         for laufrand in range(1,10):
        #             # xr=np.random.randint(Minx,Maxx)
        #             # yr=np.random.randint(Miny,Maxy)
                    
        #             # rr=(xr**2+yr**2)**.5
        #             # rr=round(rr)
        #             rr=laufr
        #             # phir=np.arctan2(yr,xr)
        #             phir=np.random.rand(1)*wid+PhiM
        #             valphi=np.pi*2/100/rr
        #             Ind=(Rr<rr+3.2) & (Rr>rr) & (Phi<phir+valphi) & (Phi>phir-valphi)
        #             Ind2+=Ind
                    # print(np.sum(Ind))
        Ind2=Ind2>0
        Im[Ind2]=256

        Im[Im/256<.1]=.1*256
        # Im[Im/256>.9]=256
        
        Im=np.sin(R*val2)+Im/256*2-1
        # Im=np.sin(X*val2+np.sin(Y*15)*5)+np.sin(Y*val2)*0+Im/256*2-1
        Im[Im<0]=0
        Im[Im>0]=1
        
        # Im3=np.sin(R2*val2)+.1*2-1
        # Im3[Conf]=-1
        # Im=np.sin(X*val2+np.sin(Y*15)*5)+np.sin(Y*val2)*0+Im/256*2-1
        # Im[(Im3>0) ]=1
        # mk=np.max(ImK)
        # Im[ImK>.2*mk]=1
        # Im=np.abs(Im-1)
        Im=np.pad(Im,100,'maximum')
        cv2.imwrite(Name + '_Gray' + '.jpg', Im*256)

plt.clf()
Im2=(Im*255)
Im2 = cv2.cvtColor(Im2.astype(np.uint8), cv2.COLOR_RGB2BGR)
plt.imshow(Im2)
plt.axis('off')






