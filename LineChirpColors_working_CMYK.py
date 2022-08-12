# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:49:27 2022

@author: Terence
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 11:57:21 2022

@author: schmi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 22:37:51 2022

@author: schmi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 18:17:50 2022

@author: schmi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:58:40 2022

@author: schmi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 13:50:28 2022

@author: schmi
"""
# from DieelectricDia import SaveImage
import os
from matplotlib.collections import LineCollection
from mpl_toolkits import mplot3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math as m
import scipy.ndimage as nd
# %matplotlib qt5
import matplotlib as mpl
from scipy import interpolate
from scipy.signal import medfilt2d, find_peaks
from Functions import imadjust,ColorQunatization,ColorQunatizationRGB,readimageCMYK,linewidth_from_data_units, readimage,SaveImagec,readimageRGB, readimageHSV, ImageByLineChirp
mpl.rc("figure", dpi=220)

#%% Here the definitions must be filled
Folder='Images/'
Name='ML-01-01'
Name='Fritz4-01'
# Name='as1-01'
# Ending='.tif'
Ending='.png'

# Name='ampersand-grid-1200'
# Ending='.jpg'
color='HSV'
Filename=Folder+Name+Ending
SaveImage=True
ResizedPX=2048/4    
Contrast=[0,1]
CutWhiteParts=True
CutNonColor=True
NumberColors=5
AdjustColorBrightness=.7
EnhanceColorSaturation=0.2
colspace='LAB'
# colspace='RGB'


ImageWidthPlotter=500
PencilDiameter=.3


x0=-.8
y0=.0
x0c=-.9
y0c=.9

Amplitude=.8
ChirpFactor=.35
Linedensity=.75*4.5
#%%
Im,x,y,X,Y=readimage(Filename,Contrast,ResizedPX,color=color)
if CutWhiteParts==True:
    Im[Im<.03]=0

Im=Im**.7
ImRGB=readimageRGB(Filename,Contrast,ResizedPX,color=color)



ImCMYK=readimageCMYK(Filename,Contrast,ResizedPX)

Cc=ImCMYK[:,:,0]
Mc=ImCMYK[:,:,1]
Yc=ImCMYK[:,:,2]
Kc=ImCMYK[:,:,3]

plt.imshow(ImRGB,origin='lower')
plt.title('RGB')
plt.show()

plt.imshow(Cc,cmap='gray')
plt.title('C')
plt.colorbar()
plt.show()
plt.imshow(Mc,cmap='gray')
plt.title('M')
plt.colorbar()
plt.show()
plt.imshow(Yc,cmap='gray')
plt.title('Y')
plt.colorbar()
plt.show()
plt.imshow(Kc,cmap='gray')
# plt.clim(-1,1)
plt.colorbar()
plt.show()

#%% Get the color matrices
# Color[ImBW<.2]=-1 #Choose only bright enough colors no blacks...
# Color[ImColor<.2]=-1 #Choose only bright enough colors no blacks...


#%%
plt.clf()
for lauf in range(4):
    if lauf==0:
        Im=Cc
        col='c'
        x0=0
        y0=1
    elif lauf==1:
        Im=Mc
        col='m'       
        x0=1
        y0=0
    elif lauf==2:
        Im=Yc
        col='y'
        x0=0
        y0=-1
    elif lauf==3:
        Im=Kc
        col='k'
        x0=-1
        y0=0
    Im[Im<.03]=-0.001
    siz=np.shape(Im)
    x01=(x0+1)/2*siz[1]
    y01=(y0+1)/2*siz[0]
    phi=np.linspace(10,Linedensity*1.5*1200**2,int(5e6*2*Linedensity))**.5
    rm=np.max(np.sqrt((X+x0)**2+(Y+y0)**2))*2
    xline=rm/np.max(phi)*phi*np.cos(phi)+x01
    yline=rm/np.max(phi)*phi*np.sin(phi)+y01
    drm=rm/np.max(phi)*2*np.pi*Amplitude*1.1
    
    ind=(xline<-50) | (yline<-50) | (xline>siz[1]+50) | (yline>siz[0]+50)
    xline=np.delete(xline,ind)
    yline=np.delete(yline,ind)
    
    xline,yline=ImageByLineChirp(xline, yline,drm,X,Y,Im,CutWhiteParts=CutWhiteParts,CutNonColor=CutNonColor,ChirpFactor=ChirpFactor)
    ind=(xline<0) | (yline<0) | (xline>siz[1]) | (yline>siz[0])
    xline[ind]=np.nan
    yline[ind]=np.nan
    
    xline=xline/siz[1]*ImageWidthPlotter
    yline=yline/siz[0]*ImageWidthPlotter*siz[0]/siz[1]
    
    xlines2=np.array_split(xline, np.ceil(len(xline)/1000000))
    ylines2=np.array_split(yline, np.ceil(len(xline)/1000000))
    
    # plt.clf()
    
    for k in range(0,len(xlines2)):
        print(k)
        plt.plot(xlines2[k],ylines2[k],linewidth=1.1339/3/2,color=col)

plt.axis('image')
plt.xlim((0, ImageWidthPlotter))
plt.ylim(( 0,ImageWidthPlotter*siz[0]/siz[1]))
plt.axis('off')


LW=linewidth_from_data_units(PencilDiameter, plt.gca(), reference='y')
for line in plt.gca().lines:
    line.set_linewidth(LW)


if SaveImage is True:
    SaveImagec(Name + '_Line2_CMYK')
plt.show()

