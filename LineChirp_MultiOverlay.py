# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 23:09:02 2022

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
from Functions import imadjust,readimageHSV_Alternative,linewidth_from_data_units, readimage,SaveImagec,readimageRGB, readimageHSV, ImageByLineChirp,ImageByLineChirp2
mpl.rc("figure", dpi=220)

#%% Here the definitions must be filled
Folder='Images/'
Name='d1'
Ending='.jpg'
color='HSV'
Filename=Folder+Name+Ending
SaveImage=True
ResizedPX=2048/4
Contrast=[0,1]
CutWhiteParts=False
CutNonColor=False
NumberColors=1
AdjustColorBrightness=1

ImageWidthPlotter=100
PencilDiameter=.4


x0=-1
y0=.2
x0c=-1
y0c=.207

Amplitude=.6
ChirpFactor=.7/2
Linedensity=.75*1
Linedensity2=.75*1.01

#%%
Im,x,y,X,Y=readimage(Filename,Contrast,ResizedPX,color=color)
Im=Im**.7
ImRGB=readimageRGB(Filename,Contrast,ResizedPX,color=color)

ImHSV=readimageHSV_Alternative(Filename,Contrast,ResizedPX)
Color=ImHSV[:,:,0]
Color=medfilt2d(Color,5)
ImColor=ImHSV[:,:,1]
ImBW=ImHSV[:,:,2]
if NumberColors>1:
    # Im=(1-ImBW)**.4
    # ImColor[Im<.2]=ImColor[Im<.2]*2
    ImColor=ImColor**.7



lenCol=NumberColors-1

#%% Get the color matrices


# plt.imshow(ImBW,cmap='gray',origin='lower')
# plt.title('BW')
# plt.show()
# plt.imshow(ImColor,cmap='gray',origin='lower')
# plt.title('Saturation')
# plt.show()
# plt.imshow(Color,cmap='gray',origin='lower')
# plt.title('Color')
# plt.show()



Color=medfilt2d(Color,5)
hist, bin_edges = np.histogram(Color.flatten(),bins=282,range=(-.1,1))
peaks, Res = find_peaks(hist, height=100,distance=5)
Res=Res['peak_heights']
PeakPos=np.argsort(Res)
PeakPos=peaks[PeakPos[::-1]]
Col=bin_edges[PeakPos[0:lenCol]]
ls=np.shape(Color)
ColorIndArray=np.empty((ls[0],ls[1],lenCol),'float64')

for laufcol in range(lenCol):
    print(laufcol)
    ColorInd=0*ImColor
    ColorInd[np.abs(Color-Col[laufcol])<3/256]=1
    plt.imshow(ColorInd,cmap='gray',origin='lower')
    plt.show()
    
    ColorIndArray[::,::,laufcol]=ColorInd

if CutNonColor is True:
    Im[(np.sum(ColorIndArray,2)>0)]=-.00001


# Im[np.sum(ColorIndArray,2)>0]=-.00001
# Im=medfilt2d(Im,3)
#%%
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

xline,yline=ImageByLineChirp(xline, yline,drm,X,Y,Im*AdjustColorBrightness,CutWhiteParts=CutWhiteParts,CutNonColor=CutNonColor,ChirpFactor=ChirpFactor)
ind=(xline<0) | (yline<0) | (xline>siz[1]) | (yline>siz[0])
xline[ind]=np.nan
yline[ind]=np.nan

xline=xline/siz[1]*ImageWidthPlotter
yline=yline/siz[0]*ImageWidthPlotter*siz[0]/siz[1]

xlines2=np.array_split(xline, np.ceil(len(xline)/1000000))
ylines2=np.array_split(yline, np.ceil(len(xline)/1000000))
plt.clf()

for k in range(0,len(xlines2)):
    print(k)
    plt.plot(xlines2[k],ylines2[k],linewidth=1.1339/3/2,color='c')

#%%
x01=(x0c+1)/2*siz[1]
y01=(y0c+1)/2*siz[0]
phi=np.linspace(10,Linedensity2*1.5*1200**2,int(5e6*2*Linedensity2))**.5


rm=np.max(np.sqrt((X+x0c)**2+(Y+y0c)**2))*2
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


for k in range(0,len(xlines2)):
    print(k)
    plt.plot(xlines2[k],ylines2[k],linewidth=1.1339/3/2,color='k')

#%%


plt.axis('image')
plt.xlim((0, ImageWidthPlotter))
plt.ylim(( 0,ImageWidthPlotter*siz[0]/siz[1]))
plt.axis('off')


LW=linewidth_from_data_units(PencilDiameter, plt.gca(), reference='y')
for line in plt.gca().lines:
    line.set_linewidth(LW)


if SaveImage is True:
    SaveImagec(Name + '_Line2_' + str(lenCol) + '#Colors')
plt.show()

