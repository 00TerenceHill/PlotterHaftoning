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
from skimage import morphology

from Functions import imadjust,ColorQunatization,ColorQunatizationRGB,ColorQunatizationCMYK,readimageLAB,readimageCMYK,linewidth_from_data_units, readimage,SaveImagec,readimageRGB, readimageHSV, ImageByLineChirp
mpl.rc("figure", dpi=220)

#%% Here the definitions must be filled
Folder='Images/'
Name='ML-01-01'
Name='rb-01'
# Name='as1-01'
# Ending='.tif'
Ending='.png'

# Name='Barista-01'
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
EnhanceColorSaturation=0.9
colspace='LAB'
colspace='RGB'
colspace='CMYK'

RemovingSmallPixelRegions=10

ImageWidthPlotter=50
PencilDiameter=.4


x0=0
y0=-1
x0c=0
y0c=-1

Amplitude=.8/1
ChirpFactor=.3
Linedensity=.75*1
#%%
Im,x,y,X,Y=readimage(Filename,Contrast,ResizedPX,color=color)
if CutWhiteParts==True:
    IDX=Im<.03
    IDX=morphology.area_closing(IDX,area_threshold=RemovingSmallPixelRegions)
    IDX=morphology.area_opening(IDX,area_threshold=RemovingSmallPixelRegions)
   
    Im[IDX]=-0.001

Im[Im>0]=Im[Im>0]**.7
ImRGB=readimageRGB(Filename,Contrast,ResizedPX,color=color)



ImLAB=readimageLAB(Filename,Contrast,ResizedPX)


L=ImLAB[:,:,0]
A=ImLAB[:,:,1]
B=ImLAB[:,:,2]
ImColor=(ImLAB[:,:,1]**2+ImLAB[:,:,2]**2)**.5
ImColor=ImColor**(1.001-EnhanceColorSaturation)
ImColor=ImColor/np.max(ImColor)



ImCMYK=readimageCMYK(Filename,Contrast,ResizedPX)
ImCMYK=ImCMYK[:,:,0:3]
lenCol=NumberColors-1

#%% Get the color matrices
# Color[ImBW<.2]=-1 #Choose only bright enough colors no blacks...
# Color[ImColor<.2]=-1 #Choose only bright enough colors no blacks...

plt.imshow(ImRGB,origin='lower')
plt.title('RGB')
plt.show()

plt.imshow(L,cmap='gray',origin='lower')
plt.title('L')
plt.show()
plt.imshow(ImColor,cmap='gray',origin='lower')
plt.title('Saturation')
plt.colorbar()
plt.show()
plt.imshow(A,cmap='RdYlGn',origin='lower')
plt.title('A')
plt.clim(-1,1)
plt.show()
plt.imshow(-B,cmap='PuOr',origin='lower')
plt.clim(-1,1)
plt.title('B')
plt.show()

ImLAB2=ImLAB.copy()
ImLAB2[:,:,0]=0
if colspace=='LAB':
    Labels=ColorQunatization(ImLAB,lenCol,RemoveSmallPixelRegions=RemovingSmallPixelRegions)
    
    
if colspace=='CMYK':
    Labels=ColorQunatizationCMYK(ImCMYK,lenCol,RemoveSmallPixelRegions=RemovingSmallPixelRegions)
else:
    Labels=ColorQunatizationRGB(ImRGB,lenCol,RemoveSmallPixelRegions=RemovingSmallPixelRegions)

ls=np.shape(ImLAB[:,:,0])
ColorIndArray=np.empty((ls[0],ls[1],lenCol),'float64')

for laufcol in range(lenCol):
    
    ColorIndArray[::,::,laufcol]=(np.abs(Labels-laufcol)<1e-3) 
    plt.imshow(ColorIndArray[::,::,laufcol],cmap='gray',origin='lower')
    plt.show()


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

xline,yline=ImageByLineChirp(xline, yline,drm,X,Y,Im,CutWhiteParts=CutWhiteParts,CutNonColor=CutNonColor,ChirpFactor=ChirpFactor)
ind=(xline<0) | (yline<0) | (xline>siz[1]) | (yline>siz[0])
xline[ind]=np.nan
yline[ind]=np.nan

xline=xline/siz[1]*ImageWidthPlotter
yline=yline/siz[0]*ImageWidthPlotter*siz[0]/siz[1]

xlines2=np.array_split(xline, np.ceil(len(xline)/1000000))
ylines2=np.array_split(yline, np.ceil(len(xline)/1000000))

plt.clf()

# plt.clf()
# points = np.array([xline, yline]).T.reshape(-1, 1, 2)
# segments = np.concatenate([points[:-1], points[1:]], axis=1)
# lc = LineCollection(segments, linewidths=lwidths,color='black')
# a=plt.gca()
# a.add_collection(lc)

#%%

for laufcol in range(lenCol):
    ColInd=ColorIndArray[:,:,laufcol]
    if np.sum(ColInd)==0:
        continue
        print('break:' + laufcol)
    x02=(x0c+1)/2*siz[1]
    y02=(y0c+1)/2*siz[0]

    xline=rm/np.max(phi)*phi*np.cos(phi)+x02
    yline=rm/np.max(phi)*phi*np.sin(phi)+y02
    
    ind=(xline<-50) | (yline<-50) | (xline>siz[1]+50) | (yline>siz[0]+50)
    xline=np.delete(xline,ind)
    yline=np.delete(yline,ind)


    ColInd = np.repeat(ColInd[:, :, np.newaxis], 3, axis=2)
    ImCol=ImRGB.copy()
    ImCol[ColInd<0.5]=-.00001
    R=ImCol[:,:,0]
    G=ImCol[:,:,1]
    B=ImCol[:,:,2]
    
    ImCol=(ImCol[:,:,0]**2+ImCol[:,:,1]**2+ImCol[:,:,2]**2)**.5
    ImCol=(ImColor)*AdjustColorBrightness
    ImCol[ColorIndArray[:,:,laufcol]<0.5]=-.00001

    xline,yline=ImageByLineChirp(xline, yline,drm,X,Y,ImCol,CutWhiteParts=CutWhiteParts,CutNonColor=CutNonColor,ChirpFactor=ChirpFactor)
    
    ind=(xline<0) | (yline<0) | (xline>siz[1]) | (yline>siz[0])
    xline[ind]=np.nan
    yline[ind]=np.nan

    xline=xline/siz[1]*ImageWidthPlotter
    yline=yline/siz[0]*ImageWidthPlotter*siz[0]/siz[1]

    xlines=np.array_split(xline, np.ceil(len(xline)/1000000))
    ylines=np.array_split(yline, np.ceil(len(xline)/1000000))
    
    for k in range(0,len(xlines)):
        print(k)
        plt.plot(xlines[k],ylines[k],linewidth=1.1339/3/2,color=(np.mean(R[R>0]), np.mean(G[G>0]), np.mean(B[B>0]) ) )

for k in range(0,len(xlines2)):
    print(k)
    plt.plot(xlines2[k],ylines2[k],linewidth=1.1339/3/2,color='k')

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

