# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 23:17:25 2022

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
mpl.rc("figure", dpi=300)

from Functions import imadjust, readimage,SaveImagec,readimageHSV
mpl.rc("figure", dpi=300)
mpl.rcParams['axes.facecolor']='black'
mpl.rcParams['savefig.facecolor']='black'

Folder='Images/'
Name='d1'
Ending='.jpg'
Filename=Folder+Name+Ending
ResizedPX=2048/4
Contrast=[0, 1]
SaveImage=True
#%%
Im, x,y,X,Y=readimage(Filename,Contrast,ResizedPX)
Im=(Im+.1)/1.1
Im=(1-Im)**2/2

x=x*np.pi
y=y*np.pi
X=X*np.pi
Y=Y*np.pi

x0=-0
y0=-0
phi=np.linspace(10,2*1200**2,500000)**.5
rm=np.max(np.sqrt((X+x0)**2+(Y+y0)**2))
xline=rm/np.max(phi)*phi*np.cos(phi)+x0
yline=rm/np.max(phi)*phi*np.sin(phi)+y0
drm=rm/np.max(phi)*2*np.pi

# lwidths=interpolate.griddata((X.flatten(),Y.flatten()), Im.flatten(),(xline,yline),fill_value=-1) #*10*np.sqrt((xline)**2+yline**2)

f = interpolate.RectBivariateSpline(x,y,Im)
lwidths=(f.ev(yline,xline))



xlines=np.array_split(xline, np.ceil(len(phi)/1000000))
ylines=np.array_split(yline, np.ceil(len(phi)/1000000))

plt.clf()
points = np.array([xline, yline]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, linewidths=lwidths,color='white')
a=plt.gca()
# a.set_facecolor("black")
fi=plt.gcf()
fi.patch.set(facecolor = "black")
a.add_collection(lc)
plt.axis('image')
plt.ylim((np.min(x),np.max(x)))
plt.xlim((np.min(y),np.max(y)))
plt.axis('off')
if SaveImage is True:
    SaveImagec(Name + '_LineThickness')
