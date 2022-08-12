# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:25:57 2022

@author: Terence
"""

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
import Functions as fu

from scipy import interpolate
mpl.rc("figure", dpi=720)
# %% Read image
Folder='Images/'
Name='VG1'
Ending='.jpg'
Im = cv2.imread(Folder+Name+Ending)
Im = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)

SaveImage=True
ResizedPX=2048
Pad=100
Linedensity=64*5 

Contrast=[0,1]
x0=1.5
y0=-.5
linew=.3
#%%

Filename=Folder+Name+Ending
Im,x,y,X,Y=fu.readimage_normalized(Filename,Contrast,ResizedPX)
Im=np.abs(1-Im)
Im[Im<.1]=.1

Phi=np.arctan2(Y-y0,X-x0)
R=np.sqrt((X-x0)**2+(Y-y0)**2)
Im=np.sin(R*Linedensity)+(Im*2-1)
dRm=2*np.pi/Linedensity
Im[Im<0]=0
Im[Im>0]=1
Im=np.pad(Im,Pad,'maximum')

xr=np.linspace(np.min(x),np.max(x),20)
yr=np.linspace(np.min(y),np.max(y),20)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

rho, phi=cart2pol(xr,yr)
rho=np.ceil(rho*Linedensity/2/np.pi)*2*np.pi/Linedensity
xr,yr=pol2cart(rho,phi)
xr=xr
yr=yr



xr,yr = np.meshgrid(xr, yr)


xr=xr.flatten()
yr=yr.flatten()

xr=xr+2*(np.random.rand(len(xr))-.5)*.45*(xr[2]-xr[1])
yr=yr+2*(np.random.rand(len(xr))-.5)*.45*(xr[2]-xr[1])



# xr=np.random.rand(1,100)*2-1
# yr=np.random.rand(1,100)*2-1
dRx,dRy=np.gradient(R)
Nr=(dRx**2+dRy**2)**.5
dRx=dRx/Nr
dRy=dRy/Nr

dRx2 = interpolate.RectBivariateSpline(y, x, dRx)
dRy2 = interpolate.RectBivariateSpline(y, x, dRy)


# Z2 = interp_spline(y2, x2)

xr2=xr+dRy2.ev(yr,xr)*2.5*dRm
yr2=yr+dRx2.ev(yr,xr)*2.5*dRm

xr4=np.vstack((xr,xr2))
yr4=np.vstack((yr,yr2))
xr4=Pad+(xr4+np.max(x))/np.max(x)*(np.shape(Im)[1]-2*Pad)/2
yr4=Pad+(yr4+np.max(y))/np.max(y)*(np.shape(Im)[0]-2*Pad)/2


#%%


plt.imshow(Im,cmap='gray',origin='lower')
plt.axis('image')
plt.axis('off')
plt.plot(xr4,yr4,'w',linewidth=linew)

if SaveImage is True:
    fu.SaveImagec(Name + '_HaftoneLiethickness')
plt.show()





