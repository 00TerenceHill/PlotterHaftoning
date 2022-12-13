# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 22:26:07 2022

@author: schmi
"""
import scipy as sc
import voronoi
import matplotlib.tri as trip
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math as m
import scipy.ndimage as nd
# %matplotlib qt5
import matplotlib as mpl
from scipy import interpolate
import os
from IPython import display
from skimage import io
# from sklearn.cluster import KMeans
from scipy.signal import medfilt2d, find_peaks
from skimage import morphology
from scipy.signal import convolve2d

from scipy import interpolate
class data_linewidth_plot():
    def __init__(self, x, y, **kwargs):
        self.ax = kwargs.pop("ax", plt.gca())
        self.fig = self.ax.get_figure()
        self.lw_data = kwargs.pop("linewidth", 1)
        self.lw = 1
        self.fig.canvas.draw()

        self.ppd = 72./self.fig.dpi
        self.trans = self.ax.transData.transform
        self.linehandle, = self.ax.plot([],[],**kwargs)
        if "label" in kwargs: kwargs.pop("label")
        self.line, = self.ax.plot(x, y, **kwargs)
        self.line.set_color(self.linehandle.get_color())
        self._resize()
        self.cid = self.fig.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self, event=None):
        lw =  ((self.trans((1, self.lw_data))-self.trans((0, 0)))*self.ppd)[1]
        if lw != self.lw:
            self.line.set_linewidth(lw)
            self.lw = lw
            self._redraw_later()

    def _redraw_later(self):
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.fig.canvas.draw_idle())
        self.timer.start()

def linewidth_from_data_units(linewidth, axis, reference='y'):
    """
    Convert a linewidth in data units to linewidth in points.

    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    fig = axis.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == 'y':
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())
    # Convert length to points
    length *= 72
    # Scale linewidth to value range
    return linewidth * (length / value_range)

def ImageByLineChirp(xline,yline,drm,X,Y,ImU,CutWhiteParts=False,CutNonColor=False,ChirpFactor=1):
    dxl=np.gradient(xline)
    dyl=np.gradient(yline)
    dxr=np.sqrt(dxl**2+dyl**2)
    dxl=dxl/dxr
    dyl=dyl/dxr
    lwidths2=interpolate.griddata((X.flatten(),Y.flatten()), ImU.flatten(),(xline,yline),fill_value=-1) #*10*np.sqrt((xline)**2+yline**2)
    ind=lwidths2<-5
    lwidths2=np.delete(lwidths2,ind)
    xline=np.delete(xline,ind)
    yline=np.delete(yline,ind)
    dyl=np.delete(dyl,ind)
    dxl=np.delete(dxl,ind)
    lwidths=np.cumsum(lwidths2*.15*4*2)
    xline=xline-dyl*np.sin(lwidths*ChirpFactor)*imadjust(lwidths2,0,.8,0,1,gamma=1)*drm/2
    yline=yline+dxl*np.sin(lwidths*ChirpFactor)*imadjust(lwidths2,0,.8,0,1,gamma=1)*drm/2
    if CutWhiteParts is True:
        xline[(lwidths2<.0)  ]=np.nan
        yline[(lwidths2<.0) ]=np.nan
    if CutNonColor is True:
        xline[lwidths2<0]=np.nan
        yline[lwidths2<0]=np.nan
    
    return xline,yline

def ImageByLineChirp2(xline,yline,drm,X,Y,ImU,CutWhiteParts=False,CutNonColor=False,ChirpFactor=1):
    dxl=np.gradient(xline)
    dyl=np.gradient(yline)
    dxr=np.sqrt(dxl**2+dyl**2)
    dxl=dxl/dxr
    dyl=dyl/dxr
    lwidths2=interpolate.griddata((X.flatten(),Y.flatten()), ImU.flatten(),(xline,yline),fill_value=-1) #*10*np.sqrt((xline)**2+yline**2)
    ind=lwidths2<-5
    lwidths2=np.delete(lwidths2,ind)
    xline=np.delete(xline,ind)
    yline=np.delete(yline,ind)
    dyl=np.delete(dyl,ind)
    dxl=np.delete(dxl,ind)
    lwidths=np.cumsum(lwidths2*.15*4*2)
    xline=xline-dyl*np.sin(lwidths*ChirpFactor+lwidths2*np.pi)*lwidths2*drm/2
    yline=yline+dxl*np.sin(lwidths*ChirpFactor+lwidths2*np.pi)*lwidths2*drm/2
    if CutWhiteParts is True:
        xline[(lwidths2<.13)  ]=np.nan
        yline[(lwidths2<.13) ]=np.nan
    if CutNonColor is True:
        xline[lwidths2<0]=np.nan
        yline[lwidths2<0]=np.nan
    
    return xline,yline


def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = np.abs(((x - a) / (b - a)) )**gamma * (d - c) + c
    y[y<c]=c
    y[y>d]=d
    
    return y

def readimage(Filename,Contrast=[0, 0],ResizedPX=[512],color='gray'):
    Im = cv2.imread(Filename)
    Im = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)

    # Im=(Im-np.min(Im))/(np.max(Im)-np.min(Im))*256
    width = int( ResizedPX)
    height = int(Im.shape[0]/Im.shape[1]*ResizedPX)
    dim = (width, height)
    print('Start Resize')
    Im = cv2.resize(Im,dim , interpolation=cv2.INTER_CUBIC )
    Im=Im.astype('float')
    Im=Im[::-1, :]
    Im=Im/256
    Im=1-Im
    Im=imadjust(Im, Contrast[0],Contrast[1],0,1)
    x=-np.linspace(-1,1,Im.shape[0])*np.pi*1
    y=np.linspace(-1*Im.shape[1]/Im.shape[0],1*Im.shape[1]/Im.shape[0],Im.shape[1])*np.pi*1
    x=np.linspace(0,Im.shape[0]-1,Im.shape[0])
    y=np.linspace(0,Im.shape[1]-1,Im.shape[1])
    X,Y = np.meshgrid(y, x)
    return Im, x,y,X,Y     
def readimage_normalized(Filename,Contrast=[0, 0],ResizedPX=[512],color='gray'):
    Im = cv2.imread(Filename)
    Im = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)

    # Im=(Im-np.min(Im))/(np.max(Im)-np.min(Im))*256
    width = int( ResizedPX)
    height = int(Im.shape[0]/Im.shape[1]*ResizedPX)
    dim = (width, height)
    print('Start Resize')
    Im = cv2.resize(Im,dim , interpolation=cv2.INTER_CUBIC )
    Im=Im.astype('float')
    Im=Im[::-1, :]
    Im=Im/256
    Im=1-Im
    Im=imadjust(Im, Contrast[0],Contrast[1],0,1)
    x=np.linspace(-1,1,Im.shape[1])
    y=np.linspace(-1*Im.shape[0]/Im.shape[1],1*Im.shape[0]/Im.shape[1],Im.shape[0])
    # x=np.linspace(-1,1,Im.shape[0])
    # y=np.linspace(-1,1,Im.shape[1])
    X,Y = np.meshgrid(x,y)
    return Im, x,y,X,Y  

def readimageRGB(Filename,Contrast=[0, 0],ResizedPX=[512],color='gray'):
    Im = cv2.imread(Filename)
    Im = cv2.cvtColor(Im, cv2.COLOR_BGR2RGB)
        
    # Im=(Im-np.min(Im))/(np.max(Im)-np.min(Im))*256
    width = int( ResizedPX)
    height = int(Im.shape[0]/Im.shape[1]*ResizedPX)
    dim = (width, height)
    print('Start Resize')
    Im = cv2.resize(Im,dim , interpolation=cv2.INTER_CUBIC )
    Im=Im.astype('float')
    Im=Im[::-1, :]
    Im=Im/256
    return Im

def readimageHSV(Filename,Contrast=[0, 0],ResizedPX=[512],color='gray',NumberOfColors=64):
    Im = cv2.imread(Filename)
    width = int( ResizedPX)
    height = int(Im.shape[0]/Im.shape[1]*ResizedPX)
    dim = (width, height)
    print('Start Resize')
    Im = cv2.resize(Im,dim , interpolation=cv2.INTER_CUBIC )
    # Im=ColorQunatization(Im,NumberOfColors)
    
    Im = cv2.cvtColor(Im, cv2.COLOR_BGR2HSV)
        
    # Im=(Im-np.min(Im))/(np.max(Im)-np.min(Im))*256
    Im=Im.astype('float')
    Im=Im[::-1, :,:]
    Im=Im/256
    return Im

def readimageLAB(Filename,Contrast=[0, 0],ResizedPX=[512],color='gray',NumberOfColors=64):
    Im = cv2.imread(Filename)
    width = int( ResizedPX)
    height = int(Im.shape[0]/Im.shape[1]*ResizedPX)
    dim = (width, height)
    print('Start Resize')
    Im = cv2.resize(Im,dim , interpolation=cv2.INTER_CUBIC )
    # Im=ColorQunatization(Im,NumberOfColors)
    
    Im = cv2.cvtColor(Im, cv2.COLOR_BGR2Lab)
        
    # Im=(Im-np.min(Im))/(np.max(Im)-np.min(Im))*256
    Im=Im.astype('float')
    Im=Im[::-1, :,:]
    Im=Im/255
    Im[:,:,1]=(Im[:,:,1]-.5)*2
    Im[:,:,2]=(Im[:,:,2]-.5)*2
    
    return Im


def readimageCMYK(Filename,Contrast=[0, 0],ResizedPX=[512],color='gray',NumberOfColors=64):

    # from PIL import Image
    # image = Image.open(Filename)
    
    # cmyk_image = image.convert('CMYK')
    # Im=np.asarray(cmyk_image)
    # Im=Im.astype('float')
    
    # width = int( ResizedPX)
    # height = int(Im.shape[0]/Im.shape[1]*ResizedPX)
    # dim = (width, height)
    # print('Start Resize')
    # Im = cv2.resize(Im,dim , interpolation=cv2.INTER_CUBIC )
    # # Im=ColorQunatization(Im,NumberOfColors)
    # Im=Im/255
    # Im[Im>1]=1
    # Im=Im[::-1, :,:]
    # return Im
    Im = cv2.imread(Filename)
    width = int( ResizedPX)
    height = int(Im.shape[0]/Im.shape[1]*ResizedPX)
    dim = (width, height)
    print('Start Resize')
    Im = cv2.resize(Im,dim , interpolation=cv2.INTER_CUBIC )
    # Im=ColorQunatization(Im,NumberOfColors)
    
    
    img = Im.astype(np.float64)/255.
    K = 1 - np.max(img, axis=2)
    C = (1-img[...,2] - K)/(1-K+.001)
    M = (1-img[...,1] - K)/(1-K+.001)
    Y = (1-img[...,0] - K)/(1-K+.001)

    
    CMYK_image= (np.dstack((C,M,Y,K)))

    CMYK_image=CMYK_image[::-1, :,:]
    return CMYK_image

def readimageHSV_Alternative(Filename,Contrast=[0, 0],ResizedPX=[512],color='gray',NumberOfColors=64):
    Im = cv2.imread(Filename)
    width = int( ResizedPX)
    height = int(Im.shape[0]/Im.shape[1]*ResizedPX)
    dim = (width, height)
    print('Start Resize')
    Im = cv2.resize(Im,dim , interpolation=cv2.INTER_CUBIC )
    # Im=ColorQunatization(Im,NumberOfColors)
    Im2 = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)
    # Im[:,:,0]=np.max(Im2)-Im2
    # Im[:,:,1]=np.max(Im2)-Im2
    # Im[:,:,2]=0

    Im[:,:,0]=Im2
    Im[:,:,1]=Im2
    Im[:,:,2]=np.max(Im2)

    Im = cv2.cvtColor(Im, cv2.COLOR_BGR2HSV)
        
    # Im=(Im-np.min(Im))/(np.max(Im)-np.min(Im))*256
    Im=Im.astype('float')
    Im=Im[::-1, :,:]
    Im=Im/256
    return Im


def readimagec(Filename,Contrast=[0, 0],ResizedPX=[512]):
    Im = cv2.imread(Filename)
    Im = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)
    # Im=(Im-np.min(Im))/(np.max(Im)-np.min(Im))*256
    width = int( ResizedPX)
    height = int(Im.shape[0]/Im.shape[1]*ResizedPX)
    dim = (width, height)
    print('Start Resize')
    Im = cv2.resize(Im,dim , interpolation=cv2.INTER_CUBIC )
    Im=Im.astype('float')
    Im=Im[::-1, :]
    Im=Im/256
    Im=1-Im
    Im=imadjust(Im, Contrast[0],Contrast[1],0,1)
    x=np.linspace(0,Im.shape[0]-1,Im.shape[0])
    y=np.linspace(0,Im.shape[1]-1,Im.shape[1])
    x=x-np.mean(x)
    y=y-np.mean(y)
    x=x/np.max(x)
    y=y/np.max(y)
    X,Y=np.meshgrid(x,y)

    X,Y = np.meshgrid(y, x)
    return Im, x,y,X,Y

# def ColorQunatization(Im,NumberOfColors):
     
#     original = Im
#     n_colors = NumberOfColors
    
#     arr = original.reshape((-1, 3))
#     kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)
#     labels = kmeans.labels_
#     centers = kmeans.cluster_centers_
#     less_colors = centers[labels].reshape(original.shape)

#     return   less_colors  

def inpaint_nans(im):
    
    ipn_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]]) # kernel for inpaint_nans
    nans = np.isnan(im)
    while np.sum(nans)>0:
        im[nans] = 0
        vNeighbors = convolve2d((nans==False),ipn_kernel,mode='same',boundary='symm')
        im2 = convolve2d(im,ipn_kernel,mode='same',boundary='symm')
        im2[vNeighbors>0] = im2[vNeighbors>0]/vNeighbors[vNeighbors>0]
        im2[vNeighbors==0] = np.nan
        im2[(nans==False)] = im[(nans==False)]
        im = im2
        nans = np.isnan(im)
    return im
def ColorQunatization(Im,NumberOfColors,RemoveSmallPixelRegions=15):
    L=Im[:,:,0]
    A=Im[:,:,1]
    B=Im[:,:,2]
    ImColor=(A**2+B**2)**.5
    # Gray=(A**2+B**2+L**2)**.5
    indx=(ImColor<.02) | (L<.2)
    A[indx]=0
    B[indx]=0
    L[indx]=0
    
    Im[:,:,0]=L
    Im[:,:,1]=A
    Im[:,:,2]=B
    
    image=Im*256
    k=NumberOfColors+1
    
    
    
    i = np.float32(image).reshape(-1,3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    ret,label,center = cv2.kmeans(i, k , None, condition,10,cv2.KMEANS_RANDOM_CENTERS)
    # center = np.uint8(center)
    # final_img = center[label.flatten()]
    # final_img = final_img.reshape(image.shape)
    Label=label.reshape(image.shape[0:2])

    val=np.mean(Label[indx])
    Label[indx]=-1
    Label[Label>val]=Label[Label>val]-1    
    Label=morphology.area_closing(Label,area_threshold=RemoveSmallPixelRegions)
    Label=morphology.area_opening(Label,area_threshold=RemoveSmallPixelRegions)
    # Label=medfilt2d(Label,5)
    return Label

def ColorQunatizationRGB(Im,NumberOfColors,RemoveSmallPixelRegions=15):
    R=Im[:,:,0]
    G=Im[:,:,1]
    B=Im[:,:,2]
    # ImColor=(A**2+B**2)**.5
    Gray=(R**2+G**2+B**2)**.5/3**.5
    indx=(Gray>.8) | (Gray<.2)
    R[indx]=0
    G[indx]=0
    B[indx]=0
    
    Im[:,:,0]=R
    Im[:,:,1]=G
    Im[:,:,2]=B
    
    image=Im*256
    k=NumberOfColors+1
    
    
    
    i = np.float32(image).reshape(-1,3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,50,1.0)
    ret,label,center = cv2.kmeans(i, k , None, condition,10,cv2.KMEANS_RANDOM_CENTERS)
    # center = np.uint8(center)
    # final_img = center[label.flatten()]
    # final_img = final_img.reshape(image.shape)
    final_img=label.reshape(image.shape[0:2])
    # for lauf in range(k):
    #     lab =final_img==lauf
    #     lab = morphology.binary_dilation(
    #         lab, np.ones((3,3))      )
    #     # lab= cv2.dilate(lab, np.ones((5,5)), iterations=1)
    #     lab2 = morphology.remove_small_objects(lab, 150)
    #     lab2 = morphology.remove_small_holes(lab2, 150)
        
    #     lab3=0*final_img
    #     lab3[lab2]=lauf
    #     final_img[lab]=lab3[lab]
    # final_img=medfilt2d(final_img,5)
    
    # final_img=final_img.astype('float')
    # valid_mask = final_img > 0
    # coords = np.array(np.nonzero(valid_mask)).T
    # values = final_img[valid_mask]
    # it = interpolate.LinearNDInterpolator(coords, values, fill_value=-1)
    # final_img = it(list(np.ndindex(final_img.shape))).reshape(final_img.shape)
    # final_img=np.round(final_img)

    val=np.mean(final_img[indx])
    final_img[indx]=-1
    final_img[final_img>val]=final_img[final_img>val]-1    
    final_img=morphology.area_closing(final_img,area_threshold=RemoveSmallPixelRegions)
    final_img=morphology.area_opening(final_img,area_threshold=RemoveSmallPixelRegions)
    # final_img=medfilt2d(final_img,3)
    return final_img

def SaveImage(Name):
    directory = os.path.dirname('./ResultsPDF/' + Name + '/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname('./ResultsJPG/' + Name + '/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig('./ResultsPDF/' + Name + '/' + Name +   '.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('./ResultsJPG/' + Name + '/' + Name +   '.jpg', bbox_inches='tight', pad_inches=0)

def SaveImagec(Name):
    directory = os.path.dirname('./ResultsPDF/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname('./ResultsJPG/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname('./ResultsSVG/')
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    plt.savefig('./ResultsJPG/' + Name +   '.jpg', dpi=700, bbox_inches='tight', pad_inches=0)
    plt.savefig('./ResultsPDF/' + Name  +   '.pdf',transparent=True, bbox_inches='tight', pad_inches=0)
    plt.savefig('./ResultsSVG/' + Name  +   '.svg',transparent=True, bbox_inches='tight', pad_inches=0)

                