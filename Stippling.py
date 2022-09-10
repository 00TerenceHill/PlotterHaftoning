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
from Functions import imadjust, readimages,SaveImage,SaveImagec
mpl.rc("figure", dpi=300)

# Read image
Folder='Images/'
Name='VG1'
Ending='.jpg'
Filename=Folder+Name+Ending
ResizedPX=2048/4
FrequencyVal=.15*.4
Contrast=[0.05,.95]
n_iter=300
numpoints=26e3
startp=np.floor(numpoints/3/3/3*1.2)
# startp=3000

pointsize=[.2, 1]

Im, x,y,X,Y=readimages(Filename,Contrast,ResizedPX)
Im=Im**1
plt.imshow(1-Im,origin='lower',cmap='gray')
plt.axis('off')
plt.show()

print('Calculating image')

xline=np.random.rand(int(startp))*(np.max(x))
yline=np.random.rand(int(startp))*(np.max(y))


xline=np.linspace(0,np.max(x),100)
yline=np.linspace(0,np.max(y),100)
dx=xline[2]-xline[1]
dy=yline[2]-yline[1]

xline,yline=np.meshgrid(xline,yline)
xline=xline.flatten()
yline=yline.flatten()
xline2=xline+dx/2
yline2=yline+dy/2
xline=np.append(xline,xline2)
yline=np.append(yline,yline2)
ImInt=interpolate.griddata((X.flatten(),Y.flatten()), Im.flatten(),(xline,yline),fill_value=-1) #*10*np.sqrt((xline)**2+yline**2)
xline=np.delete(xline,ImInt<0.1)
yline=np.delete(yline,ImInt<0.1)

plt.plot(xline,yline)
# %%
fig=plt.figure(1)
plt.clf()
# print(np.sum(pos))
density, x,y,X,Y=readimages(Filename,Contrast,ResizedPX*4)
density_P = density.cumsum(axis=1)
density_Q = density_P.cumsum(axis=1)
points=np.array([xline*4,yline*4]).T
lauf=1
while len(points)<numpoints:
    if lauf>1:
        
        SaveImagec(Name + '_Stippling' +str(lauf))
        tri = sc.spatial.Delaunay(points)
        verts = tri.points[tri.vertices]
        # Triangle vertices
        A = verts[:,0,:].T
        B = verts[:,1,:].T
        C = verts[:,2,:].T
        Aver=np.array((A+B+C)/3).T
        MeanP=(A+B+C)/3
        xc=MeanP[0,:]
        yc=MeanP[1,:]

        xn=np.floor(xc).astype(int)
        yn=np.floor(abs(yc)).astype(int)
        # colors = self.ImRGB[yn, xn,:]

        points=np.append(points,Aver,axis=0)
        print(len(points))
        print(len(points)/startp)
        
    for i in range(n_iter):
        print(i)
       
            
        regions, points = voronoi.centroids(points, density, density_P, density_Q)
        plt.cla()
        # tri = sc.spatial.Delaunay(points)
        # verts = tri.points[tri.vertices]
        # collection = trip.PolyCollection(verts)
        # collection.set_facecolor(tri.vertices*0+1)
        # collection.set_edgecolors(tri.vertices*0)
        # plt.gca().add_collection(collection)
        # plt.gca().autoscale_view()

        scatter = plt.scatter(points[:, 0], points[:, 1], s=1,
                                     facecolor="k", edgecolor="None")
    
        Pi = points.astype(int)
        X = np.maximum(np.minimum(Pi[:, 0], density.shape[1]-1), 0)
        Y = np.maximum(np.minimum(Pi[:, 1], density.shape[0]-1), 0)
        sizes = (pointsize[0] +
                 (pointsize[1]-pointsize[0])*density[Y, X])
        scatter.set_offsets(points)
        scatter.set_sizes(sizes)
        plt.axis('off')
        plt.axis('image')
        plt.axis('off')
        fig.canvas.draw()
        fig.canvas.flush_events()
        lauf=lauf+1
        
        display.clear_output(wait=True)
        display.display(plt.gcf())
    

SaveImagec(Name + '_Stippling')

#%%

