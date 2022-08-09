import cv2
import numpy as np

import Functions as fu

import matplotlib.pyplot as plt
# %matplotlib qt5
def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.
    Im=x.astype('float')
    maxi=np.max(Im[:])
    Im=Im/maxi
    
    try:
        for lauf in range(0,Im.shape[2]):
            x=Im[:,:,lauf]
            y = (((x - a) / (b - a)) ) * (d - c) + c
            y[y<c]=c
            y[y>d]=d
            Im[:,:,lauf]=y
    except:
        x=Im
        y = (((x - a) / (b - a)) ) * (d - c) + c
        y[y<c]=c
        y[y>d]=d
        Im=y

    return Im

# Read image
Folder='Images/'
Name='d1'
Ending='.jpg'
color='HSV'
Filename=Folder+Name+Ending
Im = cv2.imread(Filename)
#   Im = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)


scale_percent = 1600 # percent of original size
width = int( 6000)
height = int(Im.shape[0]/Im.shape[1]*6000)
dim = (width, height)
print('Start Resize')
Im = cv2.resize(Im,dim , interpolation=cv2.INTER_LANCZOS4 )
# Im = cv2.resize(Im,dim , interpolation=cv2.INTER_CUBIC )
# Im = cv2.resize(Im,dim , interpolation=cv2.INTER_NEAREST)
##
print('Start Imadjust')

Im=imadjust(Im,.3,.9,0,1)
# result=image.
# Save the image
print('Start Save')
plt.imshow(Im)
plt.axis('image')
plt.axis('off')
fu.SaveImagec(Name + 'UpScaled')
plt.show()
# plt.savefig(Name + '_UpScaled' +'.jpg', dpi=700, bbox_inches='tight', pad_inches=0)
print('Start Plot')

# plt.imshow(Im)
# plt.draw()
# plt.show()##  




