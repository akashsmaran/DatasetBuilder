import cv2
import numpy as np
import glob
from sklearn.cluster import KMeans
import cv2
# import os
## getting the mask from the rgb images
def preprocessing(img, i):
    # resizing using aspect ratio intact and finding the circle
    # reduce size retain aspect ratio intact
    # invert BGR 2 RGB
    RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
    cv2.imwrite("/home/akash/Documents/Projects/unet/data/membrane/train1/image/%d.png"%i, RGB)
    cv2.imwrite("/home/akash/Documents/Projects/unet/data/membrane/test1/%d.png"%i, RGB)
    Ig = RGB[:, :, 2]
    #  convert in to float and get log trasform for contrast streching
    g = 0.2 * (np.log(1 + np.float32(Ig)))
    normalized_image = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)
    # change into uint8
    cvuint = cv2.convertScaleAbs(normalized_image)
    # cvuint8.dtype
    ret, th = cv2.threshold(cvuint, 0, 255, cv2.THRESH_OTSU)
    ret1,th1 = cv2.threshold(Ig,0,255,cv2.THRESH_OTSU)
    # closeing operation
    # from skimage.morphology import disk
    # from skimage.morphology import erosion, dilation, opening, closing, white_tophat
    # selem = disk(30)
    # cls = opening(th, selem)
    # plot_comparison(orig_phantom, eroded, 'erosion')
    # in case using opencv
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35,35))
    cls = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    #Im = cls*rz # the mask with resize image
    # cv2.imwrite('mynew.jpg', mask)
    return (th,th1,cls,normalized_image,RGB)
import argparse
import sys
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")    
ap.add_argument("-n", "--num", required = True, help = "num")
args = vars(ap.parse_args())

path_dir = (args["image"])
print(path_dir)
i = (int)(args["num"])
print('running code....')
img = cv2.imread(path_dir)
(th,th1,cls,g,RGB) = preprocessing(img, i)
from matplotlib import pyplot as plt
    # plt.imshow(cls)
    # plt.show()
    # cv2.imshow('asd',cls)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # plot the data
titles = ['Original Image', 'log_transform','mask using logT','mask without log_T ']
images = [RGB,g,cls,th]
cv2.imwrite('/home/akash/Documents/Projects/unet/data/membrane/train1/label/%d.png'%i, g)
for i in range(0,len(images)):
    print(i)
    plt.subplot(2, 3, i + 1)
    plt.imshow((images[i]),'gray')
    plt.title(titles[i])        
    plt.xticks([]), plt.yticks([])


plt.show()
