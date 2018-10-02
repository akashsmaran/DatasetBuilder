"""
import numpy as np
import cv2
import sklearn.cluster


sample = cv2.imread("test1/00000005.jpg")
sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
clf = sklearn.cluster.KMeans(n_clusters=2)
labels = clf.fit_predict(sample)

import matplotlib.pyplot as plt

print(labels)
#plt.show()
"""
"""
import numpy as np
import cv2
 
img = cv2.imread('test1/00000005.jpg')
Z = img.reshape((-1,3))
 
# convert to np.float32
Z = np.float32(Z)
 
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 2.0)
K = 3
ret,label,center=cv2.kmeans(Z,K,None,criteria,2,cv2.KMEANS_RANDOM_CENTERS)
 
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('re1',img)
cv2.waitKey(0)
 
cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
import cv2
import numpy as np
 
class Segment:
	def __init__(self,segments=5):
		#define number of segments, with default 5
	       self.segments=segments

	def kmeans(self,image):
	       #Preprocessing step
	       image=cv2.GaussianBlur(image,(7,7),0)
	       vectorized=image.reshape(-1,3)
	       vectorized=np.float32(vectorized)
	       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10, 1.0)
	       ret,label,center=cv2.kmeans(vectorized,self.segments,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	       res = center[label.flatten()]
	       segmented_image = res.reshape((image.shape))
	       return label.reshape((image.shape[0],image.shape[1])), segmented_image.astype(np.uint8)

	def extractComponent(self,image,label_image,label):
	       component=np.zeros(image.shape,np.uint8)
	       component[label_image==label]=image[label_image==label]
	       return component

if __name__=="__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True,
                help = "Path to the image")
    ap.add_argument("-n", "--segments", required = False,
               type = int,  help = "# of clusters")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"])
    if len(sys.argv)==3:      
	seg = Segment()
        label, result= seg.kmeans(image)
    else:
        seg=Segment(args["segments"])
        label, result=seg.kmeans(image)

    cv2.imshow("input",image)
    cv2.imshow("segmented",result)
    cv2.waitKey(0)
    extracted=seg.extractComponent(image,label,3)
    cv2.imshow("extracted",extracted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


