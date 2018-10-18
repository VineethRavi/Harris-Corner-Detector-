import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
# Importing neccessary libraries for HW4

# The function below is used for drawing lines, between points.
# The line is drawn between the two images, where corresponding points are matched.
def draw_lines(image1,sift_pts1,image2,sift_pts2):
    
    s1=np.shape(image1)
    s2=np.shape(image2)
    # Combining Both the images, to be placed adjacent to each other
    Siftimage=np.zeros((s1[0],s1[1]+s2[1],3)) 
    Siftimage[:,0:s1[1]]=image1
    Siftimage[:,s1[1]:s1[1]+s2[1]]=image2
    # Drawing Lines for SIFT points matched
    for i in range(0,len(sift_pts1)):
        cv.line(Siftimage,(int(sift_pts1[i][1]),int(sift_pts1[i][0])),(s1[1]+int(sift_pts2[i][1]),int(sift_pts2[i][0])),(0,0,255),thickness=2)

    return Siftimage

# Matching the corresponding interest points, between the two images
def match_sift(image1,Sift_Corners1,des1,image2,Sift_Corners2,des2):
    
    l1=len(Sift_Corners1)
    l2=len(Sift_Corners2)
    euclidian=np.zeros((l1,l2)) # Similarity metric for feature vectors
    
    for i in range(0,l1):
        for j in range(0,l2): # Normalizing Lengths to make it invariant to change in illumniation
            euclidian[i][j]=np.linalg.norm(des1[i]/np.linalg.norm(des1[i])-des2[j]/np.linalg.norm(des2[j]))
    # Computing the Euclidean distance between the 128 bit feature vectors        
    sift_threshold=np.min(euclidian)*(2)
    sift_pts1=[]
    sift_pts2=[]
    # Finding similar interest points and mapping them with each other
    for i in range(0,len(euclidian)):
        val=np.min(euclidian[i])    # Similar to the SSD for Harris corner detector
        idx=np.argmin(euclidian[i])
        if(val<=sift_threshold):
            sift_pts1.append([Sift_Corners1[i][0],Sift_Corners1[i][1]])
            sift_pts2.append([Sift_Corners2[idx][0],Sift_Corners2[idx][1]])    
    # Adding similar and corresponding interest points to the list        
    
    return sift_pts1,sift_pts2

# Using the built in SIFT function from cv library
sift = cv.xfeatures2d.SIFT_create()

# Reading the two images
image1clr=cv.imread("1.jpg") 
image2clr=cv.imread("2.jpg")
# Used for reducing the size of truck images, to improve computation speed.
#image1clr=cv.resize(image1clr, (0,0), fx=0.50, fy=0.50)
#image2clr=cv.resize(image2clr, (0,0), fx=0.50, fy=0.50)
# Converting from color to grayscale
image1=cv.cvtColor(image1clr,cv.COLOR_RGB2GRAY)
image2=cv.cvtColor(image2clr,cv.COLOR_RGB2GRAY)
# Finding the interest points and the descriptor vectors for every interest point in the image
kp1,des1 = sift.detectAndCompute(image1,None)
kp2,des2 = sift.detectAndCompute(image2,None)

Sift_Corners1=[]
Sift_Corners2=[]
# Extracting the interest points for both images
for i in kp1:
    Sift_Corners1.append([i.pt[1],i.pt[0]])
    
for i in kp2:
    Sift_Corners2.append([i.pt[1],i.pt[0]])
    
# Finding the similarity metric and mapping similar interest points
sift_pts1,sift_pts2=match_sift(image1,Sift_Corners1,des1,image2,Sift_Corners2,des2)    
# Drawing Lines between the corresponding interest points in the image
Siftimage=draw_lines(image1clr,sift_pts1,image2clr,sift_pts2)  

cv.imwrite('SIFTimage.jpg',Siftimage)
# Writing the Output corresponding interest points in both images, with lines drawn





