import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
# Importing neccessary libraries for HW4

# The function below is used for drawing lines, between points.
# The line is drawn between the two images, where corresponding points are matched.

def draw_lines(image1,SSDpts1,NCCpts1,image2,SSDpts2,NCCpts2):
    
    s1=np.shape(image1)
    s2=np.shape(image2)
    # Combining Both the images, to be placed adjacent to each other
    SSDimage=np.zeros((s1[0],s1[1]+s2[1],3))
    SSDimage[:,0:s1[1]]=image1
    SSDimage[:,s1[1]:s1[1]+s2[1]]=image2
    
    NCCimage=np.zeros((s1[0],s1[1]+s2[1],3))
    NCCimage[:,0:s1[1]]=image1
    NCCimage[:,s1[1]:s1[1]+s2[1]]=image2
    
    for i in range(0,len(SSDpts1)): # Drawing Lines for SSD points matched
        cv.line(SSDimage,(SSDpts1[i][1],SSDpts1[i][0]),(s1[1]+SSDpts2[i][1],SSDpts2[i][0]),(0,0,255),thickness=2)
        
    for i in range(0,len(NCCpts1)): # Drawing Lines for NCC points matched
        cv.line(NCCimage,(NCCpts1[i][1],NCCpts1[i][0]),(s1[1]+NCCpts2[i][1],NCCpts2[i][0]),(0,255,0),thickness=2)
    
    return SSDimage,NCCimage

# Matching the corresponding interest points, between the two images
def match_harris_corners(image1,Harris_Corners1,image2,Harris_Corners2):
    
    N_Wsize=21  # Window size used for computing the SSD and NCC distance metrics
    
    l1=len(Harris_Corners1)
    l2=len(Harris_Corners2)
    
    SSD=np.zeros((l1,l2)) # SSD matrix
    NCC=np.zeros((l1,l2)) # NCC matrix
    
    SSD_pts1=[]     # Lists to the store the points which have most similarity
    SSD_pts2=[]
    NCC_pts1=[]
    NCC_pts2=[]
    
    for i in range(0,l1): # Extracting Patches around every interest point
        W1=image1[-(N_Wsize/2)+Harris_Corners1[i][0]:(N_Wsize/2)+Harris_Corners1[i][0]+1,-(N_Wsize/2)+Harris_Corners1[i][1]:(N_Wsize/2)+Harris_Corners1[i][1]+1]
        m1=np.mean(W1)  
        s1=W1-m1
        for j in range(0,l2):
            W2=image2[-(N_Wsize/2)+Harris_Corners2[j][0]:(N_Wsize/2)+Harris_Corners2[j][0]+1,-(N_Wsize/2)+Harris_Corners2[j][1]:(N_Wsize/2)+Harris_Corners2[j][1]+1]
            m2=np.mean(W2)  # computing mean value
            s2=W2-m2
            SSD[i][j]=np.sum((W1-W2)*(W1-W2)) # Computing SSD metric
            if(np.sum(s1*s2)!=0):
                NCC[i][j]=np.sum(s1*s2)/(np.sqrt(np.sum(s1*s1)*np.sum(s2*s2)))
            # Computing NCC metric
            
    SSD_threshold=np.min(SSD)*10    # Threshold for SSD
    NCC_threshold=0.95            # Threshold for NCC
    # The thresholds were modified dynamically and statically for different image pairs and scales
    # This was done to increase number of interest point correspondences 
    list1=[]
    list2=[]
    
    for i in range(0,len(SSD)):
        val=np.min(SSD[i])
        idx=np.argmin(SSD[i])  # Computing the similarity metric against the threshold
        if((val<=SSD_threshold)and(idx not in list1)):
            list1.append(idx)  # Appending the points with most similarity for SSD
            SSD_pts1.append([Harris_Corners1[i][0],Harris_Corners1[i][1]])
            SSD_pts2.append([Harris_Corners2[idx][0],Harris_Corners2[idx][1]])
        
    for i in range(0,len(NCC)):
        val=np.max(NCC[i])
        idx=np.argmax(NCC[i]) # Computing the similarity metric against the threshold
        if((val>=NCC_threshold)and(idx not in list2)):
            list2.append(idx) # Appending the points with most similarity for NCC
            NCC_pts1.append([Harris_Corners1[i][0],Harris_Corners1[i][1]])
            NCC_pts2.append([Harris_Corners2[idx][0],Harris_Corners2[idx][1]])


    return SSD_pts1,SSD_pts2,NCC_pts1,NCC_pts2

# Computing the Harris Corner Points
def harris_corner(image,sigma):  

    N_Haar =int(round(np.ceil(4*sigma/2))*2)  
    # Haar filter N value to be chosen as even number based on sigma value
    HaarX=np.zeros((N_Haar,N_Haar))
    HaarY=np.zeros((N_Haar,N_Haar))
    
    HaarX[:,0:N_Haar/2]=-1     # Initializing the HaarX filters
    HaarX[:,N_Haar/2:N_Haar]=1
    
    HaarY[0:N_Haar/2]=1      # Initializing the HaarY filters
    HaarY[N_Haar/2:N_Haar]=-1
    # Applying the Haar Filters to the image
    Gx=cv.filter2D(image,-1,HaarX).astype('float')
    Gy=cv.filter2D(image,-1,HaarY).astype('float')
    
    Gx2=Gx*Gx   # Computing the gradients in different directions
    Gy2=Gy*Gy
    Gxy=Gx*Gy
    
    N_HC=int(round(np.ceil(5*sigma/2))*2)+1
    # Computing the N patch size of computing C matrix to identify corner points
    HC_dr=np.zeros(np.shape(image))
    count=0
    for i in range(N_HC/2,np.shape(image)[0]-N_HC/2):
        for j in range(N_HC/2,np.shape(image)[1]-N_HC/2):
            c11=np.sum(Gx2[-(N_HC/2)+i:(N_HC/2)+i+1,-(N_HC/2)+j:(N_HC/2)+j+1])
            c22=np.sum(Gy2[-(N_HC/2)+i:(N_HC/2)+i+1,-(N_HC/2)+j:(N_HC/2)+j+1])
            c12=np.sum(Gxy[-(N_HC/2)+i:(N_HC/2)+i+1,-(N_HC/2)+j:(N_HC/2)+j+1])
            # Computing the C Matrix, and identifying the threshold for eigen values    
            C=np.array([[c11,c12],[c12,c22]])
            if(np.trace(C)!=0): 
                HC_dr[i][j]=np.linalg.det(C)/(np.trace(C)*np.trace(C))
            # Computing the Harris corner detector responses 
    N_WinMax=29  # Window Size for computing the maxima, and finalizing the interest points in an image
    Harris_Corners=[]
    # List for storing the interest points
    for i in range(N_WinMax/2,np.shape(image)[0]-N_WinMax/2):
        for j in range(N_WinMax/2,np.shape(image)[1]-N_WinMax/2):
            R1=HC_dr[-(N_WinMax/2)+i:(N_WinMax/2)+i+1,-(N_WinMax/2)+j:(N_WinMax/2)+j+1]
            H_Max=np.max(R1)
            #  Finding the maxima, based on the Window Patch size used
            if((HC_dr[i][j]==H_Max)and(HC_dr[i][j]>np.mean(HC_dr))):
                Harris_Corners.append([i,j])
            # Computing the corner with the threshold of mean value in the matrix
    return Harris_Corners       
    


# Reading the two images

image1clr=cv.imread("1.jpg") 
image2clr=cv.imread("2.jpg")
# Used for reducing the size of truck images, to improve computation speed.
#image1clr=cv.resize(image1clr, (0,0), fx=0.25, fy=0.25) 
#image2clr=cv.resize(image2clr, (0,0), fx=0.25, fy=0.25)
# Converting from color to grayscale
image1=cv.cvtColor(image1clr,cv.COLOR_RGB2GRAY).astype('float')
image2=cv.cvtColor(image2clr,cv.COLOR_RGB2GRAY).astype('float')
# Sigma array of values, we can add or delete any number of values we want in the scale space.
# Feel free to add more values, to compute and map interest points at that scale.
sigma=[1,2,3,4]

for i in range(0,4): # Computing the interest points and mapping for every scale
    Harris_Corners1=harris_corner(image1,sigma[i]) # COMPUTING the Harris corner detector responses
    Harris_Corners2=harris_corner(image2,sigma[i])
    # Computing the Distance metrics for SSD and NCC to establish similarity among the interest points
    SSD_pts1,SSD_pts2,NCC_pts1,NCC_pts2=match_harris_corners(image1,Harris_Corners1,image2,Harris_Corners2)   
    # Drawing Lines between the corresponding interest points in the image
    SSDimage,NCCimage=draw_lines(image1clr,SSD_pts1,NCC_pts1,image2clr,SSD_pts2,NCC_pts2)  
    
    cv.imwrite('SSDimage'+str(sigma[i])+'.jpg',SSDimage)
    cv.imwrite('NCCimage'+str(sigma[i])+'.jpg',NCCimage)
    # Writing the Output corresponding interest points in both images, with lines drawn
    






