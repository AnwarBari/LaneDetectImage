import cv2 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import requests
 

def draw_the_line(img,lines): 
    img= np.copy(img)
    blank_image= np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    for line in lines: 
        for x1, y1, x2, y2 in line : 
            cv2.line(blank_image, (x1,y1), (x2,y2), (0,255,0), 3)
    img=cv2.addWeighted(img,0.8,blank_image,1,0.0)
    return img 

def region_of_interest(img, vertices): 
    mask = np.zeros_like(img)
    
    match_mask_color= 255
    cv2.fillPoly(mask,vertices,match_mask_color)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image



#======================== step 1 ==============================
image = cv2.imread('C:\\LaneDetection\\road.jpeg',1)
image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

print (image.shape)
height = image.shape[0] 
width = image.shape[1] 

plt.imshow(image)
plt.show()

#======================== step 2 ==============================
gray_image= cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
canny_image = cv2.Canny(gray_image,100,200)
plt.imshow(canny_image)
plt.show()
#======================== step 3 ==============================
region_of_interest_vertices = [
    (0,height), 
    (width/2, height/2),
    (width,height)
 ]
cropped_image = region_of_interest(canny_image,np.array([region_of_interest_vertices],np.int32))
plt.imshow(cropped_image)
plt.show()
#======================== steps 4,5 ==============================

lines = cv2.HoughLinesP(cropped_image,
                         rho=6,         # distance resolution of the accumolator in pixels 
                         theta=np.pi/60, # angle resolution of the accumolator in radians
                         threshold=160,  # only those lines are returned that got enough vates > threshould 
                         lines=np.array([]),
                         minLineLength=40,
                         maxLineGap=25)

#======================== step 6 ==============================
image_with_lines = draw_the_line(image,lines)


plt.imshow(image_with_lines)
plt.show()
