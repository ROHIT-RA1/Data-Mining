import cv2 , os , numpy , pathlib
import matplotlib.pyplot as plt
from genericpath import exists
from PIL import Image 
from brain_extraction import *


folderCreation()
SubFolderCreation()

dir_path = 'testPatient/Data'
count = countReqFiles(dir_path)

# Image slicing Using templatematching 
for x in range(1,count+1):
    imagepath_x = 'testPatient/Slices/OUT_{}'.format(x)
    Orig_image = cv2.imread('testPatient/Data/IC_{}_thresh.png'.format(x))
    Grayscale_image = cv2.imread('testPatient/Data/IC_{}_thresh.png'.format(x),0)
    # getting the template to use for matching
    template = cv2.imread('Template.jpeg',0)

    width = template.shape[1]
    height = template.shape[0]
    # Template Matching
    result = cv2.matchTemplate(Grayscale_image,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    location = numpy.where(result >=threshold)

    for cordinates in zip(*location[::-1]):
        cv2.rectangle(Orig_image,cordinates, (cordinates[0]+width,cordinates[1] + height), (0, 255, 255), 2)
    location_array = list(zip(*location[::-1]))
    crop_dimensions = location_array[1][0]-location_array[0][0]

    cntr = 1
    for cordinates in zip(*location[::-1]):
        sliced_img = Orig_image[cordinates[1]:cordinates[1]+crop_dimensions, cordinates[0]: cordinates[0]+crop_dimensions]
        img = Image.fromarray(sliced_img)
        img.save(imagepath_x + "/slice_{}.png".format(cntr))
        testimg = cv2.imread(imagepath_x + "/slice_{}.png".format(cntr),0)
        BlackPxlCount = numpy.count_nonzero([testimg<=76])
        if(BlackPxlCount>=13760):
            os.remove(imagepath_x+"/slice_{}.png".format(cntr))
        cntr= cntr+1
 

# finding Boundary edge of the brain images 
for y in range(1,113):
    finalpath_y = 'testPatient/Boundaries/OUT_{}'.format(y)
    sliced_path = 'testPatient/Slices/OUT_{}'.format(y)

    for filename in os.listdir(sliced_path):
        slice_img = cv2.imread(sliced_path + "/" + filename)
        if(type(slice_img) is numpy.ndarray) :
          b = slice_img[:,:,0]
          g = slice_img[:,:,1]
          r = slice_img[:,:,2]

          contour1 , hierarchy1 = cv2.findContours(b,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
          image_contour_b = slice_img.copy()
          cv2.drawContours(image_contour_b, contour1, -1, (0, 255, 0), 2, cv2.LINE_AA)
          cv2.destroyAllWindows()

          contour2 , hierarchy2 = cv2.findContours(g,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
          image_contour_g = slice_img.copy()
          cv2.drawContours(image_contour_g, contour1, -1, (0, 255, 0), 2, cv2.LINE_AA)
          cv2.destroyAllWindows()

          contour3 , hierarchy3 = cv2.findContours(r,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
          image_contour_r = slice_img.copy()
          cv2.drawContours(image_contour_r, contour1, -1, (0, 255, 0), 2, cv2.LINE_AA)
          cv2.destroyAllWindows()

          final_image = Image.fromarray(image_contour_r)
          
          final_image.save(finalpath_y+ "/Boundary_{}".format(filename.split('_')[1]))
       

 