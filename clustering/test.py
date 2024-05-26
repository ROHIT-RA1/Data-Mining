import cv2 , os , numpy , pathlib
import matplotlib.pyplot as plt
from genericpath import exists
from PIL import Image 
import pandas as pd
from clustering import *
from sklearn.cluster import DBSCAN
from collections import defaultdict

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
        cv2.rectangle(Orig_image,cordinates, (cordinates[0]+width,cordinates[1] + height), (0, 0, 0), 2)
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
 

# new    
for x in range(1,count+1):
    imagepath_x = 'testPatient/Slices/OUT_{}'.format(x)
    cluster_imagepath_x = 'testPatient/Clusters/OUT_{}'.format(x)
    cntr=1
    for filename in os.listdir(imagepath_x):
        tempClusteredImage = cv2.imread(imagepath_x + "/" + filename)
        width = tempClusteredImage.shape[1]
        height = tempClusteredImage.shape[0]
        for h in range(0,height-1):
            for w in range(0,width-1):
                imageCoordinates = tempClusteredImage[h,w]
                if(imageCoordinates[0]==imageCoordinates[1]==imageCoordinates[2]):
                    imageCoordinates[0]=0
                    imageCoordinates[1]=0
                    imageCoordinates[2]=0
                else:
                    imageCoordinates[0]=255
                    imageCoordinates[1]=0
                    imageCoordinates[2]=255

        finalImage = Image.fromarray(tempClusteredImage)
        finalImage.save(cluster_imagepath_x+'/cluster_{}.png'.format(cntr))
        cntr = cntr + 1


for x in range(1,count+1):
    cluster_imagepath_x = 'testPatient/Clusters/OUT_{}'.format(x)
    dataframe = pd.DataFrame()
    slice_num =[]
    cluster_cnt=[]
    csv_cnt_lnt = []
    cntr = 1
    for filename in os.listdir(cluster_imagepath_x):
        tempClusteredImage = cv2.imread(cluster_imagepath_x + "/" + filename)
        width = tempClusteredImage.shape[1]
        height = tempClusteredImage.shape[0]
        non_blk_pnts = []
            
        for x in range(height):
            for y in range(width):
                if tempClusteredImage[x,y].any()>0:
                    non_blk_pnts.append(list([x,y]))
        if(len(non_blk_pnts)!=0):
            dbsc = DBSCAN(eps=2, min_samples=5).fit(non_blk_pnts)
            labels = dbsc.labels_
            unique_lbls = list(set(labels))
            dict_fnl = defaultdict(list)
            for a, b in zip(labels, non_blk_pnts):
                dict_fnl[a].append(b)
            cntCsvCount = 0
            for dictSize in range(len(dict_fnl)):
                csvCount=0

                for point in unique_lbls:
                    h,w = dict_fnl[point][0]
                    csvCount = csvCount+len(dict_fnl[point])
                if csvCount > 135:
                    cntCsvCount = cntCsvCount + 1
                csv_cnt_lnt.append(cntCsvCount)
            
            slice_num.append(cntr)
            cluster_cnt.append(cntCsvCount)
        else:
            slice_num.append(cntr)
            cluster_cnt.append(0)    
        non_blk_pnts = []
        cntr=cntr+1
        
    raw_data = {'slice_num': slice_num, 'cluster_cnt':cluster_cnt}
    dataframe = pd.DataFrame(raw_data, columns = ['slice_num', 'cluster_cnt'])
    dataframe.to_csv(cluster_imagepath_x + '/count.csv', index=False)
 