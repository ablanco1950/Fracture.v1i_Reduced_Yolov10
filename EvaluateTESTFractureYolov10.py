# -*- coding: utf-8 -*-
"""
Created on Jun 2024

@author: Alfonso Blanco
"""
#######################################################################
# PARAMETERS
######################################################################
dir=""
dirname= "testFractureOJumbo1\\images"
dirnameLabels="testFractureOJumbo1\\labels"


#dirnameYolo ="C:\\Fracture.v1i_Reduced_Yolov10\\runs\\train\\exp4\\weights\\best.pt"

#dirnameYolo ="C:\\Fracture.v1i_Reduced_Yolov10\\runs\\train\\exp4\\weights\\last.pt"

dirnameYolo="last114epoch0603.pt"

import cv2
import time
Ini=time.time()

""" This gives error after upgrading ultralytics
from ultralytics import YOLOv10
model = YOLOv10(dirnameYolo)
"""
# change for
from ultralytics import YOLO
model = YOLO(dirnameYolo)

class_list = model.model.names
print(class_list)

import numpy as np

import os
import re

import imutils

########################################################################
def loadimages(dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     TabFileName=[]
   
    
     print("Reading imagenes from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
        
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                
                 
                 image = cv2.imread(filepath)
                 #print(filepath)
                 #print(image.shape)                           
                 images.append(image)
                 TabFileName.append(filename)
                 
                 Cont+=1
     
     return images, TabFileName
########################################################################
def loadlabels(dirnameLabels):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirnameLabels + "\\"
     
     Labels = []
     TabFileLabelsName=[]
     Tabxyxy=[]
     ContLabels=0
     ContNoLabels=0
         
     print("Reading labels from ",imgpath)
        
     for root, dirnames, filenames in os.walk(imgpath):
         
         for filename in filenames:
                           
                 filepath = os.path.join(root, filename)
                
                 f=open(filepath,"r")

                 Label=""
                 xyxy=""
                 for linea in f:
                      
                      indexFracture=int(linea[0])
                      Label=class_list[indexFracture]
                      xyxy=linea[2:]
                      
                                            
                 Labels.append(Label)
                 
                 if Label=="":
                      ContLabels+=1
                 else:
                     ContNoLabels+=1 
                 
                 TabFileLabelsName.append(filename)
                 Tabxyxy.append(xyxy)
     return Labels, TabFileLabelsName, Tabxyxy, ContLabels, ContNoLabels

def unconvert(width, height, x, y, w, h):

    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)

    return xmin, ymin, xmax, ymax

# ttps://medium.chom/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c
def DetectBoneFractureWithYolov10 (img):
  
   TabcropBoneFracture=[]
   
   y=[]
   yMax=[]
   x=[]
   xMax=[]
   Tabclass_name=[]
   
   # https://blog.roboflow.com/yolov10-how-to-train/
   results = model(source=img)
   for i in range(len(results)):
       # may be several plates in a frame
       result=results[i]
       
       xyxy= result.boxes.xyxy.numpy()
       confidence= result.boxes.conf.numpy()
       class_id= result.boxes.cls.numpy().astype(int)
       #print(class_id)
       out_image = img.copy()
       for j in range(len(class_id)):
           con=confidence[j]
           label=class_list[class_id[j]] + " " + str(con)
           box=xyxy[j]
           
           cropBoneFracture=out_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
           
           TabcropBoneFracture.append(cropBoneFracture)
           y.append(int(box[1]))
           yMax.append(int(box[3]))
           x.append(int(box[0]))
           xMax.append(int(box[2]))

           # Tabclass_name only contains confidence, there is only a class and the name is not interesting
           Tabclass_name.append(label)
            
      
   return TabcropBoneFracture, y,yMax,x,xMax, Tabclass_name


###########################################################
# MAIN
##########################################################

Labels, TabFileLabelsName, TabxyxyTrue, ContLabels, ContNoLabels= loadlabels(dirnameLabels)

#print("Number of images to test : " + str(len(Labels)))

#print("Number of files without labels : " + str(ContNoLabels))
#print("Number of files with labels : " + str(ContLabels))


imagesComplete, TabFileName=loadimages(dirname)

print("Number of images to test: " + str(len(imagesComplete)))

ContError=0
ContHit=0
ContNoDetected=0

for i in range (len(imagesComplete)):
 
            if TabFileLabelsName[i][:len(TabFileLabelsName[i])-4] != TabFileName[i][:len(TabFileName[i])-4]:
                 print("ERROR SEQUENCING IMAGES AN LABELS " + TabFileLabelsName[i][:len(TabFileLabelsName[i])-4] +" --" + TabFileName[i][:len(TabFileName[i])-4])
                 break
            # no se consideran las que no vienen labeladas
            if Labels[i] == "": continue
            gray=imagesComplete[i]
           
            imgTrue=imagesComplete[i]
            
            xyxyTrue=TabxyxyTrue[i].split(" ")
            yTrue=float(xyxyTrue[1])* float(imgTrue.shape[0])
            yMaxTrue=float(xyxyTrue[3])* float(imgTrue.shape[0])
            xTrue=float(xyxyTrue[0])* float(imgTrue.shape[1])
            xMaxTrue=float(xyxyTrue[2])* float(imgTrue.shape[1])
            start_pointTrue=(int(xTrue),int(yTrue)) 
            end_pointTrue=(int(xMaxTrue),int( yMaxTrue))
                      
            # Put text
            text_locationTrue = (int(xMaxTrue),int(yMaxTrue))
            text_colorTrue = (255,255,255)
           
            XcenterYcenterWH=TabxyxyTrue[i].split(" ")
            width=float(imgTrue.shape[0])
            height=float(imgTrue.shape[1])
            x=float(XcenterYcenterWH[0])
            y=float(XcenterYcenterWH[1])
            w=float(XcenterYcenterWH[2])
            h=float(XcenterYcenterWH[3])
            xTrue,yTrue,xMaxTrue,yMaxTrue=unconvert(width, height, x, y, w, h)
           
            start_pointTrue=(int(xTrue),int(yTrue)) 
            end_pointTrue=(int(xMaxTrue),int( yMaxTrue))
           
            colorTrue=(0,0,255)
            # Using cv2.rectangle() method
            # Draw a rectangle with green line borders of thickness of 2 px
            imgTrue = cv2.rectangle(imgTrue, start_pointTrue, end_pointTrue,(0,255,0), 2)
           
            # Put text
            text_locationTrue = (int(xMaxTrue),int(yMaxTrue))
            text_colorTrue = (255,255,255)
            #cv2.putText(imgTrue, Labels[i] ,text_locationTrue
            #            , cv2.FONT_HERSHEY_SIMPLEX , 1
            #            , text_colorTrue, 2 ,cv2.LINE_AA)
            cv2.putText(imgTrue, "" ,text_locationTrue
                        , cv2.FONT_HERSHEY_SIMPLEX , 1
                        , text_colorTrue, 2 ,cv2.LINE_AA)

            #cv2.imshow('True', imgTrue)
            #cv2.waitKey(0)

            #"""
            TabImgSelect, y, yMax, x, xMax, Tabclass_name =DetectBoneFractureWithYolov10(gray)
            #print(gray.shape)
            if TabImgSelect==[]:
                print(TabFileName[i] + " NON DETECTED")
                ContNoDetected=ContNoDetected+1 
                continue
            else:
                #ContDetected=ContDetected+1
                print(TabFileName[i] + " DETECTED ")
                
                
            for z in range(len(TabImgSelect)):
                #if TabImgSelect[z] == []: continue
                gray1=TabImgSelect[z]
                #cv2.waitKey(0)
                start_point=(x[z],y[z]) 
                end_point=(xMax[z], yMax[z])
                color=(255,0,0)
                # Using cv2.rectangle() method
                # Draw a rectangle with blue line borders of thickness of 2 px
                img = cv2.rectangle(gray, start_point, end_point,(255,0,0), 2)
                # Put text
                text_location = (x[z], y[z])
                text_color = (255,255,255)
                if Tabclass_name[z][:len(Labels[i])] !=Labels[i]:
                     #print(len(Tabclass_name[z]))
                     #print(len(Labels[i]))
                     print("ERROR " + TabFileName[i] + "Predicted "+ Tabclass_name[z] + " true is " + Labels[i])
                     ContError+=1
                else:
                     #print("HIT " + TabFileName[i] + "Predicted "+ Tabclass_name[z] )
                     ContHit+=1
                cv2.putText(img, str(Tabclass_name[z][len(Labels[i]):]) ,text_location
                        , cv2.FONT_HERSHEY_SIMPLEX , 1
                        , text_color, 2 ,cv2.LINE_AA)
                cv2.putText(gray1, str(Tabclass_name[z][len(Labels[i]):]) ,text_location
                        , cv2.FONT_HERSHEY_SIMPLEX , 1
                        , text_color, 2 ,cv2.LINE_AA)
                        
                #cv2.imshow('Bone Fracture', gray1)
                #cv2.waitKey(0)
                break
            #      
            #show_image=cv2.resize(img,(1000,700))
            #cv2.imshow('Frame', show_image)
            cv2.imshow('Frame', img)
            cv2.waitKey(0)
           
             
              
print("")           
print("NO detected=" + str(ContNoDetected))
#print("Errors=" + str(ContError))
#print("Hits=" + str(ContHit))
print("")      
print( " Time in seconds "+ str(time.time()-Ini))
