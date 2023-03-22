import myYolov7
import subprocess
from subprocess import Popen
import torch
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
model = myYolov7.my_yolov7('last.pt','cpu',0.6)

def predict_YOLOv7_proces(Name):
    
    count = 0
    for filename in os.listdir('test_image/' + Name):
        count +=1
        imgs = 'test_image/' + Name+'/' +filename 
        if imgs.split('.')[1] == 'jpg':
            savepath = 'output_img/image.jpg'
        # else:
        #     savepath = 'output_img/video.mp4'
        result, det =  model.detect(imgs,savepath,count)
        # for xyxy in reversed(det):                   
        #     print( int( xyxy[0]), int(xyxy[1]),int(xyxy[2]),int(xyxy[3]))
           
predict_YOLOv7_proces(Name='Hoang')