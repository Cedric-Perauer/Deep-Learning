import pandas as pd 
import numpy as np 
from shutil import copyfile
import os 
import cv2
import csv

d_path = "fsd_dataset/images/"
l_path = "fsd_dataset/labels/"
img_goal_dir = "dataset/FSOCO_Transformed/"
ref_csv = "dataset/test.csv"
csvfile = pd.read_csv(ref_csv) 
train_labels = l_path + "train/"
test_labels = l_path + "test/"
val_labels = l_path + "val/"

def mv_imgs(path,goal_dir):
    for subdir,dirs, files in os.walk(path):
        for f in files :
            src = subdir + "/" + f 
            dst = goal_dir + f 
            copyfile(src,dst)

def label_converter(label_dir):
    df = pd.DataFrame(columns=csvfile.columns)
    rows = csvfile.iloc[0,:]
    df = df.append(rows) 
    for file in os.listdir(label_dir):
            f = open(label_dir + file)
            content = f.readlines()
            img = cv2.imread(img_goal_dir + file.split(".")[0]+".jpg")
            height, width, _ = img.shape
            scale = 1
            labels_line = file.split(".")[0]+".jpg" + "," + "N/A," + str(width) + "," + str(height) + "," + str(scale)
            row_dict = {n:" " for n in df.columns}
            row_dict['please see k-means anchor boxes in train.csv'] = file.split(".")[0]+".jpg"
            row_dict['Unnamed: 1'] = 'N/A'
            row_dict['Unnamed: 2'] = str(width)
            row_dict['Unnamed: 3'] = str(height)
            row_dict['Unnamed: 4'] = str(1)
            for idx,line in enumerate(content) : 
               line_arr = line.split(" ") 
               x = float(line_arr[1]) * width 
               y = float(line_arr[2]) * height
               cx = float(line_arr[3]) * width 
               cy = float(line_arr[4]) * height
               label_arr =  str([int(round(x)),int(round(y)),int(round(cx)),int(round(cy))]) 
               row_dict["Unnamed: " + str(idx+5) ] = label_arr
            df = df.append(row_dict,ignore_index=True)
    if label_dir.split("/")[2] == "test": 
          df.to_csv(r'test.csv', index = False)
    
    elif label_dir.split("/")[2] == "train":
          df.to_csv(r'train.csv', index = False)
    else : 
          df.to_csv(r'test.csv', index = False)
          #f_csv.writelines(labels_line)
#mv_imgs(d_path,img_goal_dir)
label_converter(train_labels)
label_converter(val_labels)
label_converter(test_labels)




