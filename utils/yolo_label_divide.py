import numpy as np 
import os 
import random
from sklearn.model_selection import train_test_split



def split_sets(dir,split): 
    test_train_split = split
    all_files = []
    for f in os.listdir(dir): 
            filename = os.fsdecode(f)
            all_files.append(filename)

    print(len(all_files))
    train_files, test_files = train_test_split(all_files,test_size=split)
    val_files, test_files = train_test_split(test_files,test_size=0.5)

    
    train = open("train.txt","w+")

    for f in train_files : 
            train.write("/home/cedric/yolov3/data/images/"+ str(f) + "\n")

    val = open("val.txt","w+")
    
    for f in val_files : 
            val.write("/home/cedric/yolov3/data/images/"+ str(f) + "\n")
    
    test = open("test.txt","w+")
    for f in test_files : 
            test.write("/home/cedric/yolov3/data/images/"+ str(f) + "\n")



    

dir = "/home/cedric/yolov3/data/images/"
split = 0.2
split_sets(dir,split)
