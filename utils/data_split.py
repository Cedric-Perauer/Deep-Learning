import os 
import numpy as np 
import random 
from shutil import copyfile
import  shutil 

base_path = os.path.expanduser("~/Datasets/flower_photos")

train_path = base_path + '/train/'
test_path = base_path + '/test/'
valid_path = base_path + '/valid/'

data_dirs = [test_path, valid_path,train_path]

classes = ['sunflowers','daisy','dandelion','roses','tulips']
dirs = os.listdir(base_path)
train_pct = 0.8 

for dir in dirs :
    if dir in classes : 
        print(dir) 
        print('-------------') 
        f = os.listdir(base_path + '/' +  dir + '/')
        random.shuffle(f)
        len_train = int(len(f) * 0.8) 
        len_test_val = len(f) - len_train 
        len_test = int(0.5 * len_test_val) 
        len_val = len_test_val - len_test
        train_files = f[:len_train]
        test_files = f[len_train: len_train + len_test]
        valid_files = f[len_train + len_test : ]
        for d in data_dirs :
            if not os.path.exists(d + dir) : 
                os.mkdir(d + dir)
        for f in train_files : 
            shutil.copy(base_path + "/" + dir + "/" + f,train_path + '/' +  dir + '/')      
        for f in test_files : 
            shutil.copy(base_path + "/" + dir + "/" + f,test_path + '/' +  dir + '/')
        for f in valid_files : 
            shutil.copy(base_path + "/" + dir + "/" + f,valid_path + '/' +  dir + '/')
