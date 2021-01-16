import pandas as pd
import numpy as np 
from shutil import copyfile


train_csv = "train.csv"
test_csv = "test.csv"
val_csv = "validate.csv"

val_set = pd.read_csv(val_csv)
test_set = pd.read_csv(test_csv)
train_set = pd.read_csv(train_csv)

img_location = "YOLO_Dataset/"

def iter_df(df,df_name,write_txt=True):
       txt_dir = "MIT_Dataset/labels/test"
       img_dir = "MIT_Dataset/img/test"
       if df_name == "validate.csv":
           txt_dir = "MIT_Dataset/labels/val"
           img_dir = "MIT_Dataset/img/val"
       elif df_name == "train.csv":
           txt_dir = "MIT_Dataset/labels/train"
           img_dir = "MIT_Dataset/img/train"
       for idx, row in df.iterrows(): 
            if idx == 0: 
               continue
            name = row[0]
            img_w = int(row["Unnamed: 2"])
            img_h = int(row["Unnamed: 3"])
             
            #get boxes all single class 
            i = 5
            label_f = open(txt_dir + "/" + name.split(".")[0] + ".txt","w") 
            #copy image file
            src = img_location + name
            dst = img_dir + "/" +    name
            copyfile(src,dst)
            for i in range(5,100):
                try: 
                     arr_split = row[i][1:-1].split(",")
                except: 
                    break
                x = float(arr_split[0]) / img_w
                y = float(arr_split[1]) / img_h
                w = float(arr_split[2]) / img_w
                h = float(arr_split[3]) / img_h
                label_f.write("{} {} {} {} {}\r".format(0,x,y,w,h)) 
             
iter_df(val_set,val_csv)
iter_df(test_set,test_csv)
iter_df(train_set,train_csv)
