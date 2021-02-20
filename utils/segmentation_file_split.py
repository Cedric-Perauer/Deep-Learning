import os.path as oss 
import json
import os
import cv2
import random 
import shutil 

base_dir = oss.expanduser("~")
fsd_dataset = oss.expanduser("~/fsoco_raw/data")
dataset_dir = oss.expanduser("~/fsoco_raw")
label_text_dir = oss.expanduser("~/fsoco_raw/data")
dataset_dir = oss.expanduser("~/fsoco_raw/data/label")
img_img_dir = oss.expanduser("~/fsoco_raw/data/img")
label_img_dir = oss.expanduser("~/fsoco_raw/data/label")


mode = 1 
watermark_h = 140 
labels = []
num = 489 #number of images 
num_train = int(0.7 * num)
num_test = int(0.15 * num) 
num_val = num - num_train - num_test
print(num_val + num_test + num_train) 
images = [] 

def image_conversion(img_dir,label_dir): 
    """
    for img in images :
        name_arr = img.split("/") 
        image = cv2.imread(img)
        h,w,c = image.shape
        print(img_dir + "/" + name_arr[-1]) 
        crop_img = image[watermark_h:h-watermark_h,watermark_h:w-watermark_h,:]
        cv2.imwrite(img_dir + "/" + name_arr[-1], crop_img )
        #add labels 
    """
    for label in labels :
        print(label)
        shutil.copy(label, label_dir) 


def collect_data(data_dir):
    cnt = 0 
    for subdir, dirs, files in os.walk("/Users/cedricperauer/fsoco_raw"):
        for file in files:
            print(file)
            if file[-4:] == "json" and file != "meta.json": 
                f_name = subdir + "/" + file
                labels.append(f_name)
                cnt += 1 
            elif file[-3:] == "png" or file[-3:] == "jpg" : 
                f_name = subdir + "/" + file
                images.append(f_name)
    

def split_set(txt_dir,img_dir,base_dir): 
     
    #dirs for the labels 
    train_dir_img =  base_dir + "/img/train/" 
    test_dir_img =  base_dir + "/img/test/" 
    val_dir_img =  base_dir + "/img/val/" 
    
    train_dir_txt =  base_dir + "/labels/train/" 
    test_dir_txt =  base_dir + "/labels/test/" 
    val_dir_txt =  base_dir + "/labels/val/" 
    all = os.listdir(img_dir)
    random.shuffle(all) 
    print(len(all))  
    
    for i in range(0,num_train): 
        print(all[i]) 
        f = all[i][:-4]
       
        shutil.move(img_dir + "/" + all[i],train_dir_img  )   
        shutil.move(txt_dir + "/" + f  + ".json",train_dir_img)   
      
    for i in range(num_train,num_train+num_test): 
        
        f = all[i][:-4]
        
        shutil.move(img_dir + "/" + all[i],train_dir_img)   
        shutil.move(txt_dir + "/" + f  + ".json",train_dir_img)   
     

    for i in range(num_train + num_test, num): 
 
        f = all[i][:-4]
        shutil.move(img_dir + "/" + all[i],train_dir_img)   
        shutil.move(txt_dir + "/" + f  + ".json",train_dir_img)   
      

if __name__ == "__main__": 
    collect_data(base_dir)
    image_conversion(img_img_dir,label_img_dir) 
    split_set(label_img_dir,img_img_dir,fsd_dataset)
(base) cedricperauer@Cedrics-MacBook-Air fsoco_raw %             
(base) cedricperauer@Cedrics-MacBook-Air fsoco_raw % vim split.py 

    cnt = 0
    for subdir, dirs, files in os.walk("/Users/cedricperauer/fsoco_raw"):
        for file in files:
            if file[-4:] == "json" and file != "meta.json" and file != "test.json" and file != "train.json" and file != "val.json" :
                f_name = subdir + "/" + file
                labels.append(f_name)
                cnt += 1
            elif file[-3:] == "png" or file[-3:] == "jpg" :
                f_name = subdir + "/" + file
                images.append(f_name)



def split_set(txt_dir,img_dir,base_dir):

    #dirs for the labels 
    train_dir_img =  base_dir + "/img/train/"
    test_dir_img =  base_dir + "/img/test/"
    val_dir_img =  base_dir + "/img/val/"

    train_dir_txt =  base_dir + "/labels/train/"
    test_dir_txt =  base_dir + "/labels/test/"
    val_dir_txt =  base_dir + "/labels/val/"


    all = os.listdir(img_dir)
    random.shuffle(all)
    print(all)

    for i in range(0,num_train):
        print(i)
        f = all[i][:-4]
        shutil.move(img_dir + "/" + all[i],train_dir_img)
        shutil.move(txt_dir + "/" + all[i]  + ".json",train_dir_txt)

    for i in range(num_train,num_train+num_test):

        f = all[i][:-4]

        shutil.move(img_dir + "/" + all[i],test_dir_img)
        shutil.move(txt_dir + "/" + all[i]  + ".json",test_dir_txt)


    for i in range(num_train + num_test, num):

        f = all[i][:-4]

        shutil.move(img_dir + "/" + all[i],val_dir_img)
        shutil.move(txt_dir + "/" + all[i]  + ".json",val_dir_txt)


if __name__ == "__main__":
    collect_data(base_dir)
    image_conversion(img_store_dir,label_store_dir) 
    split_set(label_store_dir,img_store_dir,fsd_dataset)
