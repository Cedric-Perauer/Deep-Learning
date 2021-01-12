import csv
import pandas 
import random 

csv_name = "all.csv"
csv_name_train = "train.csv"
#icsv_file = pd.read_csv(csv_name)
csvfile = open(csv_name, 'r').readlines()

csvfile_train = open(csv_name_train, 'r').readlines()
header_train = csvfile_train[0:3]
header = csvfile[0:3]

csvfile = csvfile[3:]
random.shuffle(csvfile)


num_total = len(csvfile)
num_train = int(num_total * 0.7)
num_test = int(num_total * 0.15)
num_val = num_total - num_train - num_test



open('dataset_new/train.csv', 'w+').writelines(header_train + csvfile[:num_train])
open('dataset_new/test.csv', 'w+').writelines(header + csvfile[num_train:num_train+num_test])
open('dataset_new/validate.csv', 'w+').writelines(header + csvfile[num_train+num_test:])
