'''
Generate Tran and Test Dataset of Lungs
Davide Belli

Requires:
- './images' 		contains all the subdirectories with lungs dataset images
- 'testdata.csv'	contains list of image names, one per row, with one header line at the beginning
- 'traindata.csv'	contains list of image names, one per row, with one header line at the beginning
'''

import os.path
from os import listdir
from os.path import isfile, join
from shutil import copyfile
from shutil import copy2

mypath = "./images"
output_train = "./dataset_lungs/train/train"
output_test = "./dataset_lungs/test/test"

try:
    os.makedirs(output_train)
    os.makedirs(output_test)
except OSError:
    pass

print("Directories found:")
print(listdir(mypath))

images_list = []

# Find all images in every subdirectory
def recursive_image_finder(path):
    file_list = listdir(path)
    for f in file_list:
        if ".png" not in f:
            recursive_image_finder(path+"/"+f)
        else:
            images_list.append((path+"/"+f,f))



# Find all images in ./images
recursive_image_finder(mypath)
print("\nTotal images file found:", len(images_list))
            
# Parse file with no_findings images list
with open("testdata.csv","r") as f:
    no_findings_test = f.readlines()
with open("traindata.csv","r") as f:
    no_findings_train = f.readlines()
    
#Remove headers
del no_findings_test[0]
del no_findings_train[0]
    
for k in range(len(no_findings_test)):
    no_findings_test[k] = no_findings_test[k].replace("\n","")
for k in range(len(no_findings_train)):
    no_findings_train[k] = no_findings_train[k].replace("\n","")
    
print("Images with no findings in test: ", len(no_findings_test)," and in train: ", len(no_findings_train))

# Select images with no findings
healthy_images_test = []
healthy_images_train = []

print("\nStarting to parse images")
for i, h in enumerate(images_list):
    pth, img = h
    if img in no_findings_test:
        healthy_images_test.append((pth,img))
    if img in no_findings_train:
        healthy_images_train.append((pth,img))
    if(i % int(len(images_list)/10) == 0):
        print(int(i/len(images_list)*100), "%...")
        
print("Images saved with no findings in test:", len(healthy_images_test), " and in train: ", len(healthy_images_train))


print("\nGenerating trainset (takes longer)...")

for i, h in enumerate(healthy_images_train):
    pth, img = h
    copy2(pth, output_train)
    
    if(i % int(len(healthy_images_train)/10) == 0):
        print(int(i/len(healthy_images_train)*100), "%...")
   
print("\nGenerating testset (takes longer)...")

for i, h in enumerate(healthy_images_test):
    pth, img = h
    copy2(pth, output_test)
    
    if(i % int(len(healthy_images_test)/10) == 0):
        print(int(i/len(healthy_images_test)*100), "%...")
        