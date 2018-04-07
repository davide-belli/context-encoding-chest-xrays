import os.path
from os import listdir
from os.path import isfile, join
from shutil import copyfile
from shutil import copy2
import random
from PIL import Image
import numpy as np

random.seed(1234)

mypath = "./dataset_lungs/train/"
output_train = "./dataset_lungs/train_randomPatches/train_randomPatches/"

try:
    os.makedirs(output_train)
except OSError:
    pass

print("Directories found:")
print(listdir(mypath))

images_list = []

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )

with open("traindata_bb.csv","r") as f:
    coordinates = f.readlines()
    
del coordinates[0]


crop = np.zeros((128,128))
for index, c in enumerate(coordinates):
    if index % 1000 == 0:
        print(index, "/", len(coordinates))
    c.replace("\n","")
#     print(c)
    
    _, filename,llx0,lly0,llx1,lly1,rlx0,rly0,rlx1,rly1 = c.split(",")
    inname = mypath + "train/" + filename + ".png"
    image = load_image(inname)
    
    for k in range(20):
        outname = output_train + filename + "_" + str(k) + ".png"
        coin = random.randint(0, 1)
        if not coin:
            x0, x1, y0, y1 = int(llx0), int(llx1), int(lly0), int(lly1)
        else:
            x0, x1, y0, y1 = int(rlx0), int(rlx1), int(rly0), int(rly1)

#         print(inname)
#         print(coin, x0, x1, y0, y1)
        if (x1-x0 > 128):
            X = random.randint(x0+64, x1-64)
        else:
            x0 = max(64,x0)
            x1 = min(1024-64,x1)
            if x0 > x1:
#                 print("skip", index)
                continue
            X = random.randint(x0, x1)
#             print(k, filename)
            
        if (y1-y0) >128:
            Y = random.randint(y0+64, y1-64)
        else:
            y0 = max(64,y0)
            y1 = min(1024-64,y1)
            if y0 > y1:
#                 print("skip", index)
                continue
            Y = random.randint(y0, y1)
#             print(k, filename)
#         print(X, Y)
        s = image[X-64:X+64, Y-64:Y+64].shape
        if (s[0], s[1]) != (128,128):
            print("SIZE ERROR", crop.shape, c, X, Y)
            continue
            
        if len(image.shape) > 2:
            crop[:,:] = image[X-64:X+64, Y-64:Y+64, 0]
        else:
            crop[:,:] = image[X-64:X+64, Y-64:Y+64]

#         print(crop.shape)
#         save_image(image[:,0:20], outname)
#         save_image(image, outname)
        save_image(crop, outname)
#     input()