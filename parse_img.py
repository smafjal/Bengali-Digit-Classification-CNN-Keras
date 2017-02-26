#!usr/bin/env python
__author__="smafjal"

from PIL import Image
import os
import glob
from skimage.io import imread
import numpy as np
import pickle

img_size=28, 28

class ImageReader():
    def __init__(self, dir):
        self.data_dir = dir

    def read_image(self):
        files = os.listdir(self.data_dir)
        image_list = []
        label_list=[]
        for filename in files:
            label = self.get_label(filename)
            if label == -1:
                print "Label is not found"
                continue

            filename = os.path.join(self.data_dir, filename) + "/*.bmp"
            print "Image Label:----> ",label
            for img_file in glob.glob(filename):
                img=Image.open(img_file).convert("L")
                img=img.resize(img_size, Image.ANTIALIAS)
                img=np.array(img,dtype=np.float64).reshape(-1)
                image_list.append(img)
                a=[0]*10; a[label]=1 # making hot vector
                a=np.array(a,dtype=np.int32)
                label_list.append(a)
        return np.array(image_list),np.array(label_list)

    def shuffle_data(self,data_x,data_y):
        a=np.arange(len(data_x))
        np.random.shuffle(a)
        np.random.shuffle(a)
        data_x=data_x[a]
        data_y=data_y[a]
        return data_x,data_y

    def data_normalization(self,data):
        for i in range(len(data)):
            for j in range(len(data[i])):
                val=((data[i][j]*1.0)/255.0)
                data[i][j]=val
        return data

    def data_normalization2(self,matrix):
        sum = np.sum(matrix)
        if sum > 0.:
            return matrix / sum
        else:
            return matrix

    def get_label(self, filename):
        for i in range(10):
            if int(filename[-1]) == i:
                return i
        return -1

    def save_pickle(self,data,file_name):
        with open(file_name+".pickle","wb") as w:
            pickle.dump(data,w)

def main():
    data = ImageReader("data/train")
    img_data,img_label = data.read_image()
    img_data,img_label=data.shuffle_data(img_data,img_label)
    img_data=data.data_normalization(img_data)

    print "Data saving on pickle format "
    data.save_pickle(img_data,"pickle/img_data")
    data.save_pickle(img_label,"pickle/img_label")

    print "Data-Len: ",len(img_data),len(img_data[0])

if __name__ == "__main__":
    main()
