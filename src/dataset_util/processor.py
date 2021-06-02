import os
import numpy as np
import torch
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy.io import loadmat
from PIL import Image
from glob import glob
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class MarketDataset(object):
    def __init__(self, root_path, image, train):
        self.paths = []
        self.files = []
        self.root_path = root_path
        self.attribute_market = self.load_mats(r"C:\\Users\\Div\\Desktop\\Research\\reid\\reid\\Market-1501-v15.09.15\\Attributes\\market_attribute.mat", train)

        if image is True:
            self.paths = self.add_path(root_path, 0)
            for img in self.paths:
                self.image_loader(img)
        else:
            self.paths = self.add_path(root_path, 1)
            for mat in self.paths:
                self.files.append(self.load_mats(mat))

    def load_mats(self, file_name, train):
        mat = loadmat(file_name)
        if train is True:
            df = pd.DataFrame.from_records(mat["market_attribute"]["train"][0][0][0])
        elif train is False:
            df = pd.DataFrame.from_records(mat["market_attribute"]["test"][0][0][0])
        map = {}
        for attr in df:
            map[attr] = df[attr][0][0]
        df2 = pd.DataFrame(map)
        return df2

    def image_loader(self, path):
        splits = path.split("\\")
        img_name = splits[len(splits) - 1]
        self.files.append(((np.array(Image.open(path))), self.attribute_market.loc[self.attribute_market["image_index"] == img_name[:img_name.find("_")]]))
        # Showing a random image to see it works.
        #plt.imshow(self.files[0][0])
        #print(self.files[0][1])
        #plt.show()

    def add_path(self, path, type):
        file_paths = []
        for file in os.listdir(path):
            if type == 0:
                if file[-4:] == ".jpg":
                    file_paths.append(os.path.join(path, file))
            elif type == 1:
                if file[-4:] == ".mat":
                    file_paths.append(os.path.join(path, file))
        return file_paths

    def __getitem__(self, idx):
        item = self.files[idx]
        return item[0], item[1]
        

test_obj = MarketDataset("C:\\Users\\Div\\Desktop\\Research\\reid\\reid\\Market-1501-v15.09.15\\Images\\bounding_box_test", True, False)
train_obj = MarketDataset("C:\\Users\\Div\\Desktop\\Research\\reid\\reid\\Market-1501-v15.09.15\\Images\\bounding_box_train", True, True)

test_sample_image, test_sample_attr = test_obj.__getitem__(12000)
# For train images, those image_indexes for some reason are not a part of market_attribute.mat
# I don't know why...
train_sample_image, train_sample_attr = train_obj.__getitem__(12000)

plt.imshow(test_sample_image)
plt.show()
print(test_sample_attr)
plt.imshow(train_sample_image)
plt.show()
print(train_sample_attr)