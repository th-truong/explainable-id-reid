import os
import numpy as np
import torch
import matplotlib as plt 
import pandas as pd 
from scipy.io import loadmat
from PIL import Image
from glob import glob
import sys

class MarketDataset(object):
    def __init__(self, root_image_path, root_label_path):
        super().__init__()
        image_paths = []
        label_paths = []

        for path in root_image_path:
            image_paths.append(self.add_path(path))

        for path in root_label_path:
            label_paths.append(self.add_path(path))

        self.image_paths = image_paths
        self.label_paths = label_paths

    def load_mats(self, file_name):
        mat = loadmat(file_name)
        df2 = pd.DataFrame.from_records(mat["market_attribute"][0][0][0][0])
        map = {}
        for attr in df2:
            map[attr] = df2[attr][0][0]

        self.df2 = df2

    def image_loader(self, path):
        paths = []

        # Root hardcoded in right now.
        root = "C:\\Users\\Div\\Desktop\\Research\\reid\\reid\\Market-1501-v15.09.15"
        path = os.listdir(root)

        for sub_dirs in path:
            paths.append(os.path.join(root, sub_dirs))

        bounding_box_test = []
        bounding_box_train = []
        gt_bbox = []
        query = []

        for img in paths:
            name = img.rsplit("\\", 1)[1]
            if "." not in name:
                for img_name in os.listdir(img):
                    if ".mat" not in img_name and ".db" not in img_name:
                        globals()[name].append(np.array(Image.open(os.path.join(img, img_name))))
                        print("in progress...")
            print(f"{img} done!")

        print(len(gt_bbox))
        # Showing a random image to see it works.
        plt.imshow(gt_bbox[1000])
        plt.show()
        self.bounding_box_test = bounding_box_test
        self.bounding_box_train = bounding_box_train
        self.gt_bbox = []
        self.query = query

    def add_path(self, path):
        path_to_add = glob(path + "\*")

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "Market1501", self.image_paths[idx])
        attribute_path = os.path.join(self.root, "market_attribute", self.label_paths[idx])
        img = Image.open(img_path)
        attribute = open(attribute_path, "r")
        
        img = np.array(img)
        
        return img, attribute
        
    


path_1 = 'C:\SD_Card\Summer_Research\Market-1501-v15.09.15.zip\Market-1501-v15.09.15\bounding_box_test'
path_2 = 'C:\SD_Card\Summer_Research\market_attribute'
data = MarketDataset(path_1, path_2)
