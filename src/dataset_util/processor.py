import os
import numpy as np
import torch
import matplotlib as plt 
import pandas as pd 
from scipy.io import loadmat
from PIL import Image

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
