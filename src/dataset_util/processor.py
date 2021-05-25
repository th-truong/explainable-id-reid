import os
import numpy as np
import torch
import matplotlib as plt 
import pandas as pd 
from scipy.io import loadmat

class MarketDataset(Dataset):
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