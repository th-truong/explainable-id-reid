import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from PIL import Image
import sys
import confuse
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class MarketDataset(object):
    def __init__(self, root_path, image, train):
        self.paths = []
        self.root_path = root_path
        self.attribute_market = self.load_mats(
            config['market_1501_ds']['att_path'].get(), train)

        if image is True:
            self.paths = self.add_path(root_path, 0)

        else:
            self.paths = self.add_path(root_path, 1)

    def load_mats(self, file_name, train):
        mat = loadmat(file_name)
        if train is True:
            df = pd.DataFrame.from_records(
                mat["market_attribute"]["train"][0][0][0])
        elif train is False:
            df = pd.DataFrame.from_records(
                mat["market_attribute"]["test"][0][0][0])
        map = {}
        for attr in df:
            map[attr] = df[attr][0][0]
        df2 = pd.DataFrame(map)
        for attr in df:
            map[attr] = df[attr][0][0]
        df2 = pd.DataFrame(map)
        for col in list(df2.columns):
            if col != "age" and col != "image_index":
                df2[col] -= 1
        return df2

    def image_loader(self, path):
        splits = path.split("\\")
        img_name = splits[len(splits) - 1]
        return np.array(Image.open(path)), self.attribute_market.loc[self.attribute_market["image_index"] == img_name[:img_name.find("_")]]

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
        img, attributes = self.image_loader(self.paths[idx])
        return img, attributes

    # Need a view_sample method
    def view_sample(self, idx):
        img, attr = self.__getitem__(idx)
        plt.imshow(img)
        plt.show()
        print(attr)


if __name__ == "__main__":
    config = confuse.Configuration('market1501', __name__)
    config.set_file(Path(
        r"C:\\Users\\netra\\GithubEncm369\\reid\\explainable-id-reid\\src\\dataset_util\\market1501.yml"))
    test_obj = MarketDataset(
        config['market_1501_ds']['test_path'].get(), True, False)
    train_obj = MarketDataset(
        config['market_1501_ds']['train_path'].get(), True, True)

    test_obj.view_sample(12000)
    train_obj.view_sample(12000)
