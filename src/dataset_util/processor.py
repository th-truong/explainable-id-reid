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
from torchvision.transforms import functional as F
import traceback


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
config = confuse.Configuration('market1501', __name__)
config.set_file(Path(
    r"C:\\Users\\netra\\GithubEncm369\\reid\\explainable-id-reid\\src\\dataset_util\\market1501.yml"))


class MarketDataset(object):
    # the modes are:
    # 0 - train
    # 1 - validate
    # 2 - test
    def __init__(self, root_path, image, mode):
        self.paths = []
        self.mode = mode
        self.root_path = root_path
        self.attribute_market = self.load_mats(
            config['market_1501_ds']['att_path'].get(), mode)
        # The second parameter in add_path is 0 for images, 1 for .mat files
        if image is True:
            self.paths = self.add_path(root_path, 0)
        else:
            self.paths = self.add_path(root_path, 1)

    def __len__(self):
        print(len(self.paths))
        return len(self.paths)

    def load_mats(self, file_name, mode):
        mat = loadmat(file_name)
        if mode == 0 or mode == 1:
            df = pd.DataFrame.from_records(
                mat["market_attribute"]["train"][0][0][0])
        elif mode == 2:
            df = pd.DataFrame.from_records(
                mat["market_attribute"]["test"][0][0][0])
        map = {}
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
        temp_count = 0
        for file in os.listdir(path):
            if type == 0:
                if file[-4:] == ".jpg":
                    # For validation, indexes from 0002 to 0199 (100 identities) are used.
                    # Everything else is used for training (0201 onwards).
                    if self.mode == 0:
                        if int(file[0:4]) > 199:
                            file_paths.append(os.path.join(path, file))
                    elif self.mode == 1:
                        if int(file[0:4]) <= 199:
                            file_paths.append(os.path.join(path, file))
                    elif self.mode == 2:
                        if file[0:2] != "-1" and file[0:4] != "0000":
                            file_paths.append(os.path.join(path, file))
            elif type == 1:
                if file[-4:] == ".mat":
                    file_paths.append(os.path.join(path, file))
        return file_paths

    def __getitem__(self, idx):
        img, attributes = self.image_loader(self.paths[idx])
        self.targets = {}
        # Always sorted alphabetically
        cols = sorted(list(attributes.columns))
        # Index positions: [upblack, upblue, upgray, upgreen, uppurple, upred, upwhite, upyellow]
        up_colours = []
        # Index positions: [downblack, downblue, downbrown, downgray, downgreen, downpink, downpurple, downwhite, downyellow]
        down_colours = []
        # print(attributes[cols[0]].item())
        for col in cols:
            # No need to include image_index
            if col == "image_index":
                if self.mode == 1:
                    print(attributes[col].item())
                continue
            # Creating a one-hot encoded for age
            try:
                if col == "age":
                    age_list = [0] * 4
                    age_list[int(attributes[col].item()) - 1] = 1
                    self.targets[col] = torch.Tensor(age_list)
                # Grouping the up colors together since they are mutually exclusive
                elif "up" in col and col != "up":
                    # print(attributes[col])
                    up_colours.append(int(attributes[col].item()))
                    # Since there are 9 attributes that contain "up" in it, but one of them is
                    # just "up", which does not correspond to colours, we subtracted one to indicate completion.
                    if len(up_colours) == sum("up" in c for c in cols) - 1:
                        self.targets["up_colours"] = torch.tensor(up_colours)
                # Grouping the down colors together since they are mutually exclusive
                elif "down" in col and col != "down":
                    down_colours.append(int(attributes[col].item()))
                    # Since there are 9 attributes that contain "down" in it, but one of them is
                    # just "down", which does not correspond to colours, we subtracted one to indicate completion.
                    if len(down_colours) == sum("down" in c for c in cols) - 1:
                        self.targets["down_colours"] = torch.tensor(
                            down_colours)
                # For all other attributes.
                else:
                    self.targets[col] = torch.tensor(
                        int(attributes[col].item()))
            except:
                continue
        img = F.to_tensor(img)
        return (img, self.targets)
    # Need a view_sample method

    def view_sample(self, idx):
        img, attr_map = self.__getitem__(idx)
        plt.imshow(img)
        plt.show()
        print(attr_map)
        return (img, attr_map)


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
