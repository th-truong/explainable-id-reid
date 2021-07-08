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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class MarketDataset(object):
    # the modes are:
    # 0 - train
    # 1 - validate
    # 2 - test
    def __init__(self, root_path, image, mode, one_hot_encoded, attributes_to_use):
        self.paths = []
        self.mode = mode
        self.root_path = root_path
        self.one_hot = one_hot_encoded
        self.attribute_market = self.load_mats(
            config['market_1501_ds']['att_path'].get(), mode)
        self.attributes_to_use = attributes_to_use

        # The second parameter in add_path is 0 for images, 1 for .mat files
        if image is True:
            self.paths = self.add_path(root_path, 0)

        else:
            self.paths = self.add_path(root_path, 1)

    def __len__(self):
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
        indexes_to_skip = []
        for row in self.attribute_market.iterrows():
            if row[1]['upblack'] == 0 and row[1]['upwhite'] == 0 and row[1]['upred'] == 0 and row[1]['uppurple'] == 0 and row[1]['upyellow'] == 0 and row[1]['upgray'] == 0 and row[1]['upblue'] == 0 and row[1]['upgreen'] == 0:
                indexes_to_skip.append(row[1]['image_index'][0])
            if row[1]['downblack'] == 0 and row[1]['downwhite'] == 0 and row[1]['downpink'] == 0 and row[1]['downpurple'] == 0 and row[1]['downyellow'] == 0 and row[1]['downgray'] == 0 and row[1]['downblue'] == 0 and row[1]['downgreen'] == 0 and row[1]['downbrown'] == 0:
                if row[1]['image_index'][0] not in indexes_to_skip:
                    indexes_to_skip.append(row[1]['image_index'][0])
        for file in os.listdir(path):
            if type == 0:
                if file[-4:] == ".jpg":
                    if file[0:4] in indexes_to_skip:
                        continue
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
        cols = []
        for col in sorted(list(attributes.columns)):
            if col in self.attributes_to_use:
                cols.append(col)
            if "down_colours" in self.attributes_to_use and "down" in col and col != "down":
                cols.append(col)
            if "up_colours" in self.attributes_to_use and "up" in col and col != "up":
                cols.append(col)
        # Index positions: [upblack, upblue, upgray, upgreen, uppurple, upred, upwhite, upyellow]
        up_colours = []
        # Index positions: [downblack, downblue, downbrown, downgray, downgreen, downpink, downpurple, downwhite, downyellow]
        down_colours = []
        for col in cols:
            # No need to include image_index
            if col == "image_index":
                continue
            if col == "age":
                # Creating a one-hot encoded for age
                if self.one_hot == True:
                    age_list = [0] * 4
                    age_list[int(attributes[col].item()) - 1] = 1
                    self.targets[col] = torch.Tensor(age_list)
                else:
                    self.targets[col] = torch.tensor(int(attributes[col].item() - 1))
            # Grouping the up colors together since they are mutually exclusive
            elif "up" in col and col != "up":
                up_colours.append(int(attributes[col].item()))
                # Since there are 9 attributes that contain "up" in it, but one of them is 
                # just "up", which does not correspond to colours, we subtracted one to indicate completion.
                if len(up_colours) == sum("up" in c for c in cols) - 1:
                    if self.one_hot == True:
                        self.targets["up_colours"] = torch.tensor(up_colours)
                    else:
                        self.targets["up_colours"] = torch.tensor(up_colours.index(1))
            # Grouping the down colors together since they are mutually exclusive
            elif "down" in col and col != "down":
                down_colours.append(int(attributes[col].item()))
                # Since there are 9 attributes that contain "down" in it, but one of them is 
                # just "down", which does not correspond to colours, we subtracted one to indicate completion.
                if len(down_colours) == sum("down" in c for c in cols) - 1:
                    if self.one_hot == True:
                        self.targets["down_colours"] = torch.tensor(down_colours)
                    else:
                        self.targets["down_colours"] = torch.tensor(down_colours.index(1))
            # For all other attributes.
            else:
                self.targets[col] = torch.tensor(int(attributes[col].item()))
        img = F.to_tensor(img)
        return (img, self.targets)

    def view_sample(self, idx):
        img, attr_map = self.__getitem__(idx)
        return (img, attr_map)

if __name__ == "__main__":
    config = confuse.Configuration('market1501', __name__)
    config.set_file(Path(r"C:\\Users\\Div\\Desktop\\Research\\reid\\reid\\explainable-id-reid\\src\\dataset_util\\market1501.yml"))
    test_obj = MarketDataset(
        config['market_1501_ds']['test_path'].get(), True, 2, False)
    train_obj = MarketDataset(
        config['market_1501_ds']['train_path'].get(), True, 0, False)

    test_obj.view_sample(10560)
    train_obj.view_sample(10560)