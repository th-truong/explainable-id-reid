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
import albumentations as A

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# The MarketDataset creates an object of the attribute file as a panda dataframe as well as
# the paths to the images used for training/validating/testing.
class MarketDataset(object):
    # the modes are:
    # 0 - train
    # 1 - validate
    # 2 - test
    def __init__(self, config, root_path, image, mode, one_hot_encoded, attributes_to_use):
        self.paths = []
        self.mode = mode
        self.root_path = root_path
        self.one_hot = one_hot_encoded
        self.attribute_market = self.load_mats(
            config['market_1501_ds']['att_path'].get(), mode)
        self.attributes_to_use = attributes_to_use

        # The second parameter in add_path is 0 for images, 1 for .mat files.
        if image is True:
            self.paths = self.add_path(root_path, 0)

        else:
            self.paths = self.add_path(root_path, 1)

    # Returns the length of the paths (the number of images).
    def __len__(self):
        return len(self.paths)
    
    # Loading mat files and putting them in a panda dataframe, based on mode: train or test.
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

    # Returning the read in the image as well as the image_index (the person identity number) for the image.
    def image_loader(self, path):
        splits = path.split("\\")
        img_name = splits[len(splits) - 1]
        img = Image.open(path)
        img = np.array(img)
        # Applying tranformations on the image to offset the imbalanced attributes.
        transform = A.Compose([
            A.CLAHE(p = 0.5),
            A.HorizontalFlip(p = 0.5)
        ])
        transformed_image = transform(image=img)['image']
        transformed = Image.fromarray(transformed_image, 'RGB')
        return transformed_image, self.attribute_market.loc[self.attribute_market["image_index"] == img_name[:img_name.find("_")]]

    # Adds the full paths to the images, seperates the training set into training and validation based on 
    # the person IDs.
    def add_path(self, path, type):
        file_paths = []
        indexes_to_skip = []
        for row in self.attribute_market.iterrows():
            if row[1]['upblack'] == 0 and row[1]['upwhite'] == 0 and row[1]['upred'] == 0 and row[1]['uppurple'] == 0 and row[1]['upyellow'] == 0 and row[1]['upgray'] == 0 and row[1]['upblue'] == 0 and row[1]['upgreen'] == 0:
                indexes_to_skip.append(row[1]['image_index'][0])
            if row[1]['downblack'] == 0 and row[1]['downwhite'] == 0 and row[1]['downpink'] == 0 and row[1]['downpurple'] == 0 and row[1]['downyellow'] == 0 and row[1]['downgray'] == 0 and row[1]['downblue'] == 0 and row[1]['downgreen'] == 0 and row[1]['downbrown'] == 0:
                if row[1]['image_index'][0] not in indexes_to_skip:
                    indexes_to_skip.append(row[1]['image_index'][0])
        for idx in indexes_to_skip:
            row = self.attribute_market.loc[self.attribute_market["image_index"] == idx].index
            self.attribute_market.drop(row, inplace = True)
        self.identities = []
        for file in sorted(os.listdir(path)):
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
                            self.identities.append(file[0:4])
            elif type == 1:
                if file[-4:] == ".mat":
                    file_paths.append(os.path.join(path, file))
        return file_paths

    # Returns the image tensor, the attribute description dictionary, and the image_index.
    def __getitem__(self, idx):
        img, attributes = self.image_loader(self.paths[idx])
        self.targets = {}
        # Always sorted alphabetically
        cols = []
        for col in sorted(list(attributes.columns)):
            if col in self.attributes_to_use or col == 'image_index':
                cols.append(col)
            if "down_colours" in self.attributes_to_use and "down" in col and col != "down":
                cols.append(col)
            if "up_colours" in self.attributes_to_use and "up" in col and col != "up":
                cols.append(col)
        # Index positions: [upblack, upblue, upgray, upgreen, uppurple, upred, upwhite, upyellow]
        up_colours = []
        # Index positions: [downblack, downblue, downbrown, downgray, downgreen, downpink, downpurple, downwhite, downyellow]
        down_colours = []
        image_index = 0
        for col in cols:
            # No need to include image_index
            if col == "image_index":
                image_index = attributes[col].item()[0]
            elif col == "age":
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
                if len(up_colours) == 8:
                    if self.one_hot == True:
                        self.targets["up_colours"] = torch.tensor(up_colours)
                    else:
                        self.targets["up_colours"] = torch.tensor(up_colours.index(1))
            # Grouping the down colors together since they are mutually exclusive
            elif "down" in col and col != "down":
                down_colours.append(int(attributes[col].item()))
                # Since there are 9 attributes that contain "down" in it, but one of them is 
                # just "down", which does not correspond to colours, we subtracted one to indicate completion.
                if len(down_colours) == 9:
                    if self.one_hot == True:
                        self.targets["down_colours"] = torch.tensor(down_colours)
                    else:
                        self.targets["down_colours"] = torch.tensor(down_colours.index(1))
            # For all other attributes.
            else:
                self.targets[col] = torch.tensor(int(attributes[col].item()))
        img = F.to_tensor(img)
        return (img, self.targets, image_index)

    # Calls getitem() and returns the values.
    def view_sample(self, idx):
        img, attr_map, image_index = self.__getitem__(idx)
        return (img, attr_map, image_index)