from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch
import os
import numpy as np
import torch
import tensor
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from PIL import Image
import sys
from pathlib import Path
import confuse
import yaml
from updates import MarketDataset

config = confuse.Configuration('market1501', __name__)
config.set_file(Path(r"D:\\Summer_Research\\Reid\\market1501.yml"))
test_obj = MarketDataset(
    config['market_1501_ds']['test_path'].get(), True, False)
train_obj = MarketDataset(
    config['market_1501_ds']['train_path'].get(), True, True)

testimg, testattr = test_obj.view_sample(12000)
trainimg, trainattr = train_obj.view_sample(12000)

trainimg = np.true_divide(trainimg, 255)
backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)
trainimg_np = torch.from_numpy(trainimg).type('torch.DoubleTensor')
trainimg_np = torch.as_tensor(trainimg_np).type('torch.DoubleTensor')
print(trainimg_np.shape)
trainimg_np = trainimg_np.unsqueeze(3)
print(trainimg_np.shape)
trainimg_np = torch.reshape(trainimg_np, (64, 3, 128, 1))
trainimg_np = trainimg_np.to(torch.double)
output = backbone(trainimg_np.float())
print(trainimg_np.shape)
print([(k, v.shape) for k, v in output.items()])
