from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision
import sys
import os
import confuse
from pathlib import Path
import torch
import numpy as np
from dataset_util.processor import MarketDataset

config = confuse.Configuration('market1501', __name__)
config.set_file(Path(
    r"C:\\Users\\netra\\GithubEncm369\\reid\\explainable-id-reid\\src\\dataset_util\\market1501.yml"))
test_obj = MarketDataset(
    config['market_1501_ds']['test_path'].get(), True, False)
train_obj = MarketDataset(
    config['market_1501_ds']['train_path'].get(), True, True)

for i in range(len(test_obj.paths)):
    test_obj.view_sample(i)
for i in range(len(train_obj.paths)):
    train_obj.view_sample(i)
#trainimg = np.true_divide(trainimg, 255)
#backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)
#trainimg_np = torch.from_numpy(trainimg).type('torch.DoubleTensor')
#trainimg_np = torch.as_tensor(trainimg_np).type('torch.DoubleTensor')
#trainimg_np = trainimg_np.unsqueeze(3)
#trainimg_np = torch.reshape(trainimg_np, (64, 3, 128, 1))
#trainimg_np = trainimg_np.to(torch.double)
#out = backbone(trainimg_np.float())
# print(trainimg_np.shape)
#print([(k, v.shape) for k, v in out.items()])
