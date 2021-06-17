from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision
import sys
import os
import confuse
from pathlib import Path
import torch
import numpy as np
import torch.nn as nn


class Classifier(nn.Module):
    # configurable, default activation layer is ReLU, but can be changed. Default setting for dropout
    # is False (not included), but can do so
    def __init__(self, num_features, num_layers, hidden_output_size, overall_output_size, activation=nn.ReLU(), dropout=False):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(num_features, hidden_output_size)])
        for i in range(num_layers - 2):
            self.linears.append(activation)
            if dropout is True:
                self.linears.append(nn.Dropout())
            self.linears.append(
                nn.Linear(hidden_output_size, hidden_output_size))
        self.linears.append(activation)
        self.linears.append(nn.Linear(hidden_output_size, overall_output_size))
        print(self.linears)
        print(nn.Sequential(*self.linears))


if __name__ == "__main__":
    config = confuse.Configuration('market1501', __name__)
    config.set_file(Path(
        r"C:\\Users\\netra\\GithubEncm369\\reid\\explainable-id-reid\\src\\dataset_util\\market1501.yml"))
    from processor import MarketDataset
    test_obj = MarketDataset(
        config['market_1501_ds']['test_path'].get(), True, False)
    train_obj = MarketDataset(
        config['market_1501_ds']['train_path'].get(), True, True)

    testimg, testattr = test_obj.view_sample(14560)
    trainimg, trainattr = train_obj.view_sample(10560)
    trainimg = np.true_divide(trainimg, 255)
    backbone = resnet_fpn_backbone(
        'resnet50', pretrained=True, trainable_layers=3)
    trainimg_np = torch.from_numpy(trainimg).type('torch.DoubleTensor')
    trainimg_np = torch.as_tensor(trainimg_np).type('torch.DoubleTensor')
    trainimg_np = trainimg_np.unsqueeze(3)
    trainimg_np = torch.reshape(trainimg_np, (64, 3, 128, 1))
    trainimg_np = trainimg_np.to(torch.double)
    out = backbone(trainimg_np.float())
    print(trainimg_np.shape)
    print([(k, v.shape) for k, v in out.items()])

    print("\n\n\n\n\n\n\n\n\n\n\n")
    obj = Classifier(4, 4, 64, 4, activation=nn.Tanh(), dropout=True)
