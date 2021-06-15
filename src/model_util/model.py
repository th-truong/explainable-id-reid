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
    def __init__(self, num_features, num_layers):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(num_features, 64)])
        self.linears.extend([nn.Linear(64, 64) for i in range(num_layers - 1)])
        length = len(self.linears)
        i = 0
        while i < length:
            if i%2 == 1 and i != length - 1:
                self.linears.insert(i, [nn.Dropout()])
            i+=1
            length = len(self.linears)
        i = 1
        length = len(self.linears)
        while i < length:
            self.linears.insert(i, [nn.ReLU()])
            i += 2
            length = len(self.linears)
        print(self.linears)

if __name__ == "__main__":
    config = confuse.Configuration('market1501', __name__)
    config.set_file(Path(r"C:\\Users\\Div\\Desktop\\Research\\reid\\reid\\explainable-id-reid\\src\\dataset_util\\market1501.yml"))
    from processor import MarketDataset
    test_obj = MarketDataset(
        config['market_1501_ds']['test_path'].get(), True, False)
    train_obj = MarketDataset(
        config['market_1501_ds']['train_path'].get(), True, True)

    testimg, testattr = test_obj.view_sample(14560)
    trainimg, trainattr = train_obj.view_sample(10560)
    trainimg = np.true_divide(trainimg, 255)
    backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)
    trainimg_np = torch.from_numpy(trainimg).type('torch.DoubleTensor')
    trainimg_np = torch.as_tensor(trainimg_np).type('torch.DoubleTensor')
    trainimg_np = trainimg_np.unsqueeze(3)
    trainimg_np = torch.reshape(trainimg_np, (64, 3, 128, 1))
    trainimg_np = trainimg_np.to(torch.double)
    out = backbone(trainimg_np.float())
    print(trainimg_np.shape)
    print([(k, v.shape) for k, v in out.items()])

    print("\n\n\n\n\n\n\n\n\n\n\n")
    obj = Classifier(4, 4)