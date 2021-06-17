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
    def __init__(self, classifier_params):
        super().__init__()
        layers_to_add = classifier_params['hidden_layers']
        self.layers = torch.nn.ModuleList()
        for layer in layers_to_add:
            if layer['type'] == 'linear':
                self.layers.append(torch.nn.Linear(**layer['kwargs']))
            elif layer['type'] == 'relu_activation':
                self.layers.append(torch.nn.ReLU(**layer['kwargs']))
            elif layer['type'] == 'dropout':
                self.layers.append(torch.nn.Dropout(**layer['kwargs']))

    def attribute_classifier(self, classifier_params):
        layers_to_add = classifier_params['attribute_classification_layers']
        self.attribute_layers = nn.ModuleList()
        for layer in layers_to_add:
            if layer['type'] == 'linear':
                self.attribute_layers.append(nn.Linear(**layer['kwargs']))
                if layer['activation'] == 'softmax':
                    self.attribute_layers.append(nn.Softmax())
                elif layer['activation'] == 'sigmoid':
                    self.attribute_layers.append(nn.Sigmoid())
                setattr(self, layer['attribute'], self.attribute_layers)
                self.attribute_layers = nn.ModuleList()
        print(self.__dir__)


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

    cfg = confuse.Configuration('model_architecture', __name__, read=False)
    cfg.set_file(
        "C:\\Users\\netra\\GithubEncm369\\reid\\explainable-id-reid\\src\\model_util\\classifier_architecture.yml")
    classifier_params = cfg.get()
    obj = Classifier(classifier_params)
    obj.attribute_classifier(classifier_params)
