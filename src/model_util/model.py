
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
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
    def __init__(self, classifier_params, backbone_output):
        super().__init__()
        layers_to_add = classifier_params["hidden_layers"]
        self.layers = nn.ModuleList()
        linear_counter = 0
        for layer in layers_to_add:
            if layer['type'] == 'linear':
                if linear_counter == 0:
                    if backbone_output == "1":
                        layer['kwargs']['in_features'] = 4096
                    elif backbone_output == "2":
                        layer['kwargs']['in_features'] = 2048
                    elif backbone_output == "3":
                        layer['kwargs']['in_features'] = 1024
                    elif backbone_output == "pool":
                        layer['kwargs']['in_features'] = 512
                self.layers.append(nn.Linear(**layer['kwargs']))
                linear_counter += 1
            elif layer['type'] == 'relu_activation':
                self.layers.append(nn.ReLU(**layer['kwargs']))
            elif layer['type'] == 'dropout':
                self.layers.append(nn.Dropout(**layer['kwargs']))
        self.attribute_classifier(classifier_params)

    def attribute_classifier(self, classifier_params):
        layers_to_add = classifier_params["attribute_classification_layers"]
        self.attribute_layers_dict = {}
        for layer in layers_to_add:
            if layer['type'] == 'linear':
                linear = nn.Linear(**layer['kwargs'])
                if layer['activation'] == 'softmax':
                    activation = nn.Softmax(dim=None)
                elif layer['activation'] == 'sigmoid':
                    activation = nn.Sigmoid()
                self.attribute_layers_dict[layer['attribute']] = nn.Sequential(
                    linear, activation)
        print(self.attribute_layers_dict)

    def forward(self, backbone_output):
        x = backbone_output
        x = torch.flatten(x, start_dim=1)
        for layer in self.layers:
            x = layer(x)
        print(x.shape)
        attribute_predictions = {}
        for attr_key in list(self.attribute_layers_dict.keys()):
            attribute_predictions[attr_key] = self.attribute_layers_dict[attr_key](
                x)
        return attribute_predictions


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    config = confuse.Configuration('market1501', __name__)
    config.set_file(Path(
        r"C:\\Users\\netra\\GithubEncm369\\reid\\explainable-id-reid\\src\\dataset_util\\market1501.yml"))
    from processor import MarketDataset
    test_obj = MarketDataset(
        config['market_1501_ds']['test_path'].get(), True, False)
    train_obj = MarketDataset(
        config['market_1501_ds']['train_path'].get(), True, True)
    torch_ds = torch.utils.data.DataLoader(test_obj,
                                           batch_size=2, num_workers=8,
                                           collate_fn=collate_fn)
    print(torch_ds)
    for batch_ndx, sample in enumerate(torch_ds):
        print(batch_ndx)
        print(sample)
    testimg, testattr = torch_ds[12560]
    trainimg, trainattr = train_obj[10560]
    trainimg = np.true_divide(trainimg, 255)
    backbone = resnet_fpn_backbone(
        'resnet50', pretrained=True, trainable_layers=3)
    #trainimg_np = torch.from_numpy(trainimg).type('torch.DoubleTensor')
    #trainimg_np = torch.as_tensor(trainimg_np).type('torch.DoubleTensor')
    print("THIS: ", trainimg.shape)
    #trainimg_np = trainimg_np.unsqueeze(3)
    #trainimg_np = torch.reshape(trainimg, (64, 3, 128, 1))
    trainimg_np = trainimg.to(torch.double)
    out = backbone(trainimg_np.float())
    print(trainimg_np.shape)
    print([(k, v.shape) for k, v in out.items()])
    print(out['2'].shape)
    print("\n\n\n\n\n\n\n\n\n\n\n")
    cfg = confuse.Configuration('model_architecture', __name__, read=False)
    cfg.set_file(
        "C:\\Users\\netra\\GithubEncm369\\reid\\explainable-id-reid\\src\\model_util\\classifier_architecture.yml")
    classifier_params = cfg.get()
    # The second argument is the output being used as a String,
    # "1", "2", "3", or "pool"
    obj = Classifier(classifier_params, "2")
    a = obj.forward(out['2'])
    print(a)
