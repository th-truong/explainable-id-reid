from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision
import sys
import os
import confuse
from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
torch.autograd.set_detect_anomaly(True)


class OverallModel(nn.Module):
    def __init__(self, backbone, classifier, output_to_use):
        super(OverallModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.output_to_use = output_to_use

    def forward(self, input, targets):
        back_out = self.backbone(input)
        output = self.classifier(back_out[self.output_to_use])
        return output


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
                        layer['kwargs']['in_features'] = 32768
                    elif backbone_output == "2":
                        layer['kwargs']['in_features'] = 8192
                    elif backbone_output == "3":
                        layer['kwargs']['in_features'] = 2048
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
        activation = None
        for layer in layers_to_add:
            if layer['type'] == 'linear':
                linear = nn.Linear(**layer['kwargs'])
            if layer['activation'] == 'softmax':
                activation = nn.Softmax(dim=None)
            elif layer['activation'] == 'sigmoid':
                activation = nn.Sigmoid()
            if activation == None:
                self.attribute_layers_dict[layer['attribute']] = nn.Sequential(
                    linear)
            else:
                self.attribute_layers_dict[layer['attribute']] = nn.Sequential(
                    linear, activation)

    def forward(self, backbone_output):
        x = backbone_output
        x = torch.flatten(x, start_dim=1)
        for layer in self.layers:
            x = layer(x)

        attribute_predictions = {}
        for attr_key in list(self.attribute_layers_dict.keys()):
            attribute_predictions[attr_key] = self.attribute_layers_dict[attr_key](
                x)
        return attribute_predictions


def collate_fn(batch):
    return tuple(zip(*batch))


writer = SummaryWriter()


def validation_loop(validation_ds, device, model, loss):
    with torch.no_grad():
        for data in tqdm(iter(validation_ds)):
            inputs, labels = data
            images = list(image.to(device) for image in inputs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in labels]
            images = torch.stack(images)
            output = model(images, targets)
            loss = torch.zeros(0)
            for attr in output:
                if attr == "age" or attr == "up_colours" or attr == "down_colours":
                    loss_fn = nn.CrossEntropyLoss()
                    out = output[attr]
                    out = out.to(torch.float32)
                    try:
                        if len(targets) >= 2:
                            target = torch.stack(
                                (targets[0][attr], targets[1][attr]))
                        else:
                            target = torch.stack(targets[0][attr])
                    except:
                        continue
                    target = target.to(torch.long)
                    local_loss = loss_fn(out, target)
                    loss = torch.add(loss, local_loss)
                    precision, recall, fscore = precision_recall_fscore_support(
                        target, out)
                else:
                    loss_fn = nn.BCELoss()
                    out = output[attr]
                    out = out.to(torch.float32)
                    try:
                        if len(targets) >= 2:
                            target = torch.stack(
                                (targets[0][attr], targets[1][attr])).view(2, 1)
                        else:
                            target = torch.stack(targets[0][attr]).view(2, 1)
                    except:
                        continue
                    target = target.to(torch.float32)
                    local_loss = loss_fn(out, target)
                    loss = torch.add(loss, local_loss)
                    precision, recall, fscore = precision_recall_fscore_support(
                        target, out)
                writer.add_scalar(
                    f"Validation {attr} precision, recall, fscore", precision, recall, fscore)
            writer.add_scalar("Validation Loss/train", loss)


def training_loop(torch_ds, validation_ds, optimizer, device, model, loss, epochs=20):
    for i in range(epochs):
        for data in tqdm(iter(torch_ds)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            optimizer.zero_grad()
            images = list(image.to(device) for image in inputs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in labels]
            images = torch.stack(images)
            output = model(images, targets)
            print("\n\n")
            loss = torch.zeros(0)
            for attr in output:
                if attr == "age" or attr == "up_colours" or attr == "down_colours":
                    loss_fn = nn.CrossEntropyLoss()
                    out = output[attr]
                    out = out.to(torch.float32)
                    try:
                        if len(targets) >= 2:
                            target = torch.stack(
                                (targets[0][attr], targets[1][attr]))
                        else:
                            target = torch.stack(targets[0][attr])
                    except:
                        continue
                    target = target.to(torch.long)
                    local_loss = loss_fn(out, target)
                    loss = torch.add(loss, local_loss)
                    writer.add_scalar(f"{attr} Loss/train", local_loss, i)
                else:
                    loss_fn = nn.BCELoss()
                    out = output[attr]
                    out = out.to(torch.float32)
                    try:
                        if len(targets) >= 2:
                            target = torch.stack(
                                (targets[0][attr], targets[1][attr])).view(2, 1)
                        else:
                            target = torch.stack(targets[0][attr]).view(2, 1)
                    except:
                        continue
                    target = target.to(torch.float32)
                    local_loss = loss_fn(out, target)
                    loss = torch.add(loss, local_loss)
                    writer.add_scalar(f"{attr} Loss/train", local_loss, i)
            if loss.nelement() != 0:
                loss.sum().backward()
            else:
                continue
            optimizer.step()
        print("DONE")
        validation_loop(validation_ds, device, model, loss)
        torch.save({'model': obj.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, str(i).zfill(3) + "resnet50_fpn_frcnn_full.tar")


if __name__ == "__main__":
    config = confuse.Configuration('market1501', __name__)
    config.set_file(Path(
        r"C:\\Users\\Div\\Desktop\\Research\\reid\\reid\\explainable-id-reid\\src\\dataset_util\\market1501.yml"))
    from processor import MarketDataset
    test_obj = MarketDataset(
        config['market_1501_ds']['test_path'].get(), True, 2, False)
    train_obj = MarketDataset(
        config['market_1501_ds']['train_path'].get(), True, 0, False)
    validate_obj = MarketDataset(
        config['market_1501_ds']['train_path'].get(), True, 1, False)

    torch_ds_test = torch.utils.data.DataLoader(test_obj,
                                                batch_size=2, num_workers=8,
                                                collate_fn=collate_fn)
    torch_ds_train = torch.utils.data.DataLoader(train_obj,
                                                 batch_size=2, num_workers=8,
                                                 collate_fn=collate_fn)
    torch_ds_val = torch.utils.data.DataLoader(validate_obj,
                                               batch_size=2, num_workers=8,
                                               collate_fn=collate_fn)

    test_data = iter(torch_ds_test)
    print(f"Count of test: {len(test_data)}")
    train_data = iter(torch_ds_train)
    attrs = []
    for img, attr in torch_ds_train:
        attrs.append(attr)
    print(attrs[len(attrs) - 1])

    print(f"Count of train: {len(train_data)}")
    validate_data = iter(torch_ds_val)
    print(f"Count of validate: {len(validate_data)}")

    # Parameters for loop:
    backbone = resnet_fpn_backbone(
        'resnet50', pretrained=True, trainable_layers=0)
    cfg = confuse.Configuration('model_architecture', __name__, read=False)
    cfg.set_file(
        "C:\\Users\\Div\\Desktop\\Research\\reid\\reid\\explainable-id-reid\\src\\model_util\\classifier_architecture.yml")
    classifier_params = cfg.get()
    # The second argument is the output being used as a String,
    # "1", "2", "3", or "pool"
    obj = Classifier(classifier_params, "3")
    model = OverallModel(backbone, obj, "3")
    model = model.train()
    children = list(model.children())
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.SGD(obj.parameters(), lr=0.001, momentum=0.9)
    epochs = 20
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    training_loop(torch_ds_train, torch_ds_val, optimizer,
                  device, model, criteria, epochs)

    print('Finished Training')
    writer.flush()
