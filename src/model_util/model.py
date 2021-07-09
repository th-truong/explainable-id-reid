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
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
torch.autograd.set_detect_anomaly(True)


class OverallModel(nn.Module):
    def __init__(self, backbone, classifier, output_to_use):
        super(OverallModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.output_to_use = output_to_use
        self.loss_layers = nn.ModuleDict({'age': nn.CrossEntropyLoss(),
                                          'backpack': nn.BCELoss(),
                                          'bag': nn.BCELoss(),
                                          'clothes': nn.BCELoss(),
                                          'down': nn.BCELoss(),
                                          'down_colours': nn.CrossEntropyLoss(),
                                          'gender': nn.BCELoss(),
                                          'hair': nn.BCELoss(),
                                          'handbag': nn.BCELoss(),
                                          'hat': nn.BCELoss(),
                                          'up': nn.BCELoss(),
                                          'up_colours': nn.CrossEntropyLoss()})

    def forward(self, input, targets):
        back_out = self.backbone(input)
        output = self.classifier(back_out[self.output_to_use])
        target_outputs = {}
        if len(targets) >= 2:
            for attr in targets[0].keys():
                target_list = []
                for index in range(len(targets)):
                    target_list.append(targets[index][attr])
                target_outputs[attr] = torch.stack(tuple(target_list))
        else: 
            target_outputs = targets[0]
            for attr in target_outputs:
                target_outputs[attr] = target_outputs[attr].view(1)
        output_dict = {}
        if self.training:
            for attribute in target_outputs:
                if str(type(self.loss_layers[attribute])) == "<class 'torch.nn.modules.loss.BCELoss'>":
                    for attr in target_outputs:
                        target_outputs[attr] = target_outputs[attr].type(torch.float32)
                    for attr in output:
                        output[attr] = output[attr].type(torch.float32)
                    output_dict[attribute] = self.loss_layers[attribute](output[attribute], target_outputs[attribute].view(output[attribute].shape))
                else: 
                    for attr in target_outputs:
                        target_outputs[attr] = target_outputs[attr].type(torch.long)
                    for attr in output:
                        output[attr] = output[attr].type(torch.float32)
                    output_dict[attribute] = self.loss_layers[attribute](output[attribute], target_outputs[attribute])
        return output, output_dict

def metric_calculator(pred_and_true, classifier_params, epoch):
    metrics = {}
    predictions = {}
    real = {}

    for attr in classifier_params:
        predictions[attr] = []
        real[attr] = []
        metrics[attr] = 0
    
    for t in pred_and_true:
        for attr in classifier_params:
            if attr == 'age' or attr == 'up_colours' or attr == 'down_colours':
                to_add = torch.argmax(t[0][attr], dim = 1)
                predictions[attr].append(to_add)
                real[attr].append(t[1][attr])
            else:
                predictions[attr].append(t[0][attr])
                real[attr].append(t[1][attr])

    for attr in predictions:
        predictions[attr] = torch.stack(tuple(predictions[attr]))

    for attr in real:
        real[attr] = torch.stack(tuple(real[attr]))    
    
    for attr in metrics:
        if attr != 'age' and attr != 'down_colours' and attr != 'up_colours':
            precision, recall, _, _ = precision_recall_fscore_support(torch.flatten(real[attr]), torch.round(torch.flatten(predictions[attr]).type(torch.float)), average = 'macro')
            accuracy = accuracy_score(torch.flatten(real[attr]), torch.round(torch.flatten(predictions[attr]).type(torch.float)))
            metrics[attr] = {'precision': precision, 'recall': recall, 'accuracy': accuracy}
        else:
            conf = confusion_matrix(torch.flatten(real[attr]), torch.round(torch.flatten(predictions[attr]).type(torch.float)))
            print(f"\nEPOCH: {epoch}")
            print(conf)
            print(classification_report(torch.flatten(real[attr]), torch.round(torch.flatten(predictions[attr]).type(torch.float)), digits=3))
            print("\n")
            precision, recall, _, _ = precision_recall_fscore_support(torch.flatten(real[attr]), torch.round(torch.flatten(predictions[attr]).type(torch.float)), average = 'macro')
            metrics[attr] = {'precision': precision, 'recall': recall}      
    return metrics

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
        layers_to_use = classifier_params["attributes_to_use"]
        self.attribute_layers_dict = {}
        activation = None
        for layer in layers_to_add:
            if layer['attribute'] in layers_to_use:
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


def validation_loop(validation_ds, device, model, classifier_params, epoch):
    predictions_and_real = []
    step_counter = 0
    with torch.no_grad():
        for data in tqdm(iter(validation_ds)):
            inputs, labels = data
            images = list(image.to(device) for image in inputs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in labels]
            target_attrs = {}
            for attr in targets[0].keys():
                target_list = []
                for index in range(len(targets)):
                    target_list.append(targets[index][attr])
                target_attrs[attr] = torch.stack(tuple(target_list))
            images = torch.stack(images)
            output, losses = model(images, targets)
            predictions_and_real.append((output, target_attrs))
            if losses == False:
                continue
            loss = torch.Tensor([0])
            for attr in losses:
                loss = loss + losses[attr].item()
                print(f"Validation: Attr: {attr} || Loss: {losses[attr].item()}")
                writer.add_scalar(f"{attr} Loss/Validation", losses[attr].item(), step_counter)
            print(f"Validation Overall Loss: {loss}")
            writer.add_scalar("Validation Overall Loss", loss, step_counter)
            step_counter += 1
        metrics = metric_calculator(predictions_and_real, classifier_params, epoch)
        for attr in metrics:
            for metric in metrics[attr]:
                writer.add_scalar(f"{attr} {metric}", metrics[attr][metric], epoch)


def training_loop(torch_ds, validation_ds, optimizer, device, model, classifier_params, epochs=20):
    step_counter = 0
    for i in range(epochs):
        for data in tqdm(iter(torch_ds)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            optimizer.zero_grad()
            images = list(image.to(device) for image in inputs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in labels]
            images = torch.stack(images)
            _,output = model(images, targets)
            if output == False:
                continue
            print("\n\n")
            loss = torch.Tensor([0])
            loss.requires_grad = True
            for attr in output:
                loss = loss + output[attr].item()
                print(f"Training: Attr: {attr} || Loss: {output[attr].item()}")
                writer.add_scalar(f"{attr} Loss/train", output[attr].item(), step_counter)
            print(f"Training: Overall Loss: {loss}")
            writer.add_scalar("Training Overall Loss", loss, step_counter)
            loss.backward()
            optimizer.step()
            step_counter += 1
        print("DONE")
        validation_loop(validation_ds, device, model, classifier_params, i)
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, str(i).zfill(3) + "resnet50_fpn_frcnn_full.tar")
    writer.flush()
    writer.close()


if __name__ == "__main__":
    writer = SummaryWriter()
    config = confuse.Configuration('market1501', __name__)
    config.set_file(Path(
        r"C:\\Users\\Div\\Desktop\\Research\\reid\\reid\\explainable-id-reid\\src\\dataset_util\\market1501.yml"))
    cfg = confuse.Configuration('model_architecture', __name__, read=False)
    cfg.set_file(
        "C:\\Users\\Div\\Desktop\\Research\\reid\\reid\\explainable-id-reid\\src\\model_util\\classifier_architecture.yml")
    classifier_params = cfg.get()

    from processor import MarketDataset
    test_obj = MarketDataset(
        config['market_1501_ds']['test_path'].get(), True, 2, False, classifier_params['attributes_to_use'])
    train_obj = MarketDataset(
        config['market_1501_ds']['train_path'].get(), True, 0, False, classifier_params['attributes_to_use'])
    validate_obj = MarketDataset(
        config['market_1501_ds']['train_path'].get(), True, 1, False, classifier_params['attributes_to_use'])

    torch_ds_test = torch.utils.data.DataLoader(test_obj,
                                                batch_size=2, num_workers=8,
                                                collate_fn=collate_fn)
    torch_ds_train = torch.utils.data.DataLoader(train_obj,
                                                 batch_size=2, num_workers=8,
                                                 collate_fn=collate_fn)
    torch_ds_val = torch.utils.data.DataLoader(validate_obj,
                                               batch_size=2, num_workers=8,
                                               collate_fn=collate_fn)

    #test_data = iter(torch_ds_test)
    #print(f"Count of test: {len(test_data)}")
    #train_data = iter(torch_ds_train)
    #print(f"Count of train: {len(train_data)}")
    #validate_data = iter(torch_ds_val)
    #print(f"Count of validate: {len(validate_data)}")

    # Parameters for loop:
    backbone = resnet_fpn_backbone(
        'resnet50', pretrained=True, trainable_layers=0)
    # The second argument is the output being used as a String,
    # "1", "2", "3", or "pool"
    obj = Classifier(classifier_params, "2")
    model = OverallModel(backbone, obj, "2")
    #for param in model.parameters():
    #    param.requires_grad = True
    for k,v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
    model = model.train()
    optimizer = optim.SGD(obj.parameters(), lr=0.00002, momentum=0.9)
    epochs = 20
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    print(next(model.parameters()).device)
    print("CUDA Availability: ", torch.cuda.is_available())
    training_loop(torch_ds_train, torch_ds_val, optimizer, device, model, classifier_params['attributes_to_use'], epochs)
    print('Finished Training')