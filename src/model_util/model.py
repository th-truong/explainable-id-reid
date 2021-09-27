from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision
import confuse
from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
torch.autograd.set_detect_anomaly(True)

class OverallModel(nn.Module):
    def __init__(self, backbone, classifier, output_to_use, device):
        super(OverallModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.output_to_use = output_to_use
        self.device = device
 #adding waits for training layer to conteract the data imbalance
    def loss_layers(self, training):
        if training:
            age_weight = torch.Tensor([2, 0.3373, 1.0606, 2])
            backpack_weight = torch.Tensor([0.6714, 1.9580])
            bag_weight = torch.Tensor([0.6779, 1.9047])
            clothes_weight = torch.Tensor([0.5773, 2])
            handbag_weight = torch.Tensor([0.5668, 2])
            hat_weight = torch.Tensor([0.5137, 2])
            up_weight = torch.Tensor([0.5253, 2])
            down_colours_weight = torch.Tensor([0.2705, 0.7320, 1.0916,
                                   0.6222, 2, 2, 2, 1.3827, 2])
            up_colours_weight = torch.Tensor([0.7447, 1.75, 1, 1.4894, 2, 1.1667, 
                                    0.3608, 0.3608])
#applying cross entropy lose function
            loss_layers = nn.ModuleDict({'age': nn.CrossEntropyLoss(weight = age_weight),
                                        'backpack': nn.BCELoss(weight = backpack_weight),
                                        'bag': nn.BCELoss(),
                                        'clothes': nn.BCELoss(weight = clothes_weight),
                                        'down': nn.BCELoss(),
                                        'down_colours': nn.CrossEntropyLoss(weight = down_colours_weight),
                                        'gender': nn.BCELoss(),
                                        'hair': nn.BCELoss(),
                                        'handbag': nn.BCELoss(weight = handbag_weight),
                                        'hat': nn.BCELoss(weight = hat_weight),
                                        'up': nn.BCELoss(weight = up_weight),
                                        'up_colours': nn.CrossEntropyLoss(weight = up_colours_weight)})
        
        loss_layers = nn.ModuleDict({'age': nn.CrossEntropyLoss(),
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
        return loss_layers

    def forward(self, input, targets):
        back_out = self.backbone(input)
        backbone_out = back_out[self.output_to_use].to(self.device)
        if self.training:
            loss_layers = self.loss_layers(True)
        else:
            loss_layers = self.loss_layers(False)
        output = self.classifier(backbone_out)
        target_outputs = {}
        if len(targets) >= 2:
            for attr in targets[0].keys():
                target_list = []
                for index in range(len(targets)):
                    target_list.append(targets[index][attr])
                target_outputs[attr] = torch.stack(tuple(target_list)).to(self.device)
        else:
            target_outputs = targets[0]
            for attr in target_outputs:
                target_outputs[attr] = target_outputs[attr].view(1).to(self.device)
        output_dict = {}
        if self.training:
            for attribute in target_outputs:
                if str(type(loss_layers[attribute])) == "<class 'torch.nn.modules.loss.BCELoss'>":
                    for attr in target_outputs:
                        target_outputs[attr] = target_outputs[attr].type(
                            torch.float32).to(self.device)
                    for attr in output:
                        output[attr] = output[attr].type(torch.float32).to(self.device)
                    output_dict[attribute] = loss_layers[attribute](
                        output[attribute], target_outputs[attribute].view(output[attribute].shape)).to(self.device)
                else:
                    for attr in target_outputs:
                        target_outputs[attr] = target_outputs[attr].type(
                            torch.long).to(self.device)
                    for attr in output:
                        output[attr] = output[attr].type(torch.float32).to(self.device)
                    output_dict[attribute] = loss_layers[attribute](
                        output[attribute], target_outputs[attribute]).to(self.device)
        return output, output_dict

class Classifier(nn.Module):
    # configurable, default activation layer is ReLU, but can be changed. Default setting for dropout
    # is False (not included), but can do so
    def __init__(self, classifier_params, input, device):
        super(Classifier, self).__init__()
        binary_layers_to_add = classifier_params["hidden_layers_binary"]
        multi_layers_to_add = classifier_params['hidden_layers_multi']
        classifier_layers_to_add = classifier_params["attribute_classification_layers"]
        layers_to_use = classifier_params["attributes_to_use"]
        self.device = device
        self.model_layers = nn.ModuleDict()
        # Adding layers to model in a loop.
        for attr in layers_to_use:
            binary_linear_counter = 0
            multi_linear_counter = 0
            layers = []
            if attr != 'age' and attr != 'down_colours' and attr != 'up_colours':
                for layer in binary_layers_to_add:
                    if layer['type'] == 'linear':
                        if binary_linear_counter == 0:
                            if input == "0":
                                layer['kwargs']['in_features'] = 131072
                            if input == "1":
                                layer['kwargs']['in_features'] = 32768
                            elif input == "2":
                                layer['kwargs']['in_features'] = 8192
                            elif input == "3":
                                layer['kwargs']['in_features'] = 2048
                            elif input == "pool":
                                layer['kwargs']['in_features'] = 512
                        layers.append(nn.Linear(**layer['kwargs']))
                        binary_linear_counter += 1
                    elif layer['type'] == 'relu_activation':
                        layers.append(nn.ReLU(**layer['kwargs']))
                    elif layer['type'] == 'dropout':
                        layers.append(nn.Dropout(**layer['kwargs']))

                for layer in classifier_layers_to_add:
                    if layer['attribute'] == attr:
                        if layer['type'] == 'linear':
                            layers.append(nn.Linear(**layer['kwargs']))
                        if layer['activation'] == 'sigmoid':
                            layers.append(nn.Sigmoid())
                self.model_layers[attr] = nn.Sequential(*layers).to(self.device)
            else:
                for layer in multi_layers_to_add:
                    if layer['type'] == 'linear':
                        if multi_linear_counter == 0:
                            if input == "0":
                                layer['kwargs']['in_features'] = 131072
                            if input == "1":
                                layer['kwargs']['in_features'] = 32768
                            elif input == "2":
                                layer['kwargs']['in_features'] = 8192
                            elif input == "3":
                                layer['kwargs']['in_features'] = 2048
                            elif input == "pool":
                                layer['kwargs']['in_features'] = 512
                        layers.append(nn.Linear(**layer['kwargs']))
                        multi_linear_counter += 1
                    elif layer['type'] == 'relu_activation':
                        layers.append(nn.ReLU(**layer['kwargs']))
                    elif layer['type'] == 'dropout':
                        layers.append(nn.Dropout(**layer['kwargs']))

                for layer in classifier_layers_to_add:
                    if layer['attribute'] == attr:
                        if layer['type'] == 'linear':
                            layers.append(nn.Linear(**layer['kwargs']))
                        if layer['activation'] == 'sigmoid':
                            layers.append(nn.Sigmoid())
                self.model_layers[attr] = nn.Sequential(*layers).to(self.device)
            
    def forward(self, backbone_output):
        x = backbone_output.to(self.device)
        x = torch.flatten(x, start_dim=1).to(self.device)

        attribute_predictions = {}
        for attr_key in list(self.model_layers.keys()):
            y = self.model_layers[attr_key](x).to(self.device)
            attribute_predictions[attr_key] = y

        return attribute_predictions


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    writer = SummaryWriter()
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
#add correct paths to classifer_architecture.yml file
    config = confuse.Configuration('market1501', __name__)
    config.set_file(Path(
        r"C:\\Users\\Div\\Desktop\\Research\\reid\\reid\\explainable-id-reid\\src\\dataset_util\\market1501.yml"))
    cfg = confuse.Configuration('model_architecture', __name__, read=False)
    cfg.set_file(
        "C:\\Users\\Div\\Desktop\\Research\\reid\\reid\\explainable-id-reid\\src\\model_util\\classifier_architecture.yml")
    architecture = cfg.get()

    from processor import MarketDataset
    test_obj = MarketDataset(
        config['market_1501_ds']['test_path'].get(), True, 2, False, architecture['attributes_to_use'])
    train_obj = MarketDataset(
        config['market_1501_ds']['train_path'].get(), True, 0, False, architecture['attributes_to_use'])
    validate_obj = MarketDataset(
        config['market_1501_ds']['train_path'].get(), True, 1, False, architecture['attributes_to_use'])

    torch_ds_test = torch.utils.data.DataLoader(test_obj,
                                                batch_size=architecture['dataloader']['kwargs']['batch_size'],
                                                num_workers=architecture['dataloader']['kwargs']['num_workers'],
                                                collate_fn=collate_fn)
    torch_ds_train = torch.utils.data.DataLoader(train_obj,
                                                 batch_size=architecture['dataloader']['kwargs']['batch_size'],
                                                 num_workers=architecture['dataloader']['kwargs']['num_workers'],
                                                 collate_fn=collate_fn)
    torch_ds_val = torch.utils.data.DataLoader(validate_obj,
                                               batch_size=architecture['dataloader']['kwargs']['batch_size'],
                                               num_workers=architecture['dataloader']['kwargs']['num_workers'],
                                               collate_fn=collate_fn)

    # Parameters for loop:
    backbone = resnet_fpn_backbone(
        **architecture['backbone']['kwargs'])
    backbone = backbone.to(device)
    # The second argument is the output being used as a String,
    # "1", "2", "3", or "pool"
    obj = Classifier(architecture, str(
        architecture['backbone_output_to_use']), device)
    model = OverallModel(backbone, obj, str(
        architecture['backbone_output_to_use']), device)
    # Freezing the FPN layers:
    for k, v in model.named_parameters():
        if "backbone.fpn" in str(k):
            v.requires_grad = False
    for k, v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
    #print(list(model.modules()))

    model = model.train()
    optimizer = optim.Adam(obj.parameters(
    ), lr=architecture['optimizer']['kwargs']['lr'])
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[
                                                     architecture['scheduler']['kwargs']['milestones']], gamma=architecture['scheduler']['kwargs']['gamma'])
    print(next(model.parameters()).device)
    print("CUDA Availability: ", torch.cuda.is_available())
    training_loop(torch_ds_train, torch_ds_val, optimizer, device, model,
                  architecture['attributes_to_use'], scheduler, architecture['epochs'])
    print('Finished Training')
