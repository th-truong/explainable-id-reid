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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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
                target_outputs[attr] = torch.stack(tuple(target_list))
        else:
            target_outputs = targets[0]
            for attr in target_outputs:
                target_outputs[attr] = target_outputs[attr].view(1)
        output_dict = {}
        if self.training:
            for attribute in target_outputs:
                if str(type(loss_layers[attribute])) == "<class 'torch.nn.modules.loss.BCELoss'>":
                    for attr in target_outputs:
                        target_outputs[attr] = target_outputs[attr].type(
                            torch.float32)
                    for attr in output:
                        output[attr] = output[attr].type(torch.float32)
                    output_dict[attribute] = loss_layers[attribute](
                        output[attribute], target_outputs[attribute].view(output[attribute].shape))
                else:
                    for attr in target_outputs:
                        target_outputs[attr] = target_outputs[attr].type(
                            torch.long)
                    for attr in output:
                        output[attr] = output[attr].type(torch.float32)
                    output_dict[attribute] = loss_layers[attribute](
                        output[attribute], target_outputs[attribute])
        return output, output_dict


def metric_calculator(pred_and_true, classifier_params, epoch, device):
    #print(pred_and_true)
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
                to_add = torch.argmax(t[0][attr], dim=1)
                predictions[attr].append(to_add)
                real[attr].append(t[1][attr])
            else:
                predictions[attr].append(t[0][attr])
                real[attr].append(t[1][attr])

    for attr in predictions:
        predictions[attr] = torch.stack(tuple(predictions[attr])).to(device)

    for attr in real:
        real[attr] = torch.stack(tuple(real[attr])).to(device)
    for attr in metrics:
        if attr != 'age' and attr != 'down_colours' and attr != 'up_colours':
            conf = confusion_matrix(torch.flatten(real[attr].cpu()), torch.round(
                torch.flatten(predictions[attr].cpu()).type(torch.float)))
            tn, fp, fn, tp = conf.ravel()
            precision, recall, _, _ = precision_recall_fscore_support(torch.flatten(real[attr].cpu(
            )), torch.round(torch.flatten(predictions[attr].cpu()).type(torch.float)), average='binary')
            accuracy = accuracy_score(torch.flatten(real[attr].cpu()), torch.round(
                torch.flatten(predictions[attr].cpu()).type(torch.float)))
            metrics[attr] = {'precision': precision, 'recall': recall, 'accuracy': accuracy,
                             'sensitivity': tp/(tp + fn), 'specificity': tn/(tn + fp)}
        else:
            labels = []
            if attr == 'age':
                labels = [0,1,2,3]
            elif attr == 'up_colours':
                labels = [0,1,2,3,4,5,6,7]
            elif attr == 'down_colours':
                labels = [0,1,2,3,4,5,6,7,8]
            conf = confusion_matrix(torch.flatten(real[attr].cpu()), torch.round(
                torch.flatten(predictions[attr].cpu()).type(torch.float)), labels = labels)
            print(conf)
            print(classification_report(torch.flatten(real[attr].cpu()), torch.round(
                torch.flatten(predictions[attr].cpu()).type(torch.float)), labels = labels, digits=3))
            precision, recall, _, _ = precision_recall_fscore_support(torch.flatten(real[attr].cpu(
            )), torch.round(torch.flatten(predictions[attr].cpu()).type(torch.float)), average='macro')
            metrics[attr] = {'precision': precision, 'recall': recall}
    return metrics


class Classifier(nn.Module):
    # configurable, default activation layer is ReLU, but can be changed. Default setting for dropout
    # is False (not included), but can do so
    def __init__(self, classifier_params, backbone_output, device):
        super(Classifier, self).__init__()
        binary_layers_to_add = classifier_params["hidden_layers_binary"]
        multi_layers_to_add = classifier_params['hidden_layers_multi']
        classifier_layers_to_add = classifier_params["attribute_classification_layers"]
        layers_to_use = classifier_params["attributes_to_use"]
        self.device = device
        self.model_layers = nn.ModuleDict()
        for attr in layers_to_use:
            binary_linear_counter = 0
            multi_linear_counter = 0
            layers = []
            if attr != 'age' and attr != 'down_colours' and attr != 'up_colours':
                for layer in binary_layers_to_add:
                    if layer['type'] == 'linear':
                        if binary_linear_counter == 0:
                            if backbone_output == "0":
                                layer['kwargs']['in_features'] = 131072
                            if backbone_output == "1":
                                layer['kwargs']['in_features'] = 32768
                            elif backbone_output == "2":
                                layer['kwargs']['in_features'] = 8192
                            elif backbone_output == "3":
                                layer['kwargs']['in_features'] = 2048
                            elif backbone_output == "pool":
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
                self.model_layers[attr] = nn.Sequential(*layers)
            else:
                for layer in multi_layers_to_add:
                    if layer['type'] == 'linear':
                        if multi_linear_counter == 0:
                            if backbone_output == "0":
                                layer['kwargs']['in_features'] = 131072
                            if backbone_output == "1":
                                layer['kwargs']['in_features'] = 32768
                            elif backbone_output == "2":
                                layer['kwargs']['in_features'] = 8192
                            elif backbone_output == "3":
                                layer['kwargs']['in_features'] = 2048
                            elif backbone_output == "pool":
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
                self.model_layers[attr] = nn.Sequential(*layers)
        print(self.model_layers)
            
    def forward(self, backbone_output):
        x = backbone_output
        x = torch.flatten(x, start_dim=1)

        attribute_predictions = {}
        for attr_key in list(self.model_layers.keys()):
            y = self.model_layers[attr_key](x)
            attribute_predictions[attr_key] = y

        return attribute_predictions


def collate_fn(batch):
    return tuple(zip(*batch))


def validation_loop(validation_ds, testing, device, model, classifier_params, data_frame, epoch):
    predictions_and_real = []
    model.eval()
    predicted_identities = []
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
                target_attrs[attr] = torch.stack(tuple(target_list)).to(device)
            images = torch.stack(images)
            images = images.to(device)
            output, _ = model(images, targets)
            if data_frame is not None:
                predicted_identities.append(identity_matcher(output, data_frame))
            predictions_and_real.append((output, target_attrs))
        metrics = metric_calculator(
            predictions_and_real, classifier_params, epoch, device)
        for attr in metrics:
            for metric in metrics[attr]:
                if not testing:
                    writer.add_scalar(f"{attr} {metric}",
                                    metrics[attr][metric], epoch)
                else:
                    print(f"{attr} {metric}: {round(metrics[attr][metric], 4)}")
    if testing:
        return predicted_identities


def training_loop(torch_ds, validation_ds, optimizer, device, model, classifier_params, scheduler, epochs=20):
    step_counter = 0
    for i in range(epochs):
        for data in tqdm(iter(torch_ds)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            optimizer.zero_grad()
            images = list(image.to(device) for image in inputs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in labels]
            images = torch.stack(images)
            images = images.to(device)
            _, output = model(images, targets)
            total_loss = sum([attr_loss for attr_loss in output.values()])
            for attr in output:
                writer.add_scalar(f"{attr} Loss/train",
                                  output[attr].item(), step_counter)
            writer.add_scalar("Training Overall Loss",
                              total_loss, step_counter)
            total_loss.backward()
            optimizer.step()
            step_counter += 1
        validation_loop(validation_ds, False, device, model, classifier_params, None, i)
        scheduler.step()
        model.train()
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, str(i).zfill(3) + "resnet50_fpn_frcnn_full.tar")
    writer.flush()
    writer.close()


def identity_matcher(attribute_predictions, data_frame):
    dict_new = {}
    for attr in list(attribute_predictions.keys()):
        if attr == 'age' or attr == 'down_colours' or attr == 'up_colours':
            if attr == 'age':
                dict_new[attr] = [int(torch.argmax(attribute_predictions[attr]).cpu().item()) + 1]
            else:
                dict_new[attr] = [int(torch.argmax(attribute_predictions[attr]).cpu().item())]
        else:
            dict_new[attr] = [int(torch.round(attribute_predictions[attr]).cpu().item())]

    down_colours = ['downblack', 'downblue', 'downbrown', 'downgray', 'downgreen', 'downpink', 'downpurple', 'downwhite', 'downyellow']
    up_colours = ['upblack', 'upblue', 'upgray', 'upgreen', 'uppurple', 'upred', 'upwhite', 'upyellow']
    for attr in list(dict_new.keys()):
        if attr == 'down_colours':
                down_col = dict_new[attr][0]
                key = down_colours[down_col]
                dict_new[key] = [1]
                for col in down_colours:
                        if col != key:
                                dict_new[col] = [0]
                del dict_new[attr]
        if attr == 'up_colours':
                up_col = dict_new[attr][0]
                key = up_colours[up_col]
                dict_new[key] = [1]
                for col in up_colours:
                        if col != key:
                                dict_new[col] = [0]
                del dict_new[attr]
    dict_attr = {}
    for attr in sorted(list(dict_new.keys())):
        dict_attr[attr] = dict_new[attr]
    row_attr = pd.DataFrame.from_dict(dict_attr)
    row_matches = []
    for idx, row in data_frame.iterrows():  
        attribute_matches = 0
        for col in row_attr.columns:
                if int(row[col]) == int(row_attr[col]):
                        attribute_matches += 1
        if attribute_matches == len(list(row_attr.columns)):
                row_matches.append(row['image_index'][0])
    return row_matches


if __name__ == "__main__":
    writer = SummaryWriter()
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

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
