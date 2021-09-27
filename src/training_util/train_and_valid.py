from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from model_util.model import OverallModel
import os
import sys
sys.path.append(os.getcwd() + "/model_util")

#metric calculator for each prediction
def metric_calculator(pred_and_true, classifier_params, epoch, device):
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
            accuracy = accuracy_score(torch.flatten(real[attr].cpu(
            )), torch.round(torch.flatten(predictions[attr].cpu()).type(torch.float)))
            metrics[attr] = {'accuracy': accuracy, 'precision': precision, 'recall': recall}
    return metrics

def identity_matcher(attribute_predictions, data_frame, rank):
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
        cols = list(row_attr.columns)
        for col in cols:
            if int(row[col]) == int(row_attr[col]):
                attribute_matches += 1
        if attribute_matches >= (len(cols) - rank + 1):
                row_matches.append(int(row['image_index'][0]))
    return row_matches

def validation_loop(validation_ds, need_to_write, device, backbone, objs, model_s, multiple, classifier_params, data_frame, rank, epoch):
    predictions_and_real = []
    predicted_identities = []
    real_identities = []
    if multiple == False:
        model_s.eval()
    with torch.no_grad():
        for data in tqdm(iter(validation_ds)):
            inputs, labels, image_index = data
            real_identities.append(image_index)
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
            if multiple == True:
                model = OverallModel(backbone, objs[1], "0", device)
                model.load_state_dict(model_s[1]['model'])
                model.eval()
                output0, _ = model(images, targets)
                model = OverallModel(backbone, objs[2], "1", device)
                model.load_state_dict(model_s[2]['model'])
                model.eval()
                output1, _ = model(images, targets)
                model = OverallModel(backbone, objs[3], "2", device)
                model.load_state_dict(model_s[3]['model'])
                model.eval()
                output2, _ = model(images, targets)
                model = OverallModel(backbone, objs[4], "3", device)
                model.load_state_dict(model_s[4]['model'])
                model.eval()
                output3, _ = model(images, targets)
                model = OverallModel(backbone, objs[0], "pool", device)
                model.load_state_dict(model_s[0]['model'])
                model.eval()
                outputpool, _ = model(images, targets)
            else:
                output_ovr = model_s(images, targets)
            predictions = {}
            for key in list(output0.keys()):
                if key == 'age' or key == 'backpack' or key == 'clothes' or key == 'hair':
                    if multiple == True:
                        predictions[key] = output1[key]
                    else: 
                        predictions[key] = output_ovr[key]
                elif key == 'down' or key == 'hat':
                    if multiple == True:
                        predictions[key] = output2[key]
                    else: 
                        predictions[key] = output_ovr[key]
                elif key == 'bag' or key == 'down_colours' or key == 'handbag' or key == 'up_colours':
                    if multiple == True:
                        predictions[key] = output3[key]
                    else: 
                        predictions[key] = output_ovr[key]
                elif key == 'up':
                    if multiple == True:
                        predictions[key] = outputpool[key]
                    else: 
                        predictions[key] = output_ovr[key]
                else:
                    if multiple == True:
                        predictions[key] = output0[key]
                    else: 
                        predictions[key] = output_ovr[key]
            
            if data_frame is not None:
                predicted_identities.append(identity_matcher(predictions, data_frame, rank))
            if need_to_write:
                predictions_and_real.append((predictions, target_attrs))
        if need_to_write:
            metrics = metric_calculator(
                predictions_and_real, classifier_params, epoch, device)
            for attr in metrics:
                for metric in metrics[attr]:
                    writer.add_scalar(f"{attr} {metric}",
                                    metrics[attr][metric], epoch)
    if not need_to_write:
        return predicted_identities, real_identities


def training_loop(torch_ds, validation_ds, optimizer, device, model, classifier_params, scheduler, epochs=20):
    step_counter = 0
    for i in range(epochs):
        for data in tqdm(iter(torch_ds)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
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
        validation_loop(validation_ds, True, device, None, None, model, False, classifier_params, None, i)
        scheduler.step()
        model.train()
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, str(i).zfill(3) + "resnet50_fpn_frcnn_full.tar")
    writer.flush()
    writer.close()