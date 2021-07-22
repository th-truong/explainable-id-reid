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


def collate_fn(batch):
    return tuple(zip(*batch))


def validation_loop(validation_ds, device, model, classifier_params, epoch):
    predictions_and_real = []
    model.eval()
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
            if model.training == True:
                print("Training")
            else:
                print("Eval")
            predictions_and_real.append((output, target_attrs))
        from model_util.model2 import metric_calculator
        metrics = metric_calculator(
            predictions_and_real, classifier_params, epoch, device)
        for attr in metrics:
            for metric in metrics[attr]:
                writer.add_scalar(f"{attr} {metric}",
                                  metrics[attr][metric], epoch)


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
            images = images.to(device)
            print(torch.cuda.is_available())
            _, output = model(images, targets)
            if model.training == True:
                print("Training")
            else:
                print("Eval")
            if output == False:
                continue
            total_loss = sum([attr_loss for attr_loss in output.values()])
            for attr in output:
                #    loss = loss + output[attr].item()
                #print(f"Training: Attr: {attr} || Loss: {output[attr].item()}")
                writer.add_scalar(f"{attr} Loss/train",
                                  output[attr].item(), step_counter)
                print(attr)
            print(f"Training: Overall Loss: {total_loss}")
            writer.add_scalar("Training Overall Loss",
                              total_loss, step_counter)
            total_loss.backward()
            optimizer.step()
            step_counter += 1
        print(step_counter)
        print("DONE")
        validation_loop(validation_ds, device, model, classifier_params, i)
        model.train()
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, str(i).zfill(3) + "resnet50_fpn_frcnn_full.tar")
    writer.flush()
    writer.close()


if __name__ == "__main__":
    writer = SummaryWriter()
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    config = confuse.Configuration('market1501', __name__)
    config.set_file(Path(
        r"C:\\Users\\netra\\GithubEncm369\\reid\\explainable-id-reid\\src\\dataset_util\\market1501.yml"))
    cfg = confuse.Configuration('model_architecture', __name__, read=False)
    cfg.set_file(
        "C:\\Users\\netra\\GithubEncm369\\reid\\explainable-id-reid\\src\\model_util\\classifier_architecture.yml")
    architecture = cfg.get()

    from dataset_util.processor import MarketDataset
    test_obj = MarketDataset(
        config['market_1501_ds']['test_path'].get(), True, 2, False, architecture['attributes_to_use'])
    train_obj = MarketDataset(
        config['market_1501_ds']['train_path'].get(), True, 0, False, architecture['attributes_to_use'])
    validate_obj = MarketDataset(
        config['market_1501_ds']['train_path'].get(), True, 1, False, architecture['attributes_to_use'])

    print(len(train_obj.identities))
    print(len(validate_obj.identities))

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

    #test_data = iter(torch_ds_test)
    #print(f"Count of test: {len(test_data)}")
    #train_data = iter(torch_ds_train)
    #print(f"Count of train: {len(train_data)}")
    #validate_data = iter(torch_ds_val)
    #print(f"Count of validate: {len(validate_data)}")

    # Parameters for loop:
    backbone = resnet_fpn_backbone(
        **architecture['backbone']['kwargs'])
    backbone = backbone.to(device)
    # The second argument is the output being used as a String,
    # "1", "2", "3", or "pool"
    from model_util.model2 import Classifier
    from model_util.model2 import OverallModel
    obj = Classifier(architecture, str(
        architecture['backbone_output_to_use']), device)
    model = OverallModel(backbone, obj, str(
        architecture['backbone_output_to_use']), device)
    # for param in model.parameters():
    #    param.requires_grad = True
    for k, v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
    model = model.train()
    optimizer = optim.SGD(obj.parameters(
    ), lr=architecture['optimizer']['kwargs']['lr'], momentum=architecture['optimizer']['kwargs']['momentum'])
    model = model.to(device)
    print(next(model.parameters()).device)
    print("CUDA Availability: ", torch.cuda.is_available())
    training_loop(torch_ds_train, torch_ds_val, optimizer, device,
                  model, architecture['attributes_to_use'], architecture['epochs'])
    print('Finished Training')
