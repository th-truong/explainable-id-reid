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
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
torch.autograd.set_detect_anomaly(True)


def collate_fn(batch):
    return tuple(zip(*batch))


writer = SummaryWriter()


def validation_loop(validation_ds, device, model):
    with torch.no_grad():
        for data in tqdm(iter(validation_ds)):
            inputs, labels = data
            images = list(image.to(device) for image in inputs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in labels]
            images = torch.stack(images)
            output = model(images, targets)
            if output == False:
                continue
            loss = torch.Tensor([0])
            for attr in output:
                loss = loss + output[attr].item()
                print(
                    f"Validation: Attr: {attr} || Loss: {output[attr].item()}")
                writer.add_scalar(
                    f"{attr} Loss/Validation", output[attr].item())
            print(f"Validation Overall Loss: {loss}")
            writer.add_scalar("Validation Overall Loss", loss)


def training_loop(torch_ds, validation_ds, optimizer, device, model, epochs=20):
    for i in range(epochs):
        for data in tqdm(iter(torch_ds)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            optimizer.zero_grad()
            images = list(image.to(device) for image in inputs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in labels]
            images = torch.stack(images)
            output = model(images, targets)
            break
            if output == False:
                continue
            print("\n\n")
            loss = torch.Tensor([0])
            loss.requires_grad = True
            for attr in output:
                loss = loss + output[attr].item()
                print(f"Training: Attr: {attr} || Loss: {output[attr].item()}")
                writer.add_scalar(f"{attr} Loss/train", output[attr].item())
            print(f"Training: Overall Loss: {loss}")
            writer.add_scalar("Training Overall Loss", loss)
            loss.backward()
            optimizer.step()
        print("DONE")
        break
        validation_loop(validation_ds, device, model)
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, str(i).zfill(3) + "resnet50_fpn_frcnn_full.tar")
    writer.flush()
    writer.close()


if __name__ == "__main__":
    config = confuse.Configuration('market1501', __name__)
    config.set_file(Path(
        r"C:\\Users\\netra\\GithubEncm369\\reid\\explainable-id-reid\\src\\dataset_util\\market1501.yml"))
    from dataset_util.processor import MarketDataset
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
    #test_data = iter(torch_ds_test)
    #print(f"Count of test: {len(test_data)}")
    # train_data = iter(torch_ds_train)
    # attrs = []
    # for img, attr in torch_ds_train:
    #     attrs.append(attr)
    # print(attrs[len(attrs) - 1])
    # print(f"Count of train: {len(train_data)}")
    # validate_data = iter(torch_ds_val)
    # print(f"Count of validate: {len(validate_data)}")
    # Parameters for loop:
    backbone = resnet_fpn_backbone(
        'resnet50', pretrained=True, trainable_layers=0)
    cfg = confuse.Configuration('model_architecture', __name__, read=False)
    cfg.set_file(
        "C:\\Users\\netra\\GithubEncm369\\reid\\explainable-id-reid\\src\\model_util\\classifier_architecture.yml")
    classifier_params = cfg.get()
    # The second argument is the output being used as a String,
    # "1", "2", "3", or "pool"
    from model_util.model2 import Classifier
    from model_util.model2 import OverallModel
    obj = Classifier(classifier_params, "3")
    model = OverallModel(backbone, obj, "3")
    # for param in model.parameters():
    #    param.requires_grad = True
    for k, v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
    model = model.train()
    optimizer = optim.SGD(obj.parameters(), lr=0.001, momentum=0.9)
    epochs = 20
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    training_loop(torch_ds_train, torch_ds_val,
                  optimizer, device, model, epochs)
    print('Finished Training')
