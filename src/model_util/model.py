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

class OverallModel(nn.Module):
    def __init__(self, backbone, classifier, output_to_use):
        super(OverallModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.output_to_use = output_to_use

    def forward(self, input, targets):
        back_out = self.backbone(input)
        output = self.classifier(back_out, self.output_to_use)

        return output, targets

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
                    activation = nn.Softmax(dim = None)
                elif layer['activation'] == 'sigmoid':
                    activation = nn.Sigmoid()
                self.attribute_layers_dict[layer['attribute']] = nn.Sequential(linear, activation)
        print(self.attribute_layers_dict)

    def forward(self, backbone_output):
        x = backbone_output
        x = torch.flatten(x, start_dim = 1)
        for layer in self.layers:
            x = layer(x)
        print(x.shape)

        attribute_predictions = {}
        for attr_key in list(self.attribute_layers_dict.keys()):
            attribute_predictions[attr_key] = self.attribute_layers_dict[attr_key](x)
        return attribute_predictions

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    config = confuse.Configuration('market1501', __name__)
    config.set_file(Path(r"C:\\Users\\Div\\Desktop\\Research\\reid\\reid\\explainable-id-reid\\src\\dataset_util\\market1501.yml"))
    from processor import MarketDataset
    test_obj = MarketDataset(
        config['market_1501_ds']['test_path'].get(), True, 2)
    train_obj = MarketDataset(
        config['market_1501_ds']['train_path'].get(), True, 0)
    validate_obj = MarketDataset(
        config['market_1501_ds']['train_path'].get(), True, 1)

    bad = 0
    good = 0
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
    print(f"Count of train: {len(train_data)}")
    validate_data = iter(torch_ds_val)
    imgs = []
    attrs = []
    for img, attr in train_data:
        imgs.append(img)
        attrs.append(attr)
        break
    print(f"Count of validate: {len(validate_data)}")
    #print(imgs[1][1].shape)

    ####TESTING HERE
    backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=0)
    cfg = confuse.Configuration('model_architecture', __name__, read= False)
    cfg.set_file("C:\\Users\\Div\\Desktop\\Research\\reid\\reid\\explainable-id-reid\\src\\model_util\\classifier_architecture.yml")
    classifier_params = cfg.get()
    # The second argument is the output being used as a String,
    # "1", "2", "3", or "pool"
    obj = Classifier(classifier_params, "3")

    model = OverallModel(backbone, obj, "3")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(imgs[0][0].shape)
    out = model.forward(imgs[0], attrs[0])
    print(out)
    criteria = nn.CrossEntropyLoss()
    # BINARY = bceloss
    optimizer = optim.SGD(obj.parameters(), lr=0.001, momentum=0.9)
    epochs = 20
    for i in range(epochs):
        for data in iter(torch_ds_train):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            optimizer.zero_grad()
            # This doesn't work because inputs is a tuple.
            images = list(image.to(device) for image in inputs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in labels]
            outputs = backbone(inputs)
            a = obj.forward(outputs['3'])
            loss = criteria(a, labels)
            loss.backward()
            optimizer.step()
        torch.save({'model': obj.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, str(i).zfill(3) + "resnet50_fpn_frcnn_full.tar")


#images = list(image.to(device) for image in images)
#            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    print('Finished Training')

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
    cfg = confuse.Configuration('model_architecture', __name__, read= False)
    cfg.set_file("C:\\Users\\Div\\Desktop\\Research\\reid\\reid\\explainable-id-reid\\src\\model_util\\classifier_architecture.yml")
    classifier_params = cfg.get()
    # The second argument is the output being used as a String,
    # "1", "2", "3", or "pool"
    obj = Classifier(classifier_params, "2")
    a = obj.forward(out['2'])
    print(a)