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
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from model_util.model import collate_fn
from model_util.model import OverallModel
from model_util.model import Classifier
from model_util.model import validation_loop
from model_util.model import metric_calculator
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    print("HIHIHIHi")
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
    backbone = resnet_fpn_backbone(
        **architecture['backbone']['kwargs'])
    backbone = backbone.to(device)
    # The second argument is the output being used as a String,
    # "1", "2", "3", or "pool"
    from model_util.model import Classifier
    from model_util.model import OverallModel
    obj = Classifier(architecture, str(
        architecture['backbone_output_to_use']), device)
    model = OverallModel(backbone, obj, str(
        architecture['backbone_output_to_use']), device)
    for k, v in model.named_parameters():
        if "backbone.fpn" in str(k):
            v.requires_grad = False
    for k, v in model.named_parameters():
        print(k)
    optimizer = optim.Adam(obj.parameters(
    ), lr=architecture['optimizer']['kwargs']['lr'])
    model = model.to(device)
    saved_model = torch.load(
        r"C:\\Users\\netra\\GithubEncm369\\reid\\explainable-id-reid\\src\\model_util\\run_back3\\099resnet50_fpn_frcnn_full.tar", map_location=torch.device('cpu'))
    model.load_state_dict(saved_model['model'])
    optimizer.load_state_dict(saved_model['optimizer'])
    validation_loop(torch_ds_test, True, device, model,
                    architecture['attributes_to_use'], validate_obj.attribute_market, 0)
