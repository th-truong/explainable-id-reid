#adding all the imports
import argparse
import torch
import confuse
from pathlib import Path
from dataset_util.processor import MarketDataset
from model_util.model import OverallModel
from model_util.model import Classifier
from model_util.model import collate_fn
from training_util.train_and_valid import validation_loop
from training_util.train_and_valid import training_loop
from training_util.train_and_valid import metric_calculator
from training_util.train_and_valid import identity_matcher
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training or Testing the model.')
    parser.add_argument('Action of train or test', action='store', type=str, help='Choose either test or train (If testing, change batch_size in classifier_architecture.yml to 1!)')
    parser.add_argument('Dataset paths', action='store', type=str, help='Enter the full path to the market1501.yml file')
    parser.add_argument('Model architecture path', action='store', type=str, help='Enter the full path to the classifier_architecture.yml file')
    parser.add_argument('Model paths', action='store', type=str, nargs='?', help='Enter the full path to the model_paths.yml file if testing')
    parser.add_argument('Rank checked for', action='store', type=int, nargs='?', help="Enter an intger to get that Rank of identity matches when testing, 1 being identities returned which match 27/27 predicted attributes.")
    args = vars(parser.parse_args())
    action = args['Action of train or test']
    dataset_paths = args['Dataset paths']
    architecture_path = args['Model architecture path']
    model_paths = args['Model paths']
    rank = args['Rank checked for']

    if action.strip().lower() == "train":
        print("Please ensure that the full path to the train, and test folders, as well as the path to the attributes files are included in the file dataset_util/market1501.yml.")
        writer = SummaryWriter()
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        config = confuse.Configuration('market1501', __name__)
        config.set_file(dataset_paths.strip())
        cfg = confuse.Configuration('model_architecture', __name__, read=False)
        cfg.set_file(architecture_path.strip())
        architecture = cfg.get()

        train_obj = MarketDataset(
            config, config['market_1501_ds']['train_path'].get(), True, 0, False, architecture['attributes_to_use'])
        validate_obj = MarketDataset(
            config, config['market_1501_ds']['train_path'].get(), True, 1, False, architecture['attributes_to_use'])

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
        model = model.train()
        optimizer = optim.Adam(obj.parameters(
        ), lr=architecture['optimizer']['kwargs']['lr'])
        model = model.to(device)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[
                                                        architecture['scheduler']['kwargs']['milestones']], gamma=architecture['scheduler']['kwargs']['gamma'])
        training_loop(torch_ds_train, torch_ds_val, optimizer, device, model,
                  architecture['attributes_to_use'], scheduler, writer, architecture['epochs'])
        
    if action.strip().lower() == "test":
        print("Please ensure that the full path to of all five models, 0, 1, 2, 3, and pool, are incuded in the file model_util/model_paths.yml")
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        config = confuse.Configuration('market1501', __name__)
        config.set_file(dataset_paths.strip())
        cfg = confuse.Configuration('model_architecture', __name__, read=False)
        cfg.set_file(architecture_path.strip())
        architecture = cfg.get()
        model_p = confuse.Configuration('model_paths', __name__)
        model_p.set_file(model_paths.strip())
        paths_m = model_p.get()
        test_obj = MarketDataset(
            config, config['market_1501_ds']['test_path'].get(), True, 2, False, architecture['attributes_to_use'])
        torch_ds_test = torch.utils.data.DataLoader(test_obj,
                                                batch_size=architecture['dataloader']['kwargs']['batch_size'],
                                                num_workers=architecture['dataloader']['kwargs']['num_workers'],
                                                collate_fn=collate_fn)
        backbone = resnet_fpn_backbone(
            **architecture['backbone']['kwargs'])
        backbone = backbone.to(device)
        # Creating Classifier objects for each model
        # The second argument is the output being used as a String,
        # "1", "2", "3", or "pool"
        obj0 = Classifier(architecture, "0", device)
        obj1 = Classifier(architecture, "1", device)
        obj2 = Classifier(architecture, "2", device)
        obj3 = Classifier(architecture, "3", device)
        objpool = Classifier(architecture, "pool", device)

        # Creation of a list of objects
        objs = [objpool, obj0, obj1, obj2, obj3]

        # Retrieving the state dicts for all the models
        model_pool = torch.load(paths_m['model_paths']['model_outputpool'])
        model2 = torch.load(paths_m['model_paths']['model_output2'], map_location = torch.device('cpu'))
        model3 = torch.load(paths_m['model_paths']['model_output3'], map_location = torch.device('cpu'))
        model0 = torch.load(paths_m['model_paths']['model_output0'], map_location = torch.device('cpu'))
        model1 = torch.load(paths_m['model_paths']['model_output1'], map_location = torch.device('cpu'))

        # Creation of a list of model dicts
        models = [model_pool, model0, model1, model2, model3]

        # Calling the validation_loop
        predictions, real = validation_loop(torch_ds_test, False, device, backbone, objs, models, True, architecture['attributes_to_use'], test_obj.attribute_market, rank, None, 0)

        true = 0
        false = 0
        
        for i in range(len(predictions)):
            if int(real[i][0]) in predictions[i]:
                true += 1
            else: 
                false += 1
        #Printing ranks 
        print(f"Rank {rank} matching on the specified models is: {(round((true/(true+false)), 4)) * 100}% ({true} true matches and {false} false matches).")
