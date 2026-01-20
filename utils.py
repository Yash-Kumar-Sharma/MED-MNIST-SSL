import torchvision
import os
import torch
import torch.nn as nn
from medmnist import RetinaMNIST, BreastMNIST, PneumoniaMNIST, BloodMNIST, DermaMNIST, PathMNIST, TissueMNIST, OrganCMNIST, OrganSMNIST, OrganAMNIST, OCTMNIST
 
def alignment_loss(embeddings, positive_pairs):
    """Minimizes distance between positive pairs."""
    z_i, z_j = embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]
    return (z_i - z_j).pow(2).sum(dim=1).mean()

def uniformity_loss(embeddings, t=2.0):
    """Encourages embeddings to be uniformly distributed."""
    sq_pdist = torch.pdist(embeddings, p=2).pow(2)
    return torch.log(torch.exp(-t * sq_pdist).mean())


def stack(data, dim=0):
  shape = data[0].shape  # need to handle empty list
  shape = shape[:dim] + (len(data),) + shape[dim:]
  x = torch.cat(data, dim=dim)
  x = x.reshape(shape)
  # need to handle case where dim=-1
  # which is not handled here yet
  # but can be done with transposition
  return x


def GetCheckpointDir(config, train_mode):

    checkpoint_path = ""
    
    if(config.dataset.imbalance):
        checkpoint_path = os.path.join("results", "imbalance_" + train_mode, config.dataset.imbalance_type, config.model.name + config.model.mode, config.dataset.name, config.backbone.name)
    else:
        checkpoint_path = os.path.join("results", "balance_" + train_mode, config.model.name + config.model.mode, config.dataset.name, config.backbone.name)

    return checkpoint_path

def GetTensorboardDir(config, train_mode):

    result_folder = ""
    if(config.dataset.imbalance):
        result_folder = os.path.join("results", train_mode + "_logs", "imbalance", config.dataset.imbalance_type, config.model.name + config.model.mode, config.dataset.name)
    else:
        result_folder = os.path.join("results", train_mode + "_logs", "balanced", config.model.name + config.model.mode, config.dataset.name)

    return result_folder


def GetBackbone(backbone_name, dataset_name, prune=False, num_class=10, one_color_channel = False):
    match backbone_name:
        #case "resnet20":
            #return ResnetVersions.resnet20_cifar()
        #case "resnet32":
            #return ResnetVersions.resnet32_cifar()
        case "resnet18":

            if("Mnist" in dataset_name and not one_color_channel):
                print("ResNet18 with (3*3) kernel")
                net = torchvision.models.resnet18(weights = None)
                net.maxpool = nn.Identity()
                net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
                net.fc = nn.Identity()
            
            elif("Mnist" in dataset_name and one_color_channel):
                print("ResNet18 with (3*3) kernel with 1-color channel")
                net = torchvision.models.resnet18(weights = None)
                net.maxpool = nn.Identity()
                net.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
                net.fc = nn.Identity()
            else:
                print("ResNet18 with (7*7) kernel")
                net = torchvision.models.resnet18(weights = None)
                net.fc = nn.Identity()

            return net
        
        case "resnet50":
            if("Mnist" in dataset_name and not one_color_channel):
                print("Resent-50 with (3*3) kernel")
                net = torchvision.models.resnet50(weights = None)
                net.maxpool = nn.Identity()
                net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
                net.fc = nn.Identity()
            
            elif("Mnist" in dataset_name and one_color_channel):
                print("ResNet50 with (3*3) kernel with 1-color channel")
                net = torchvision.models.resnet50(weights = None)
                net.maxpool = nn.Identity()
                net.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
                net.fc = nn.Identity()
            else:
                print("ResNet50 with (7*7) kernel")
                net = torchvision.models.resnet50(weights = None)
                net.fc = nn.Identity()
            
            return net

def Get_Dataset(dataset_name, data_dir, test_transform, batch_size):
    match dataset_name:
        
        case "RetinaMnist":
            train_data = RetinaMNIST(root=data_dir, split="train", transform=test_transform, download=True)
            test_data = RetinaMNIST(root=data_dir, split="test", transform=test_transform, download=True)

        case "BreastMnist":
            train_data = BreastMNIST(root=data_dir, split="train", transform=test_transform, download=True)
            test_data = BreastMNIST(root=data_dir, split="test", transform=test_transform, download=True)
        
        case "PneumoniaMnist":
            train_data = PneumoniaMNIST(root=data_dir, split="train", transform=test_transform, download=True)
            test_data = PneumoniaMNIST(root=data_dir, split="test", transform=test_transform, download=True)
        
        case "BloodMnist":
            train_data = BloodMNIST(root=data_dir, split="train", transform=test_transform, download=True)
            test_data = BloodMNIST(root=data_dir, split="test", transform=test_transform, download=True)
        
        case "DermaMnist":
            train_data = DermaMNIST(root=data_dir, split="train", transform=test_transform, download=True)
            test_data = DermaMNIST(root=data_dir, split="test", transform=test_transform, download=True)
    
        case "PathMnist":
            train_data = PathMNIST(root=data_dir, split="train", transform=test_transform, download=True)
            test_data = PathMNIST(root=data_dir, split="test", transform=test_transform, download=True)
        
        case "TissueMnist":
            train_data = TissueMNIST(root=data_dir, split="train", transform=test_transform, download=True)
            test_data = TissueMNIST(root=data_dir, split="test", transform=test_transform, download=True)
        
        case "OrganCMnist":
            train_data = OrganCMNIST(root=data_dir, split="train", transform=test_transform, download=True)
            test_data = OrganCMNIST(root=data_dir, split="test", transform=test_transform, download=True)
    
        case "OrganSMnist":
            train_data = OrganSMNIST(root=data_dir, split="train", transform=test_transform, download=True)
            test_data = OrganSMNIST(root=data_dir, split="test", transform=test_transform, download=True)
    
        case "OrganAMnist":
            train_data = OrganAMNIST(root=data_dir, split="train", transform=test_transform, download=True)
            test_data = OrganAMNIST(root=data_dir, split="test", transform=test_transform, download=True)
        
        case "OCTMnist":
            train_data = OCTMNIST(root=data_dir, split="train", transform=test_transform, download=True)
            test_data = OCTMNIST(root=data_dir, split="test", transform=test_transform, download=True)
    
    memory_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return memory_data_loader, test_data_loader
