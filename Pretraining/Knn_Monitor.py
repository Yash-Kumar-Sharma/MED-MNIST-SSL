import torch
import torchvision
import torch.nn.functional as F
from torchvision.datasets.fakedata import transforms
from tqdm import tqdm
from Preprocess.Augmentations import model_transforms,GetNormalize
import utils
import numpy
# code copied from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=RI1Y8bSImD7N
class Knn_Monitor():
    
    def __init__(self, config) -> None:     
        self.k = config.training.knn_k
        self.c = config.dataset.num_classes
        self.batch_size = config.training.batch_size
        
        if(config.dataset.name == "TinyImagenet"):
            self.data_list = config.dataset.data_list
            self.val_list = config.dataset.val_list
        else:
            self.data_list = None
            self.val_list = None

        if("Mnist" in config.dataset.name):
            convert_channel = config.dataset.convert_channel
        else:
            convert_channel = False


        transform = model_transforms(config.dataset.name, config.dataset.image_size, convert_channel=convert_channel)
        test_transform, train_transform = transform.GetTransform()
        #self.normalize = GetNormalize(config.dataset.name)
        #test_transform = torchvision.transforms.Compose([
        #    torchvision.transforms.ToTensor(),
        #    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])        
        self.memory_data_loader, self.test_data_loader = utils.Get_Dataset(dataset_name = config.dataset.name, data_dir = config.dataset.data_dir, 
                                                                           test_transform = test_transform, batch_size = self.batch_size,
                                                                           data_list=self.data_list, val_list = self.val_list)
        
        '''
        memory_data = torchvision.datasets.CIFAR10(root=config.dataset.data_dir, train=True, transform=test_transform, download=True)
        self.memory_data_loader = torch.utils.data.DataLoader(memory_data, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True)
        test_data = torchvision.datasets.CIFAR10(root=config.dataset.data_dir, train=False, transform=test_transform, download=True)
        self.test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True)
        '''
    # test using a knn monitor
    def test(self, net, dataset_name, k=200, t=0.1, hide_progress=False):

        net.eval()
        if(dataset_name == "TinyImagenet"):
            data_labels = [label for (image, label) in self.memory_data_loader]
            classes = len(data_labels)
        elif("Mnist" in dataset_name):
            classes = self.c
        else:
            classes = len(self.memory_data_loader.dataset.classes)
        '''
        elif(dataset_name == "RetinaMnist"):
            classes = 5
        elif(dataset_name == "BreastMnist" or dataset_name == "PneumoniaMnist"):
            classes = 2
        elif(dataset_name == "BloodMnist"):
            classes = 8
        elif(dataset_name == "DermaMnist"):
            classes = 7
        '''
        
        total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
        with torch.no_grad():
            # generate feature bank
            if(dataset_name == "TinyImagenet"):
                for data in tqdm(self.memory_data_loader, desc='Feature extracting', leave=False, disable=hide_progress):
                    feature = net(data[0][0].cuda(non_blocking=True))
                    feature = F.normalize(feature, dim=1)
                    feature_bank.append(feature)
            else:
                for data,targets in tqdm(self.memory_data_loader, desc='Feature extracting', leave=False, disable=hide_progress):
                    #import pdb
                    #pdb.set_trace()
                    #if(dataset_name == "BreastMnist"):
                    #    data = data.repeat(1,3,1,1)
                    #    data = self.normalize(data)
                    feature = net(data.cuda(non_blocking=True))
                    feature = F.normalize(feature, dim=1)
                    feature_bank.append(feature)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            if(dataset_name == "TinyImagenet"):
                targets = data_labels
                targets = list(numpy.concatenate(targets))
            #elif(dataset_name == "RetinaMnist" or dataset_name == "BreastMnist" or dataset_name == "PneumoniaMnist" or dataset_name == "BloodMnist" or dataset_name == "DermaMnist"):
            elif("Mnist" in dataset_name and dataset_name != "OrganAMnist"):
                targets = []
                for _, labels in self.memory_data_loader:
                    labels = labels.squeeze()
                    targets.extend(labels)
                #else:
                #    import pdb
                #    pdb.set_trace()

            elif(dataset_name != "Stl10" and dataset_name != "OrganAMnist"):
                targets = self.memory_data_loader.dataset.targets
            else:
                targets = self.memory_data_loader.dataset.labels
                targets = targets.astype(numpy.int64)
                if(dataset_name == "OrganAMnist"):
                    targets = targets.squeeze()
            #import pdb
            #pdb.set_trace()
            
            
            feature_labels = torch.tensor(targets, device=feature_bank.device)
            
            all_test_features = list()
            all_test_labels = []
            all_pred_labels = []
            # loop test data to predict the label by weighted knn search
            test_bar = tqdm(self.test_data_loader, desc='kNN', disable=hide_progress)
            for data, target in test_bar:
                
                #import pdb
                #pdb.set_trace()
                #if(dataset_name == "BreastMnist"):
                #    data = data.repeat(1,3,1,1)
                #    data = self.normalize(data)
                
                if(dataset_name == "TinyImagenet"):
                    data = data[0]
                
                #if(dataset_name == "RetinaMnist" or dataset_name == "BreastMnist" or dataset_name == "PneumoniaMnist" or dataset_name == "BloodMnist" or dataset_name == "DermaMnist"):
                if("Mnist" in dataset_name):
                    target = target.squeeze()
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                feature = net(data)
                feature = F.normalize(feature, dim=1)
                #import pdb
                #pdb.set_trace()
                
                pred_labels = self.knn_predict(feature, feature_bank, feature_labels, classes, k, t)
                
                all_test_features.append(feature)
                all_test_labels.append(target)
                all_pred_labels.append(pred_labels)

                total_num += data.size(0)
                total_top1 += (pred_labels[:, 0] == target).float().sum().item()
                test_bar.set_postfix({'Accuracy':total_top1 / total_num * 100})
            
            all_test_features = torch.cat(all_test_features, dim=0)
            all_test_labels = torch.cat(all_test_labels, dim=0)
            all_pred_labels = torch.cat(all_pred_labels, dim=0)
        
        del feature_bank
        torch.cuda.empty_cache()
        #return all_test_features, all_test_labels, all_pred_labels, total_top1 / total_num * 100
        return total_top1 / total_num * 100
    # knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
    # implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
    def knn_predict(self, feature, feature_bank, feature_labels, classes, knn_k, knn_t):
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, feature_bank)
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / knn_t).exp()
        # counts for each class
        one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels
