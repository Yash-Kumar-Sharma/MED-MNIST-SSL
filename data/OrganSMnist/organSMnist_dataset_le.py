import torchvision
import torchvision.datasets as datasets

import pytorch_lightning as pl
from Preprocess.Augmentations import model_transforms
from Feature_Extraction.features import prepare_data_features
from torch.utils.data import DataLoader

from medmnist import OrganSMNIST

class OrganSMnist_DataModule_le(pl.LightningDataModule):
    def __init__(self, model, config):
        super().__init__()

        self.model = model
        self.config = config
        self.dataset = config.dataset.name
        self.data_dir = config.dataset.data_dir
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers
        self.image_size = config.dataset.image_size
        self.convert_channel = config.dataset.convert_channel

    def prepare_data(self):     #Already Downloaded
        pass

    def setup(self, stage):
        transform = model_transforms(self.dataset, self.image_size, convert_channel=self.convert_channel)
        normalized_transform, _ = transform.GetTransform()
        
        train_set = OrganSMNIST(root = self.data_dir,
                                          transform = normalized_transform,
                                          split="train",
                                          download = False)
        
        #train_set, val_set = random_split(entire_set, [40000, 10000])

        test_set = OrganSMNIST(root = self.data_dir,
                                         transform = normalized_transform,
                                         split = "test",
                                         download = False)
        if(stage == "fit"):
            self.train_data = prepare_data_features(self.model, train_set, self.config)
        #self.val_data = prepare_data_features(self.model, val_set)
        if(stage == "test"):
            self.test_data = prepare_data_features(self.model, test_set, self.config)


    def train_dataloader(self):
        return DataLoader(self.train_data,
                      batch_size = self.batch_size,
                      num_workers = self.num_workers,
                      shuffle = True,
                      pin_memory = True)
        
    '''
    def val_dataloader(self):
        return DataLoader(self.val_data,
                      batch_size = self.batch_size,
                      num_workers = self.num_workers,
                      shuffle = False,
                      pin_memory = True)
    ''' 
    def test_dataloader(self):
        return DataLoader(self.test_data,
                      batch_size = self.batch_size,
                      num_workers = self.num_workers,
                      shuffle = False,
                      pin_memory = True)
        
