import pytorch_lightning as pl
import torchvision

from data.OurModel_Data import OurDataFromOCTMnist
from Preprocess.MedMnist_SSLDataAugmentation import MedMnist_SSLDataAugmentation
from torch.utils.data import DataLoader

def Get_Augmentation(model_name):
    
    model_function = model_name + "DataAugmentation"
    exec(f"generated_model = {model_function}", globals())
    return globals()['generated_model']


class OCTMnist_DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        #self.cfg = cfg
        
        self.data_dir = config.dataset.data_dir
        self.crop_max = config.dataset.crop_max
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers
        self.dataset = config.dataset.name
        self.image_size = config.dataset.image_size
        self.drop_last = config.dataset.drop_last
        self.mean = config.dataset.mean
        self.std = config.dataset.std
        self.one_color_channel = config.dataset.one_color_channel
        self.model_name = config.model.name
        
        self.K = config.model.K
         
        filter = int(config.model.gaussian_factor * self.image_size)
        if(filter % 2 == 0):
            kernel_size = filter + 1
        else:
            kernel_size = filter
        
        augmentation = Get_Augmentation(self.model_name)
        self.transform = augmentation(image_size = self.image_size,
                                                   kernel_size = kernel_size,
                                                   crop_max = self.crop_max,
                                                   mean = self.mean,
                                                   std = self.std)
        print(self.model_name + " augmentation Loading...")

    def prepare_data(self):     #Already Downloaded
        '''       
        dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
        download_url(dataset_url, '.')
        with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
            tar.extractall(path='./data')
        '''

    def setup(self, stage):
        
        if(stage == "fit"):
            resize = torchvision.transforms.Resize(size=(self.image_size, self.image_size))
            if(self.one_color_channel):
                default_transform = torchvision.transforms.Compose([resize, torchvision.transforms.ToTensor()])
            else:
                grayscale = torchvision.transforms.Grayscale(num_output_channels=3)
                default_transform = torchvision.transforms.Compose([resize, grayscale, torchvision.transforms.ToTensor()])
            print(self.model_name + " Data Loading...")
            if(self.model_name == "MedMnist_SSL"):     
                self.train_set = OurDataFromOCTMnist(K=self.K,
                                         root = self.data_dir,
                                         transform = default_transform,
                                         split = "train",)
                                         #download = True)
            
 
    def on_after_batch_transfer(self, batch, dataloader_idx):
        
        if self.trainer.training:

            if(self.model_name == "MedMnist_SSL"):     
                data, data_transform = batch        
                d = data.size()
                train_x = data.view(d[0]*d[1], d[2],d[3], d[4])
                train_x_transform = data_transform.view(d[0]*d[1], d[2],d[3], d[4])
                #import pdb
                #pdb.set_trace()
                train_x = self.transform.normalize(train_x)
                train_x_transform = self.transform.transforms_medical(train_x_transform)

                return train_x,train_x_transform
            
    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          shuffle = True,
                          pin_memory = False,
                          drop_last=self.drop_last,
                          )
        
        
    
    
