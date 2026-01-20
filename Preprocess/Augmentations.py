# These are the transformations
import torchvision
import torchvision.transforms as transforms
from configparser import ConfigParser

from torchvision.transforms.functional import normalize
#import config

def GetNormalize(dataset):
  normalize = None
  if(dataset == "Cifar10" or dataset == "Cifar10_lt"):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                      std=[0.247, 0.243, 0.262]) # CIFAR10
  if(dataset == "Imagenet" or dataset == "Stl10" or dataset == "TinyImagenet"):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]) #Imagenet, #STL10, #Tiny-Imagenet, #cub200-2011
  if(dataset == "Cifar100"):
    normalize = transforms.Normalize(mean = [0.5071, 0.4867, 0.4408], 
                                        std = [0.2675, 0.2565, 0.2761]) #CIFAR100
  if(dataset == "Cub200"):
    normalize = transforms.Normalize(mean = [0.485, 0.499, 0.432],
                                        std = [0.228, 0.227, 0.266]) #Cub200
  if(dataset == "Aircrafts"):
    normalize = transforms.Normalize(mean = [0.489, 0.487, 0.455],
                                        std = [0.246, 0.242, 0.268]) #Aircrafts
  if(dataset == "Cars"):
    normalize = transforms.Normalize(mean = [0.470, 0.460, 0.454],
                                        std = [0.267, 0.265, 0.270]) #Cars
  if(dataset == "Pets"):
    normalize = transforms.Normalize(mean = [0.485, 0.449, 0.432],
                                        std = [0.229, 0.224, 0.225]) #Pets
  '''3-color channel MedMnist dataset'''
  if(dataset == "RetinaMnist"):
    normalize = transforms.Normalize(mean=[0.3948, 0.2425, 0.1536],
                                      std=[0.2924, 0.1948, 0.1446]) # RetinaMnist
  if(dataset == "BloodMnist"):
    normalize = transforms.Normalize(mean=[0.7944, 0.6597, 0.6962],
                                      std=[0.2108, 0.2368, 0.1109]) # RetinaMnist
  if(dataset == "DermaMnist"):
    normalize = transforms.Normalize(mean=[0.7637, 0.5383, 0.5615],
                                      std=[0.1358, 0.1526, 0.1675]) # RetinaMnist
  if(dataset == "PathMnist"):
    normalize = transforms.Normalize(mean=[0.7398, 0.5334, 0.7062],
                                      std=[0.1196, 0.1720, 0.1211]) # RetinaMnist
  
  '''32*32'''
  #if(dataset == "BloodMnist"):
  #  normalize = transforms.Normalize(mean=[0.7944, 0.6597, 0.6963],
  #                                    std=[0.2107, 0.2367, 0.1108]) # RetinaMnist
  
  '''1-color channel MedMnist dataset'''
  if(dataset == "BreastMnist"):
    normalize = transforms.Normalize(mean=[0.3306, 0.3306, 0.3306],
                                      std=[0.2025, 0.2025, 0.2025]) # RetinaMnist
  if(dataset == "PneumoniaMnist"):
    normalize = transforms.Normalize(mean=[0.5710, 0.5710, 0.5710],
                                      std=[0.1651, 0.1651, 0.1651]) # RetinaMnist
  if(dataset == "TissueMnist"):
    normalize = transforms.Normalize(mean=[0.1021, 0.1021, 0.1021],
                                      std=[0.0977, 0.0977, 0.0977]) # RetinaMnist
  if(dataset == "OrganCMnist"):
    normalize = transforms.Normalize(mean=[0.4896, 0.4896, 0.4896],
                                      std=[0.2638, 0.2638, 0.2638]) # RetinaMnist
  if(dataset == "OrganSMnist"):
    normalize = transforms.Normalize(mean=[0.4903, 0.4903, 0.4903],
                                      std=[0.2651, 0.2651, 0.2651]) # RetinaMnist
  if(dataset == "OrganAMnist"):
    normalize = transforms.Normalize(mean=[0.4660, 0.4660, 0.4660],
                                      std=[0.2763, 0.2763, 0.2763]) # RetinaMnist
  if(dataset == "OCTMnist"):
    normalize = transforms.Normalize(mean=[0.1890, 0.1890, 0.1890],
                                      std=[0.1907, 0.1907, 0.1907]) # RetinaMnist
  if(normalize == None):
    raise Exception("Datasets are as - cifar-10, Imagenet, cifar-100, TinyImagenet, STL-10, Imagenet")
  
  return normalize
  
class model_transforms:
  def __init__(self,dataset,image_size, convert_channel = False):
    self.dataset = dataset
    self.image_size = image_size
    self.convert_channel = convert_channel

  def GetTransform(self):
    '''
    config_object = ConfigParser()
    config_object.read("Config/con.dat")

    auginfo = config_object["augmentations"]
    '''

    normalize = GetNormalize(self.dataset)

    grayscale = torchvision.transforms.Grayscale(num_output_channels=3)
    #color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4,
    #                                      saturation=0.4, hue=0.1)

    #rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    #rnd_gray = transforms.RandomGrayscale(p=0.2)
    rnd_rcrop = transforms.RandomResizedCrop(size=self.image_size, scale=(0.08, 1),
            interpolation=transforms.InterpolationMode.BILINEAR)

    resize_image = torchvision.transforms.Resize(size=(self.image_size,self.image_size))
    rnd_hflip = transforms.RandomHorizontalFlip(p=0.5)

    #augmented_train_transform = transforms.Compose([resize_image,rnd_rcrop, rnd_hflip,
    #                                  rnd_color_jitter, rnd_gray,
    #                                  transforms.ToTensor(), normalize])
    augmented_train_transform = transforms.Compose([rnd_rcrop, rnd_hflip,
                                      transforms.ToTensor(), normalize])

    if("Mnist" in self.dataset and self.convert_channel):
        counterpart_train_transform = transforms.Compose([resize_image, grayscale,transforms.ToTensor(),normalize])
    else:
        counterpart_train_transform = transforms.Compose([resize_image,transforms.ToTensor(), normalize])
    #counterpart_train_transform = transforms.Compose([transforms.ToTensor(),normalize])

    return counterpart_train_transform, augmented_train_transform

