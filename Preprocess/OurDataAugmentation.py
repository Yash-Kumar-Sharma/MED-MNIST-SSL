import torch
from omegaconf import OmegaConf
from torch import nn
from torch import Tensor
from kornia.augmentation import ColorJiggle, ColorJitter, RandomHorizontalFlip, RandomResizedCrop, Normalize, RandomGrayscale, RandomGaussianBlur, Resize, RandomRotation, RandomGaussianNoise, RandomAffine
from torchvision.transforms import transforms

class OurDataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, image_size, kernel_size, crop_max, mean, std):
        super().__init__()
        self.mean = OmegaConf.to_object(mean)
        self.std = OmegaConf.to_object(std)
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.crop_max = crop_max
        self.pilTotensor = transforms.PILToTensor()
        self.normalize = Normalize(mean = self.mean, std = self.std)

        
        self.transforms = nn.Sequential(
            #ColorJiggle(brightness=0.8, contrast=0.8,saturation=0.8, hue=0.4, p=0.8),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            RandomGrayscale(p=0.2),
            #RandomResizedCrop(size=self.image_size, scale=(0.08, self.crop_max)),
            RandomResizedCrop(size=(self.image_size, self.image_size),
                              scale=(0.08, self.crop_max)),
            RandomHorizontalFlip(),
            RandomGaussianBlur(kernel_size=(self.kernel_size, self.kernel_size),sigma=(0.1,2.0),p=0.5),
            Normalize(mean = self.mean, std = self.std)
        )

        self.transforms_medical = nn.Sequential(
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01,p=0.8),
            RandomResizedCrop(size=(self.image_size, self.image_size),
                              scale=(0.2, 1.0)),
            RandomHorizontalFlip(p=0.5),
            RandomGaussianBlur(kernel_size= (3,3), sigma= (0.1, 1.0)),
            Normalize(mean = self.mean, std = self.std)
        )
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        x = self.transforms(x)  # BxCxHxW
        x_out = self.normalize(x)
        
        return x_out
