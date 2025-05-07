import torch
import pandas as pd
import os 
from torch.utils.data import Dataset
import torchvision.io as TIO
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm

def is_cuda():
    """
    Check if CUDA is available and return a boolean value.
    """
    return torch.cuda.is_available()

## Turning Dataset into pytorch
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, color_transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.color_transform = color_transform
        # Sort image files by the number in the filename
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = TIO.read_image(img_path, mode=TIO.ImageReadMode.RGBA).float() / 255.0
        img_rgb = image[:3, :, :]  # Extract RGB channels
        mask = image[3, :, :]  # Extract alpha channel as segmentation mask

        if self.transform:
            combined = torch.cat((img_rgb, mask.unsqueeze(0)), dim=0)
            transformed = self.transform(combined)
            img_rgb = transformed[:3, :, :]
            mask = transformed[3, :, :]
        
        if self.color_transform:
            img_rgb = self.color_transform(img_rgb)
        
        # Convert mask to binary (0 or 1)
        mask = (mask > 0).float()
        return img_rgb, mask
    
class ToDevice(torch.nn.Module):
    """
    A custom PyTorch module to move tensors to a specified device.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, x):
        return x.to(self.device)
    
    def __repr__( self ):
        return f"{self.__class__.__name__},{self.device})"

def IoUmetric(pred, gt):
    pred = torch.argmax(pred, dim=1)

    
    

    intersection = gt * pred
    union = gt + pred - intersection

    iou = intersection.sum(dim=(1, 2)) / (union.sum(dim=(1, 2)) + 1e-6)  # Avoid division by zero

    return iou.mean()

class IoUloss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, gt):
        return -(IoUmetric(pred, gt).log())

def replace_alpha_channel(image, alpha_channel):
    """
    Replace the alpha channel of an image with a new alpha channel.
    
    Args:
        image (torch.Tensor): The original image tensor.
        alpha_channel (torch.Tensor): The new alpha channel tensor.
        
    Returns:
        torch.Tensor: The image with the replaced alpha channel.
    """
    # Ensure the alpha channel is in the correct shape
    if len(alpha_channel.shape) == 2:
        alpha_channel = alpha_channel.unsqueeze(0)
    # Concatenate the RGB channels with the new alpha channel
    return torch.cat((image[:3, :, :], alpha_channel), dim=0)

def calc_iou(model, loader):
    """
    Calculate the Intersection over Union (IoU) for a given model and data loader.
    
    Args:
        model (torch.nn.Module): The segmentation model.
        loader (torch.utils.data.DataLoader): The data loader for the dataset.
        
    Returns:
        float: The average IoU across all batches.
    """
    accuracy_list = []
    iou_list = []
    model.eval()
    with torch.no_grad():
        for batch, targets in tqdm.tqdm(loader):
            batch = batch.to(next(model.parameters()).device)
            targets = targets.to(next(model.parameters()).device)
            outputs = model(batch)
            iou = IoUmetric(outputs, targets)
            accuracy_list.append((outputs.argmax(dim=1) == targets).float().mean())
            iou_list.append(iou.item())
    
    return sum(iou_list) / len(iou_list), sum(accuracy_list) / len(accuracy_list)