import torch
import pandas as pd
import os 
from torch.utils.data import Dataset
import torchvision.io as TIO
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm
import cv2

from . import segmentation
from . import file_utils
from skimage.color import rgb2lab
import numpy as np
import random

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
    
class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = torch.flip(img, [1])  
        return img


class RandomLABColorTransform:
    def __init__(self, lightness_shift=2.38, a_shift=0.76, b_shift=1.37, noise_fraction=0.1):
        """
        Random color augmentation in the LAB color space.
        
        Args:
            lightness_shift (float): The maximum range to randomly shift the lightness (L channel).
            chroma_shift (float): The maximum range to randomly shift the chroma (A and B channels).
        """
        self.lightness_shift = lightness_shift * noise_fraction
        self.a_shift = a_shift * noise_fraction
        self.b_shift = b_shift * noise_fraction

    def __call__(self, img_rgb):
        """
        Args:
            img_rgb (torch.Tensor): Image tensor in RGB format (C, H, W).
        
        Returns:
            torch.Tensor: Augmented image tensor in RGB format (C, H, W).
        """
        # Convert the image from Tensor (C, H, W) to numpy array (H, W, C)
        img_rgb = img_rgb.permute(1, 2, 0).cpu().numpy()

        # Convert RGB to LAB color space
        img_lab = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

        # Apply random lightness shift
        if random.random() > 0.5:  # Randomly decide whether to shift lightness

            shift_L = random.gauss(0, self.lightness_shift)
            img_lab[..., 0] = np.clip(img_lab[..., 0] + shift_L * 255/100, 0, 255)

        # Apply random chroma shift (A and B channels)
        if random.random() > 0.5:
            shift_A = random.gauss(0, self.a_shift)
            img_lab[..., 1] = np.clip(img_lab[..., 1] + shift_A * 255, -128, 127)

        if random.random() > 0.5:
            # Apply random B channel shift
            shift_B = random.gauss(0, self.b_shift)
            img_lab[..., 2] = np.clip(img_lab[..., 2] + shift_B * 255, -128, 127)
        # Convert back to RGB
        img_rgb_augmented = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # Convert back to tensor (C, H, W)
        img_rgb_augmented = torch.tensor(img_rgb_augmented).float() / 255.0
        img_rgb_augmented = img_rgb_augmented.permute(2, 0, 1)  # Convert to (C, H, W)
        
        return img_rgb_augmented

class SwitchClusterTransform:
    def __init__(self,cluster1=(47.8/100 *255,-0.75+128,5.07+128),cluster2=(46.92/100 *255,0.65+128,6.97+128),cluster_size=1):
        """
        Random color augmentation in the LAB color space.
        
        Args:
            lightness_shift (float): The maximum range to randomly shift the lightness (L channel).
            chroma_shift (float): The maximum range to randomly shift the chroma (A and B channels).
        """
        self.cluster1 = cluster1
        self.cluster2 = cluster2
        self.cluster_size = cluster_size

    def __call__(self, img_rgb):
        """
        Args:
            img_rgb (torch.Tensor): Image tensor in RGB format (C, H, W).
        
        Returns:
            torch.Tensor: Augmented image tensor in RGB format (C, H, W).
        """
        # Convert the image from Tensor (C, H, W) to numpy array (H, W, C)
        img_rgb = img_rgb.permute(1, 2, 0).cpu().numpy()

        # Convert RGB to LAB color space
        img_lab = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        if random.random() > 0.5:
            
            # Calculate the vector to the nearest cluster mean
            mean_lab = np.mean(img_lab, axis=(0, 1))
            dist_to_cluster1 = np.linalg.norm(mean_lab - np.array(self.cluster1), axis=-1)
            dist_to_cluster2 = np.linalg.norm(mean_lab - np.array(self.cluster2), axis=-1)
            if min(dist_to_cluster1, dist_to_cluster2) < self.cluster_size:
                # If the distance to the nearest cluster is small, switch the color
                img_lab = img_lab + (np.array(self.cluster2) - np.array(self.cluster1))
            # Determine the nearest cluster
            nearest_cluster = np.where(dist_to_cluster1 < dist_to_cluster2, self.cluster1, self.cluster2)
            other_cluster = np.where(dist_to_cluster1 < dist_to_cluster2, self.cluster2, self.cluster1)

            # Calculate the vector from the nearest cluster to the other cluster
            vector_to_other_cluster = np.array(other_cluster) - np.array(nearest_cluster)

            # Place the color in the location relative to the other cluster
            img_lab = img_lab + vector_to_other_cluster
        # Convert back to RGB
        img_rgb_augmented = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Convert back to tensor (C, H, W)
        img_rgb_augmented = torch.tensor(img_rgb_augmented).float() / 255.0
        img_rgb_augmented = img_rgb_augmented.permute(2, 0, 1)  # Convert to (C, H, W)

        return img_rgb_augmented

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
    gt = 1-gt
    pred = 1-pred

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
    iou_sd = np.std(iou_list)
    return sum(iou_list) / len(iou_list), sum(accuracy_list) / len(accuracy_list), iou_sd

def determine_color_transform(path):
    """
    Determine color transform plane based on the directory of images
    """
    reader = file_utils.ImageReader(path)
    mean_colors = []
    for img, imgpath in tqdm.tqdm(reader):
        try:
            LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            L, A, B = cv2.split(LAB)
            mean_L = np.mean(L)
            mean_A = np.mean(A)
            mean_B = np.mean(B)
            mean_colors.append([mean_L, mean_A, mean_B])
        except Exception as e:
            print(f"Error converting image {imgpath} to LAB: {e}")
            plt.imshow(img)
            plt.show()


    mean_colors = np.array(mean_colors)
    mean_colors = pd.DataFrame(mean_colors, columns=['L', 'A', 'B'])
    mean_colors.to_csv('./colour_map', index=False)

if __name__ == "__main__":
    # Example usage of calc_iou
    image_dir = "./coreclean/Dataset/bigpatch"  # Replace with your image directory
    batch_size = 100

    # Define transformations
    transform = None
    color_transform = None

    # Create dataset and dataloader
    dataset = CustomImageDataset(image_dir)
    test, train = torch.utils.data.random_split(dataset, [int(len(dataset)*0.2), len(dataset) - int(len(dataset)*0.2)])
    dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)

    # Load your model (replace with your model loading code)
    model = segmentation.SegNet(kernel_size=5)  # Replace with your model
    model.load_state_dict(torch.load('./coreclean/Dataset/models/7/segnetRF2.pth'))  # Load your model weights

    # Calculate IoU
    iou, accuracy, sd = calc_iou(model, dataloader)
    print(f"Mean IoU: {iou:.4f}, Mean Accuracy: {accuracy:.4f}, IoU SD: {sd:.4f}")
#    # Test RandomFlip
    #transform = RandomFlip(p=1.0)  # Set probability to 1.0 to always flip for testing
    ## Create a random test image (C, H, W) with half random and half white
    #random_half = torch.rand(3, 100, 50)  # Random values for the left half
    #white_half = torch.ones(3, 100, 50)  # White values for the right half
    #test_image = torch.cat((random_half, white_half), dim=2)  # Concatenate along the width
    ## Rotate the image counterclockwise by 90 degrees
    #test_image = torch.rot90(test_image, k=1, dims=(1, 2))
    
    #flipped_image = transform(test_image)
    
    #print("Original Image Shape:", test_image.shape)
    #print("Flipped Image Shape:", flipped_image.shape)
    
    ## Visualize the original and flipped images
    #plt.figure(figsize=(10, 5))
    #plt.subplot(1, 2, 1)
    #plt.title("Original Image")
    #plt.imshow(test_image.permute(1, 2, 0).numpy())
    #plt.subplot(1, 2, 2)
    #plt.title("Flipped Image")
    #plt.imshow(flipped_image.permute(1, 2, 0).numpy())
    #plt.show()