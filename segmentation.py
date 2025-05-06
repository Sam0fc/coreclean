"""segmentation module uses existing model to segment out disturbance.
"""

import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import seg_utils

def to_device(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()

def get_device() -> torch.device:
    """ returns the device the model is on """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model: nn.Module, path: str) -> nn.Module:
    """ loads a model from a file.
    Params:
        model (nn.Module): model to load.
        path (str): path to the model file.
    """
    return model.load_state_dict(torch.load(path), map_location=get_device())

def segment_image(model: nn.Module, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """ segments out the disturbance in an image.
    Params:
        model (nn.Module): model to use.
        image (cv2.typing.MatLike): image to segment.
    """
    return model(image)

class DownConv2(nn.Module):

    def __init__(self, chin, chout, kernel_size):
        super().__init__() 
        self.seq = nn.Sequential(
                nn.Conv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chout),
                nn.ReLU(),
                nn.Conv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chout),
                nn.ReLU(),
        )
        self.mp = nn.MaxPool2d(kernel_size=2, return_indices=True) 

    def forward(self, x):
        y = self.seq(x)
        pool_shape = y.shape 
        y, indices = self.mp(y)
        return y, indices, pool_shape
    
class DownConv3(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
                nn.Conv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chout),
                nn.ReLU(),
                nn.Conv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chout),
                nn.ReLU(),
                nn.Conv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chout),
                nn.ReLU(),
        )
        self.mp = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        y = self.seq(x)
        pool_shape = y.shape 
        y, indices = self.mp(y)
        return y, indices, pool_shape
    
class UpConv2(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
                nn.Conv2d(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chin),
                nn.ReLU(),
                nn.Conv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chout),
                nn.ReLU(),
        )
        self.mup = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        y = self.mup(x, indices, output_size=output_size)
        y = self.seq(y)
        return y
        
class UpConv3(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
                nn.Conv2d(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chin),
                nn.ReLU(),
                nn.Conv2d(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chin),
                nn.ReLU(),
                nn.Conv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chout),
                nn.ReLU(),
        )
        self.mup = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        y = self.mup(x, indices, output_size=output_size)
        y = self.seq(y)
        return y
    
class SegNet(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.out_channels = 3 
        self.bn_input = nn.BatchNorm2d(3)

        self.dc1 = DownConv2(3, 64, kernel_size)
        self.dc2 = DownConv2(64, 128, kernel_size)
        self.dc3 = DownConv3(128, 256, kernel_size)
        self.dc4 = DownConv3(256, 512, kernel_size)

        self.uc4 = UpConv3(512, 256, kernel_size)
        self.uc3 = UpConv3(256, 128, kernel_size)
        self.uc2 = UpConv2(128, 64, kernel_size)
        self.uc1 = UpConv2(64, 2, kernel_size)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = self.bn_input(batch)
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)

        x = self.uc4(x, mp4_indices, output_size = shape4)
        x = self.uc3(x, mp3_indices, output_size = shape3)
        x = self.uc2(x, mp2_indices, output_size = shape2)
        x = self.uc1(x, mp1_indices, output_size = shape1)

        return x

def show_batch(inputs, targets, predictions=None):
    fig, axes = plt.subplots(nrows=3, ncols=inputs.size(0), figsize=(15, 5))
    for i in range(inputs.size(0)):
        axes[0, i].imshow(inputs[i].permute(1, 2, 0).cpu().numpy())
        axes[1, i].imshow(targets[i].cpu().numpy(), cmap='gray')
        if predictions is not None:
            axes[2, i].imshow(predictions[i].cpu().numpy())
        else:
            axes[2, i].axis('off')
    
    for ax in axes.ravel():
        ax.axis('off')
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1)
    plt.savefig('test.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    full_dataset = seg_utils.CustomImageDataset(
        image_dir="./test_images",
        transform=None,
        color_transform=None
    )
    print(len(full_dataset))
    train_data, test_data = torch.utils.data.random_split(full_dataset, [int(len(full_dataset)*0.8), int(len(full_dataset)*0.2)])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    train_batch, train_targets = next(iter(train_loader))
    test_batch, test_targets = next(iter(test_loader))


    m = SegNet(kernel_size=3)
    m.load_state_dict(torch.load('segmentation_model.pth', map_location=get_device()))
    
    with torch.no_grad():
        predictions = m(test_batch)
        predictions = torch.argmax(predictions, dim=1)
        show_batch(test_batch.cpu(), test_targets.cpu(), predictions.cpu())
        print(predictions.shape)
