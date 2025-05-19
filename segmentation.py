"""segmentation module uses existing model to segment out disturbance.
"""

import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from . import seg_utils

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


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()

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


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling."""
    factor = (size + 1) // 2
    center = factor - 1 if size % 2 == 1 else factor - 0.5
    og = (torch.arange(size).float())
    filt = (1 - torch.abs(og - center) / factor)
    return filt[:, None] * filt[None, :]

class FCN8s(nn.Module):
    def __init__(self, n_class=2):
        super(FCN8s, self).__init__()
        self.n_class = n_class

        # VGG-like encoder
        self.vgg = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # pool1

            # conv2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # pool2

            # conv3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # pool3

            # conv4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # pool4

            # conv5
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # pool5
        )

        # Fully connected layers (as conv layers)
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),  # No padding!
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.fc8 = nn.Conv2d(4096, n_class, kernel_size=1)

        # Skip connections
        self.score_pool4 = nn.Conv2d(512, n_class, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, n_class, kernel_size=1)

        # Upsampling
        self.upscore2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, padding=1)
        self.upscore_pool4 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, padding=1)
        self.upscore8 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, padding=4)

        self._init_upscore_layers()

    def _init_upscore_layers(self):
        def initialize_upscore_layer(layer):
            weight = get_upsample_filter(layer.kernel_size[0])
            c = layer.in_channels
            layer.weight.data.copy_(
                weight.view(1, 1, *weight.shape).repeat(c, c, 1, 1)
            )
            if layer.bias is not None:
                layer.bias.data.zero_()

        initialize_upscore_layer(self.upscore2)
        initialize_upscore_layer(self.upscore_pool4)
        initialize_upscore_layer(self.upscore8)

    def copy_params_from_vgg16(self, vgg16, copy_fc8=True):
        # Copy feature layers
        vgg_layers = list(vgg16.features.children())
        for l1, l2 in zip(vgg_layers, self.vgg.children()):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.shape == l2.weight.shape
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)

        # Copy fc6
        fc6_vgg = vgg16.classifier[0]  # Linear(25088, 4096)
        fc6_fcn = self.fc6[0]          # Conv2d(512, 4096, 7x7)
        fc6_fcn.weight.data.copy_(fc6_vgg.weight.data.view(4096, 512, 7, 7))
        fc6_fcn.bias.data.copy_(fc6_vgg.bias.data)

        # Copy fc7
        fc7_vgg = vgg16.classifier[3]
        fc7_fcn = self.fc7[0]
        fc7_fcn.weight.data.copy_(fc7_vgg.weight.data.view(4096, 4096, 1, 1))
        fc7_fcn.bias.data.copy_(fc7_vgg.bias.data)

        # Do not copy fc8 unless user wants to (VGG's output classes ≠ segmentation classes)
        if copy_fc8:
            fc8_vgg = vgg16.classifier[6]
            fc8_fcn = self.fc8
            fc8_fcn.weight.data.copy_(fc8_vgg.weight.data[:self.n_class].view(self.n_class, 4096, 1, 1))
            fc8_fcn.bias.data.copy_(fc8_vgg.bias.data[:self.n_class])

    def forward(self, x):
        h = x

        pool1 = self.vgg[0:5](h)
        pool2 = self.vgg[5:10](pool1)
        pool3 = self.vgg[10:17](pool2)
        pool4 = self.vgg[17:24](pool3)
        pool5 = self.vgg[24:](pool4)

        h = self.fc6(pool5)
        h = self.fc7(h)
        h = self.fc8(h)

        upscore2 = self.upscore2(h)
        score_pool4 = self.score_pool4(pool4)
        score_pool4 = score_pool4[:, :, :upscore2.size(2), :upscore2.size(3)]
        h = upscore2 + score_pool4

        upscore_pool4 = self.upscore_pool4(h)
        score_pool3 = self.score_pool3(pool3)
        score_pool3 = score_pool3[:, :, :upscore_pool4.size(2), :upscore_pool4.size(3)]
        h = upscore_pool4 + score_pool3

        upscore8 = self.upscore8(h)
        h = upscore8[:, :, :x.size(2), :x.size(3)]

        return h



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
        nn.MaxPool2d(kernel_size=2),
        DoubleConv(in_channels, out_channels,kernel_size=kernel_size)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv(in_channels, 64, kernel_size=kernel_size)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.oc = OutConv(64, out_channels)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.oc(x)

        return logits

if __name__ == "__main__":
    full_dataset = seg_utils.CustomImageDataset(
        image_dir="./coreclean/Dataset/bigpatch/",
        transform=None,
        color_transform=None
    )
    print(len(full_dataset))
    train_data, test_data = torch.utils.data.random_split(full_dataset, [int(len(full_dataset)*0.8), len(full_dataset) - int(len(full_dataset)*0.8)])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=True)

    train_batch, train_targets = next(iter(train_loader))
    test_batch, test_targets = next(iter(test_loader))


    m = SegNet(kernel_size=5)
    m.load_state_dict(torch.load('./coreclean/Dataset/modles/segnet18.pth', map_location=get_device()))
    to_device(m.eval())
    print(seg_utils.calc_iou(m, test_loader))
    
    #with torch.no_grad():
    #    predictions = m(test_batch)
    #    predictions = torch.argmax(predictions, dim=1)
    #    show_batch(test_batch.cpu(), test_targets.cpu(), predictions.cpu())
    #    print(predictions.shape)
#