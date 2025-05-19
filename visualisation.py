""" visualisation module provides tools to display core sections, as well as their colour records.
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from . import file_utils
import os
import pandas as pd
import tqdm
from . import segmentation
import torch
from . import seg_utils

def show_image(*imgs: list[cv2.typing.MatLike],head:int =-1, alpha=True) -> None:
    """ displays one or more images using matplotlib.
    Params:
        imgs (list[cv2.typing.MatLike]): images to display in BGR format.
    """
    _, axs = plt.subplots(len(imgs),1, figsize=(10,3),sharex=True)

    if len(imgs) == 1:
        print('1 image')
        axs = [axs]

    for i, img in enumerate(imgs):
        #img = img.copy()
        #img[np.all(img == [0, 0, 0], axis=-1)] = [255, 0, 255]
        if alpha:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        axs[i].imshow(img)
        if i + 1 >= head and head != -1:
            break

    plt.show()
    return None

def show_histogram(*imgs: list[cv2.typing.MatLike], head:int =-1) -> None:
    """ displays one or more images using matplotlib.
    Params:
        imgs (list[cv2.typing.MatLike]): images to display in BGR format.
    """
    _, axs = plt.subplots(min(len(imgs),head),1, figsize=(10,3),sharex=True)

    if len(imgs) == 1:
        axs = [axs]

    for i, img in enumerate(imgs):
        #img = img.copy()
        #img[np.all(img == [0, 0, 0], axis=-1)] = [255, 0, 255]
        axs[i].hist(img.ravel(), bins=256, range=[0,256])
        if i + 1 >= head and head != -1:
            break

    plt.show()
    return None

def colour_from_composite(composite_path: str):
    """ gets the colour from a composite section and makes a time series of the average colour across
    the core, with standard deviations. concatenate the images to make a time series
    Params:
        composite (str): path to composite section directory
    """
    Lstar = []
    sdL = []
    a = []
    sdA = []
    b = []
    sdB = []
    nanratio = []
    for image in tqdm.tqdm(sorted(os.listdir(composite_path))):
        if image.endswith('.png'):
            img = file_utils.read_image(os.path.join(composite_path, image))[0]
            alpha = img[:,:,3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            img[alpha==0] = [np.nan, np.nan, np.nan]
            mean_Lstar = list(np.nanmean(img[:, :, 0], axis=(1)))
            mean_a = list(np.nanmean(img[:, :, 1], axis=(1)))
            mean_b = list(np.nanmean(img[:, :, 2], axis=(1)))
            sd_Lstar = list(np.nanstd(img[:, :, 0], axis=(1)))
            sd_a = list(np.nanstd(img[:, :, 1], axis=(1)))
            sd_b = list(np.nanstd(img[:, :, 2], axis=(1)))
            nanmask = np.isnan(img[:,:,0])
            ratio = np.sum(nanmask, axis=1) / img.shape[1]
            nanratio += list(ratio)
            Lstar += mean_Lstar
            sdL += sd_Lstar
            a += mean_a
            sdA += sd_a
            b += mean_b
            sdB += sd_b
            print(sdL)
    data = {
        "Lstar": Lstar,
        "sdL": sdL,
        "a": a,
        "sdA": sdA,
        "b": b,
        "sdB": sdB,
        "nanratio": nanratio
    }
    df = pd.DataFrame(data)
    #df.to_csv('filtered.csv', index=False)

def training_val_loss(epoch_loss:str, modelpath:str,epochs:int):
    """ plots the training and validation loss over time.
    Params:
        epoch_loss (str): path to the csv file containing the epoch loss
    """
    full_dataset = seg_utils.CustomImageDataset(
        image_dir="./coreclean/Dataset/bigpatch/",
        transform=None,  # No transform applied initially
    )
    test_data, train_data = torch.utils.data.random_split(full_dataset, [int(len(full_dataset)*0.4), len(full_dataset) - int(len(full_dataset)*0.4)])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True)
    IOU = [] 
    CEL = []
    IOUSD = []
    df = pd.read_csv(epoch_loss)
    for i in range(epochs):
        model = segmentation.to_device(segmentation.SegNet(kernel_size=5))
        model.load_state_dict(torch.load(f'{modelpath}{i}.pth'))
        iou = seg_utils.calc_iou(model, test_loader)
        IOU.append(iou[0])
        IOUSD.append(iou[2])
        print(iou[0], iou[2])
        CEL.append(df['Loss'][int(i*0.6*4364/8):int((i+1)*0.6*4364/8)].mean()) 
        print(CEL[-1])
    

    with open('iou_values.txt', 'w') as f:
        for value,sd in zip(IOU,IOUSD):
            f.write(f'{value}, {sd}\n')
    plt.xlabel('Epoch')
    plt.plot(IOU, label='IOU')
    plt.plot(CEL, label='Cross Entropy Loss')
    plt.ylabel('Loss/IOU')
    plt.title('Training and Validation Loss/IOU')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #training_val_loss('./coreclean/Dataset/models/CS/epoch_loss.csv', './coreclean/Dataset/models/CS/segnetCS', 20)
    colour_from_composite('./coreclean/Composite/FinalStitch/')