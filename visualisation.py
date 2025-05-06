""" visualisation module provides tools to display core sections, as well as their colour records.
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import file_utils
import os
import pandas as pd

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
    for image in sorted(os.listdir(composite_path)):
        if image.endswith('.tif'):
            img = file_utils.read_image(os.path.join(composite_path, image))[0]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            mean_Lstar = list(np.mean(img[:, :, 0], axis=(1)))
            mean_a = list(np.mean(img[:, :, 1], axis=(1)))
            mean_b = list(np.mean(img[:, :, 2], axis=(1)))
            sd_Lstar = list(np.std(img[:, :, 0], axis=(1)))
            sd_a = list(np.std(img[:, :, 1], axis=(1)))
            sd_b = list(np.std(img[:, :, 2], axis=(1)))
            Lstar += mean_Lstar
            sdL += sd_Lstar
            a += mean_a
            sdA += sd_a
            b += mean_b
            sdB += sd_b
    plt.plot(Lstar)
    plt.show()
    data = {
        "Lstar": Lstar,
        "sdL": sdL,
        "a": a,
        "sdA": sdA,
        "b": b,
        "sdB": sdB
    }
    df = pd.DataFrame(data)
    df.to_csv('colour.csv', index=False)

if __name__ == "__main__":
    colour_from_composite('./Composite')