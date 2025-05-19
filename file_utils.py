"""
file_utils module allows saving of images and manipulation of directories for patches etc
"""
import cv2 
import os
import natsort
from tifffile import TiffFile
import matplotlib.pyplot as plt
import tqdm 

def read_image(path: str, head: int = -1) -> list[cv2.typing.MatLike]:
    """read image from path"""
    if os.path.isdir(path):
        listpaths = os.listdir(path)
        listpaths = natsort.natsorted(listpaths)

        images = list()
        for i, imagepath in enumerate(listpaths):
            if imagepath.endswith(".tif") or imagepath.endswith(".tiff"):
                #print('reading tiff')
                images.append(tif2png(path + "/" + imagepath))
            else:
                images.append(cv2.imread(path + "/" + imagepath, cv2.IMREAD_UNCHANGED))
            if i + 1 >= head and head != -1:
                break
        return images
    if path.endswith(".tif") or path.endswith(".tiff"):
        #print('reading tiff')
        return [tif2png(path)]
    return [cv2.imread(path, cv2.IMREAD_UNCHANGED)]

class ImageReader:
    """ImageReader class to read images from a directory"""
    def __init__(self, path: str):
        self.path = path
        self.listpaths = os.listdir(path)
        self.listpaths = natsort.natsorted(self.listpaths)
        self.index = 0

    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.listpaths)
    
    def get_paths(self):
        """get paths of images"""
        return self.listpaths

    def __next__(self):
        if self.index >= len(self.listpaths):
            raise StopIteration
        imagepath = self.listpaths[self.index]
        if imagepath.endswith(".tif") or imagepath.endswith(".tiff"):
            image = tif2png(self.path + "/" + imagepath)
        else:
            image = cv2.imread(self.path + "/" + imagepath, cv2.IMREAD_UNCHANGED)
        self.index += 1
        return image,imagepath

def save_image(*imgs: list[cv2.typing.MatLike], path: str) -> None:
    """save images to path"""

    if len(imgs) == 1:
        cv2.imwrite(path, imgs[0])
        return None

    print(f"saving {path}")
    for i, img in tqdm.tqdm(enumerate(imgs)):
        fullpath = path + str(i) + ".png"
        cv2.imwrite(fullpath, img)
       

def tif2png(path: str) -> cv2.typing.MatLike:
    """convert tif to png"""
    tif = TiffFile(path)
    #print(tif)
    image = tif.asarray()
    image = image.transpose(1, 0, 2)  
    image = image[::-1, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    #print("tiffed")
    return image
