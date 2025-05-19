"""patching module splits core sections into patches to be analysed with computer vision
Todo:
    Deal with weird patches without just discarding
"""
import cv2
from . import file_utils
import os
import numpy as np
import matplotlib.pyplot as plt
from . import visualisation

def patch_image(image: cv2.typing.MatLike, patch_size = 128) -> list[cv2.typing.MatLike]:
    """splits large image into a list of patches of size patch_size. image is 0 when patch overlaps
    Params:
        image (cv2.typing.MatLike) : image to split into patches
        patch_size (int) : length of square patches 
    Returns:
        list[cv2.typing.MatLike] : list of patches
    """ 
    patches = []
    for x in range(0, image.shape[0], patch_size):
        for y in range(0, image.shape[1], patch_size):
            if x + patch_size <= image.shape[0] and y + patch_size <= image.shape[1]:
                patch = image[x:x+patch_size, y:y+patch_size]
                patches.append(patch)
    return patches

def restitch_images(originals_path: str, patches_path: str, patch_size: 128, out_path: str = './STITCHED') -> list[cv2.typing.MatLike]:
    """restitches patches into a single image
    Params:
        originals_path (str) : path to original images
        patches_path (str) : path to patches
        patch_size (int) : length of square patches
    Returns:
        list[cv2.typing.MatLike] : list of stitched images
    """
    originals = file_utils.ImageReader(originals_path)

    stitched = []
    for img, path in originals:
        stem = path.split('/')[-1]
        print(stem)
        
        listdir = os.listdir(patches_path)
        files_with_stem = [f for f in listdir if stem in f]
        files_with_stem.sort(key=lambda f: int(f.split('-')[-1].split('.')[0]))
        patches = []
        for i in range(len(files_with_stem)):
            patch = file_utils.read_image(os.path.join(patches_path, files_with_stem[i]))
            if len(patch) > 0:
                patches.append(patch[0])
        print(out_path+ '/' + stem)
        file_utils.save_image(restitch_image(img, patches, patch_size),path=out_path + '/' + stem)
    return stitched


def restitch_image(original: cv2.typing.MatLike,imgs: list[cv2.typing.MatLike], patch_size = 128) -> cv2.typing.MatLike:
    """restitches patches into a single image
    Params:
        original (cv2.typing.MatLike) : original image to get shape from
        imgs (list[cv2.typing.MatLike]) : list of patches to stitch
        patch_size (int) : length of square patches
    Returns:
        cv2.typing.MatLike : stitched image
    """

    stitched = np.zeros((original.shape[0], original.shape[1], 4), dtype=np.uint8)
    x=0
    y=0
    for i, img in enumerate(imgs):
        stitched[x:x+patch_size, y:y+patch_size] = img
        y += patch_size 
        if y+patch_size >= stitched.shape[1]:
            y = 0
            x += patch_size
        if x+patch_size >= stitched.shape[0]:
            break
    return stitched

if __name__ == "__main__":
    #cropped_images = file_utils.ImageReader("./coreclean/Composite/Cropped/")
    #for img,path in cropped_images:
        #patches256 = patch_image(img, patch_size=256)
        #patches128 = patch_image(img, patch_size=128)
        #file_utils.save_image(*patches128, path=f"./coreclean/Composite/Patched128/{path}-")
        #file_utils.save_image(*patches256, path=f"./coreclean/Composite/Patched256/{path}-")

    #file_utils.save_image(*patches128, path=f"./coreclean/Dataset/bigpatch/{path}-")

    restitch_images("./coreclean/Dataset/ls/", "./coreclean/Dataset/SEC/LAST", patch_size=256)
