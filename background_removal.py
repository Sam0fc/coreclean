""" background_removal module automatically crops using otsu thresholding.

Todo:
    arbritrary classes
"""
import numpy as np
import cv2 
from skimage.filters import threshold_multiotsu
import largestinteriorrectangle as lir  
import visualisation
import file_utils
import os
import matplotlib.pyplot as plt
import tqdm

def auto_crop(img: cv2.typing.MatLike,classes=3,edge_pixels=200,core_end_pixels=0, dilate_kernel = 5, dilate_iter = 10, erode_kernel=5, erode_iter=3) -> cv2.typing.MatLike:
    """ crops image to rectangle based on colour thresholds
    Params:
        img (cv2.typing.MatLike) : image to crop
        classes (int)            : number of multi-otsu sections in the image
        edge_pixels (int)        : how many pixels to discard on each side of the core 
        core_end_pixels (int)    : pixels to discard downcore on coretop and bottom

    Todo:
        make work for arbritrary number of classes
    """
    L_star = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:,:,0]
    thresholded_colour = threshold_image(L_star, classes=classes)
    eroded_colour = erode_image(thresholded_colour, kernel=np.ones((erode_kernel, erode_kernel), np.uint8), iterations=erode_iter)
    dilated_colour = dilate_image(eroded_colour, kernel=np.ones((dilate_kernel, dilate_kernel), np.uint8), iterations=dilate_iter)


    contours, _ = cv2.findContours(dilated_colour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify the contour into a very simple polygon
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    masked_image = cv2.bitwise_and(img, mask)
    #visualisation.show_image(thresholded_colour, eroded_colour, dilated_colour,mask,masked_image, alpha=False)

    x, y, w, h = cv2.boundingRect(largest_contour)

    cropped_image = img[y+edge_pixels:y+h-edge_pixels, x+core_end_pixels:x+w-core_end_pixels]
    #print('cropped')
    
    return cropped_image

def threshold_image(channel: cv2.typing.MatLike, classes: int) -> cv2.typing.MatLike:
    """uses the multi-otsu method to get the core-like colour threshold from the image histogram
    Params: 
        channel (cv2.typing.MatLike) : single colour channel from image which is used to threshold
        classes (int)                : number of multi-otsu colour sections in the image

    Todo: 
        make work with arbritrary number of classes
    """
    thresholds = threshold_multiotsu(channel, classes=classes)
    #print(thresholds)
    return cv2.inRange(channel, int(thresholds[0]), int(thresholds[1]))

def erode_image(img: cv2.typing.MatLike, 
                kernel: np.array = np.ones((5,5), np.uint8), 
                iterations: int = 3) -> cv2.typing.MatLike:
    """erodes the large sections of core-like colours to contiguous polygons
    Params:
        img (cv2.typing.MatLike) : image to erode
        kernel (np.array) : kernel whose size affects erosion

    Todo: 
        Work out how changing the kernel changes the effect
    """
    return cv2.erode(img, kernel,iterations=iterations)

def dilate_image(img: cv2.typing.MatLike, 
                kernel: np.array = np.ones((5,5), np.uint8), 
                iterations: int = 10) -> cv2.typing.MatLike:
    """erodes the large sections of core-like colours to contiguous polygons
    Params:
        img (cv2.typing.MatLike) : image to erode
        kernel (np.array) : kernel whose size affects erosion

    Todo: 
        Work out how changing the kernel changes the effect
    """
    return cv2.dilate(img, kernel,iterations=iterations)

def test_params(kernel_range: int = 5, iterations_range: int = 10, path='./Dataset/Exp_339'):
    """tests the effect of different kernel sizes and iterations on the image
    Params:
        kernel_range (int) : range of kernel sizes to test
        iterations_range (int) : range of iterations to test
    """
    fails = np.zeros(shape=(kernel_range, iterations_range,kernel_range, iterations_range))
    UnCroppedDataset = file_utils.ImageReader(path) 
    for img, newpath in tqdm.tqdm(UnCroppedDataset):
        for dilate_kernel in range(1,kernel_range):
            for dilate_iterations in range(1,iterations_range):
                for erode_kernel in range(1,kernel_range):
                    for erode_iterations in range(1,iterations_range):
                        test = auto_crop(img, dilate_kernel=dilate_kernel, dilate_iter=dilate_iterations, erode_kernel=erode_kernel, erode_iter=erode_iterations)
                        if not(newpath.__contains__('CC')) and test.shape[1] < file_utils.read_image(path+'/'+newpath)[0].shape[1]*0.9:
                            print(newpath)
                            #plt.imshow(test)
                            #plt.show()
                            
                            fails[dilate_kernel, dilate_iterations,erode_kernel,erode_iterations] += 1
                        elif np.sum(np.all(test[:,0,:] > 240,axis=1))>test.shape[0]*0.2:
                            print(test[:,0,:]>240)
                            print(img.shape)
                            white_pixels = np.argwhere(np.all(test > 240, axis=2))
                            print("White pixel locations:", white_pixels)
                            print(newpath)
                            #plt.imshow(test)
                            #plt.show()
                            fails[dilate_kernel, dilate_iterations,erode_kernel,erode_iterations] += 1 
    return fails/UnCroppedDataset.__len__() # normalise by number of images

def test_param(erode_kernel=7, erode_iter=3, dilate_kernel=7, dilate_iter=3, path='./Dataset/Exp_339'):
    """tests the effect of different kernel sizes and iterations on the image
    Params:
        kernel_range (int) : range of kernel sizes to test
        iterations_range (int) : range of iterations to test
    """
    white = 0
    short = 0 
    UnCroppedDataset = file_utils.ImageReader(path) 
    for img, newpath in tqdm.tqdm(UnCroppedDataset):
        test = auto_crop(img, dilate_kernel=dilate_kernel, dilate_iter=dilate_iter, erode_kernel=erode_kernel, erode_iter=erode_iter)
        
        size_constraints = (test.shape[1] > (200*150*0.95) and test.shape[1] < (200*150*1.05))  or (test.shape[1] > (200*120*0.95) and test.shape[1] < (200*120*1.05))

        if np.sum(np.all(test[:,0,:] > 240,axis=1))>test.shape[0]*0.2:
            print(newpath)
            #plt.imshow(test)
            #plt.show()
            white += 1

        elif not((newpath.__contains__('CC') or newpath.__contains__('H_7'))) and not size_constraints:
            print(newpath)
            print(test.shape[1])
            #plt.imshow(test)
            #plt.show()
            short += 1


    return (white+short)/UnCroppedDataset.__len__(),white,short # normalise by number of images

if __name__ == "__main__":
    # test image
    print(test_param())
   #UnCroppedDataset = file_utils.ImageReader("./Dataset/Exp_339")
   # 


   # cropped_imgs = []
   # for img,path in UnCroppedDataset:
   #     if os.path.exists(f"./Dataset/Cropped/{path}"):
   #         print(f"Skipping {path}, already cropped.")
   #         continue
   #     cropped = auto_crop(img, classes=3, edge_pixels=200, core_end_pixels=0)
   #     file_utils.save_image(cropped, path=f"./Dataset/Cropped/{path}")
   # visualisation.show_image(*cropped_imgs, alpha=False)

