"""The coreclean.py module uses deep learning based methods for segmentation of core images.

Uses the openCV module for most of the image manipulation. numpy is used for matrix operations and 
matplotlib for visualisations.

Todo:
    arbritrary classes in background_removal
    strange patch sizes in patching
"""
from . import background_removal
from . import threshold
from . import patching
from . import seg_utils
from . import segmentation
from . import training
# import statistical_analysis
from . import chronology
from . import visualisation
from . import file_utils
from . import pipeline

import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = "model.pth"


import os
def setup_directories():
    directories = [
        "./TO_PROCESS",
        './NO_BG',
        "./PATCHES",
        "./PATCHRESULT",
        "./STITCHED",
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def remove_background():
    """
    Remove the background from images in the TO_PROCESS directory and save them to the NO_BG directory.
    """
    image_files = os.listdir("./TO_PROCESS")
    for image_file in image_files:
        image_path = os.path.join("./TO_PROCESS", image_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        no_background = background_removal.auto_crop(image)
        file_utils.save_image(no_background, path=os.path.join("./NO_BG", image_file))

def make_patches():
    """
    Create patches from images in the NO_BG directory and save them to the PATCHES directory.
    """
    image_files = os.listdir("./NO_BG")
    for image_file in image_files:
        image_path = os.path.join("./NO_BG", image_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        patches = patching.patch_image(image, patch_size=256)
        file_utils.save_image(*patches, path=os.path.join("./PATCHES", image_file) + '-')

def process_patches():
    pipeline.run_model(patches_path="./PATCHES",save_path = './PATCHRESULT', model_path=MODEL_PATH, patch_size=256, kernel_size=5, batch_size=100, output_greyscale=False)

def stitch_patches():
    patching.restitch_images(originals_path="./NO_BG", patches_path="./PATCHRESULT", out_path='./STITCHED', patch_size=256)


if __name__ == "__main__":
    # path = "./test_images/section.bmp"

    #print("coreclean")
    #full_dataset = seg_utils.CustomImageDataset(
    #    image_dir="./test_images",
    #    transform=None,
    #    color_transform=None
    #)

    #patching.restitch_images(originals_path="./Dataset/339_Ship_Labelled/", patches_path="./solved_patches", patch_size=128)
    #pipeline.run_model(patches_path="./Dataset/labelled_patches", model_path='segmentation_model.pth', patch_size=128, kernel_size=3)
    
    #lets test the iou
    # Example usage of IoUmetric with dummy predictions and targets
    batch_size = 32
    for i in range(10):
        dummy_predictions = torch.ones((batch_size, 2, 128,128)) # Random predictions with values between 0 and 1
        dummy_targets = torch.randint(0, 2, (batch_size, 128, 128))  # Random binary ground truth masks
        
        iou_score = seg_utils.IoUmetric(dummy_predictions, dummy_targets)
        print(f"IoU Score: {iou_score}")

    #fixed_random = torch.Generator().manual_seed(0)
    #train_data, test_data = torch.utils.data.random_split(full_dataset, [int(len(full_dataset)*0.9), int(len(full_dataset)*0.1)], generator=fixed_random)
    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    #test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True)
    #test_batch, test_targets = next(iter(test_loader))

#    segmentation_model = segmentation.SegNet(kernel_size=3)
    #segmentation_model.load_state_dict(torch.load('segmentation_model.pth'))
    
    #in_order = torch.utils.data.DataLoader(full_dataset, batch_size=100, shuffle=False)
    #for index,i in enumerate(iter(in_order)):
        #batch = i[0]
        #predictions = segmentation_model(i[0])  # Get the predictions from the model
        #predictions = torch.argmax(predictions, dim=1)  # Get the predicted class labels
        #for ind, (j, k) in enumerate(zip(batch, predictions)):
            #cut_out = seg_utils.replace_alpha_channel(j, k).permute(1, 2, 0).cpu().detach().numpy()
            #cut_out = cut_out * 255  # Scale the pixel values to [0, 255]
            #cut_out = cut_out.astype(np.uint8)  # Convert to uint8 for visualization
            #cut_out = cv2.cvtColor(cut_out, cv2.COLOR_BGRA2RGBA) 
            #file_utils.save_image(cut_out, path=f"./solved_patches/solved_patch{index * 100 + ind}.png")  # Save the patched result

 


    #og = file_utils.read_image("./Dataset/339_Ship_Labelled/339_U1385A_01H_1.png")[0]
    #plt.imshow(og)
    #plt.show()
    #print(og.shape)
    #images = file_utils.read_image("./solved_patches/")
    #restitched = patching.restitch_image(og, images, patch_size=128)
    #plt.imshow(restitched)
    #plt.show()
    #file_utils.save_image(restitched, path="./stitches/restitched.png")
    
    
    #segmentation_model = segmentation.SegNet(kernel_size=3)
    #fixed_random = torch.Generator().manual_seed(0)
    #train_data, test_data = torch.utils.data.random_split(full_dataset, [int(len(full_dataset)*0.9), int(len(full_dataset)*0.1)], generator=fixed_random)
    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    #test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True)
    #optimiser = torch.optim.Adam(segmentation_model.parameters(), lr=0.001)
    #training.train_model(segmentation_model, train_loader, optimiser)

    #test_batch, test_targets = next(iter(test_loader))
    #predictions = segmentation_model(test_batch)
    #print(seg_utils.IoUmetric(predictions, test_targets))
    #torch.save(segmentation_model.state_dict(), 'segmentation_model.pth')

    
    

    #for i in range(len(patched)):
    #    patched[i] = segmentation_model(patched[i])
    # restitched = patching.restitch_image(images[0], patched, patch_size=128)
    # file_utils.save_image(restitched, path="./test_images/restitched.png")
    # visualisation.show_image(restitched)

    

    #background_images = file_utils.read_image("./Exp_339/")

    # visualisation.show_image(*background_images)
    # no_backgrounds = list()
    # for i in background_images:
        # visualisation.show_image(i)
        # no_background = background_removal.auto_crop(i, edge_pixels=0)
        # no_backgrounds.append(no_background)

    # file_utils.save_image(*no_backgrounds, path="./Exp_339/nobg/")

   # patches = file_utils.read_image("./patchresult")
   # original = file_utils.read_image("./og.bmp")[0]
   # restitched = patching.restitch_image(original,patches, patch_size=128)
   # file_utils.save_image(restitched, path="./test_images/restitched.png")
   # visualisation.show_image(original, restitched)

#   original = file_utils.read_image("og.bmp")[0]
#   make_patches = patching.patch_image(original, patch_size=128)
#   file_utils.save_image(*make_patches, path="./test_images/newpatch/patch")
    #no_background = background_removal.auto_crop(image)

    #patches = patching.patch_image(no_background)
    #file_utils.save_image(*patches, path="./test_images/patches/patch")
    #restitched = patching.restitch_image(no_background,patches)
    #visualisation.show_image(image, no_background, restitched)

    #image = file_utils.read_image("./uchannel.png")[0]
    #no_background = background_removal.auto_crop(image,edge_pixels=0)
    #visualisation.show_image(image, no_background)
    #file_utils.save_image(no_background, path="./test_images/section.bmp")
