import seg_utils
import segmentation
import file_utils

import torch
import numpy as np
import cv2
import tqdm

def run_model(patches_path='./test_images', model_path='segmentation_model.pth', patch_size=128, kernel_size=3):
    """
    Run the segmentation model on a core section image and save the segmented image.
    
    Args:
        core_section_path (str): Path to the core section image.
        model_path (str): Path to the pre-trained segmentation model.
        patch_size (int): Size of the patches to be used for segmentation.
    """

    in_order, full_dataset = setup_data_loader(patches_path=patches_path, batch_size=100, evaluation=True)

    segmentation_model = setup_model(kernel_size=kernel_size, model_path=model_path)
    
    for index,i in tqdm(enumerate(iter(in_order))):
        batch = i[0]
        predictions = segmentation_model(i[0])  # Get the predictions from the model
        predictions = torch.argmax(predictions, dim=1)  # Get the predicted class labels
        for ind, (j, k) in enumerate(zip(batch, predictions)):
            cut_out = seg_utils.replace_alpha_channel(j, k).permute(1, 2, 0).cpu().detach().numpy()
            cut_out = cut_out * 255  # Scale the pixel values to [0, 255]
            cut_out = cut_out.astype(np.uint8)  # Convert to uint8 for visualization
            cut_out = cv2.cvtColor(cut_out, cv2.COLOR_BGRA2RGBA) 

            original_filename = full_dataset.image_files[index * 100 + ind]  # Get the original filename
            output_filename = f"./solved_patches/{original_filename}"  # Use the original filename for saving
            file_utils.save_image(cut_out, path=output_filename)  # Save the patched result
    

def setup_model(kernel_size=3, model_path='segmentation_model.pth'):
    """
    Setup the segmentation model by loading the pre-trained weights.
    
    Args:
        kernel_size (int): Size of the convolutional kernel.
        model_path (str): Path to the pre-trained segmentation model.
    """
    segmentation_model = segmentation.SegNet(kernel_size=kernel_size)
    segmentation_model.load_state_dict(torch.load(model_path))
    return segmentation_model
 
def setup_data_loader(patches_path='./test_images', batch_size=100, test_train_split=0.5, evaluation=False):
    """
    Setup the data loader for the patches.

    Args:
        patches_path (str): Path to the patches.
        batch_size (int): Size of the batches.
    """
    full_dataset = seg_utils.CustomImageDataset(
        image_dir=patches_path,
        transform=None,
        color_transform=None
    )
    if evaluation:
        return torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=False), full_dataset
    else: 
        # Split the dataset into training and testing sets
        train_size = int(len(full_dataset) * test_train_split)
        test_size = len(full_dataset) - train_size
        train_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader



