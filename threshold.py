"""thresold module does thresholding operations on images for testing pre-processing.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from . import background_removal
from matplotlib.colors import to_rgba

def remove_low_intensity_pixels(image, threshold_sd=1.5, otsu=False):
    """
    Remove low intensity pixels from an image based on a threshold calculated using the mean and standard deviation of the pixel intensities.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        threshold_sd (float): Standard deviation multiplier for threshold calculation.

    Returns:
        numpy.ndarray: Image with low intensity pixels removed.
    """
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[...,0]
    
    # Calculate mean and standard deviation
    mean = np.mean(gray_image)
    std_dev = np.std(gray_image)
    
    # Calculate threshold
    if otsu:
        _, threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        threshold = mean - threshold_sd * std_dev
    
    # Create a mask for pixels above the threshold
    mask = np.abs(gray_image - mean) <= mean - threshold
    
    # Plot histogram of gray image intensity
    print(threshold)
    plt.figure()
    plt.hist(gray_image.ravel(), bins=256, range=[0, 256], color='black', alpha=0.75)
    plt.axvline(x=threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold: {threshold:.2f}')
    plt.legend()
    plt.title('Histogram of Gray Image Intensity')
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')
    plt.show()
    return mask


def main():
    # Load the image
    image = cv2.imread('./coreclean/Dataset/crop_label/339_U1385A_13H_4.png', cv2.IMREAD_UNCHANGED)
    
    # Remove low intensity pixels
    mask = remove_low_intensity_pixels(image) 

    # Extract ground truth mask from alpha channel (assuming alpha channel is the 4th channel)
    if image.shape[2] == 4:
        alpha_channel = image[..., 3]
        gt_mask = 1 - (alpha_channel > 127)  # Threshold alpha to get binary mask
    else:
        raise ValueError("Image does not have an alpha channel for ground truth.")

    # Calculate Intersection over Union (IoU)
    dist_mask = 1-mask.astype(np.uint8)
    intersection = np.logical_and(dist_mask, gt_mask).sum()
    union = np.logical_or(dist_mask, gt_mask).sum()
    iou = intersection / union if union != 0 else 0

    # Calculate accuracy
    accuracy = (dist_mask == gt_mask).sum() / mask.size

    print(f"Intersection over Union (IoU): {iou:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Apply the mask to the original image
    bright_color = [0,0,0,0]
    result_image = np.zeros_like(image)
    result_image[mask] = image[mask]
    result_image[~mask] = bright_color
    
    # Display the result image with transparency and custom background
    # Create a background color using Purples colormap
    bg_color = to_rgba(plt.cm.Purples(0.8))

    # Prepare an RGBA image for display
    result_rgba = result_image.copy()
    if result_rgba.shape[2] == 3:
        # Add alpha channel if missing
        alpha = mask.astype(np.uint8) * 255
        result_rgba = np.dstack([result_rgba, alpha])
    else:
        result_rgba[..., 3] = mask.astype(np.uint8) * 255

    # Convert BGR to RGB for display
    result_rgba_rgb = result_rgba[..., :3][..., ::-1]
    result_rgba_rgb = np.dstack([result_rgba_rgb, result_rgba[..., 3]])

    # Create a background image
    bg = np.ones_like(result_rgba_rgb, dtype=np.float32)
    for c in range(4):
        bg[..., c] *= bg_color[c] * 255

    # Alpha blend result_rgba_rgb over bg
    alpha = result_rgba_rgb[..., 3:4] / 255.0
    blended = (result_rgba_rgb[..., :3] * alpha + bg[..., :3] * (1 - alpha)).astype(np.uint8) 

    blended = cv2.cvtColor(blended, cv2.COLOR_RGBA2BGR)

    cv2.imwrite('og.png', image)
    cv2.imwrite('th.png', blended)

    plt.figure(figsize=(10, 5))
    # Display the result
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Display the result image
    plt.subplot(2, 1, 2)
    plt.imshow(blended)
    plt.title('Result Image')
    plt.axis('off')
    
    plt.show()


if __name__ == '__main__':
    main()
