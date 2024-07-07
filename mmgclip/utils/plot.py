import cv2
import matplotlib.pyplot as plt
from .data_utils import create_path
from PIL import Image

def plot_cv2_image(image):
    """
    Plots a cv2 image. Handles both grayscale and color images.
    
    Args:
        image (numpy array): The image to plot.

    Returns:
        None
    """
    if len(image.shape) == 2:
        # Image is grayscale
        plt.imshow(image, cmap='gray')
    elif len(image.shape) == 3:
        # Image is color (BGR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
    else:
        raise ValueError("Image format not recognized.")
    
    plt.axis('off')  # Do not show axes to keep it clean
    plt.show()

def plot_dataloader_batch(batch, base_dataset_path):
    '''
    Plots images from the dataloader.

    Args:
        - batch: a batch from the DataLoader.
        - base_dataset_path: path to the dataset

    Returns:
        - None, it generates a plot.
    '''
    batch_size = len(batch['image_features']) # len(batch['image_features']) # force only 2 images to be plotted
    figure = plt.figure(figsize=(16, 8))  # Adjust the figure size as needed
    
    for idx in range(batch_size):
        view_path = create_path(batch["image_id"][idx], base_dataset_path)  # Assuming create_path is defined elsewhere
        view_desc = batch["image_description"][idx]
        view_img = Image.open(view_path)
        view_name = batch['image_id'][idx]

        title = f"{view_name} ({'benign' if batch['image_label'][idx] == 0 else 'malignant'})\n{view_desc}"

        subplot = figure.add_subplot(1, batch_size, idx + 1)
        subplot.axis('off')
        subplot.set_title(title.replace('.', '.\n'))
        plt.imshow(view_img, cmap='gray')

    plt.tight_layout()


