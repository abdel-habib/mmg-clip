import sys
import pprint as pp
import logging
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

def pprint(*args):
    '''
    Print large and indented objects clearly.

    Args:
        *arg: Variable number of arguments to print.
    '''
    printer = pp.PrettyPrinter(indent=4)

    printer.pprint(args)

def plot_logits_tensorboard(logits_per_image, logits_per_text, suptitle=None, writer = None, global_step = None):

    '''Logs logits probabilities to tensorboard as heatmaps, for both logits/image and logits/text. Note that the heatmaps are 
    normalized using a softmax.
    
    Args:
        logits_per_image (tensor): an [n, n] tensor representing the logits_per_image (rows: images, cols: text description).
        logtis_per_text (tensor): an [n, n] tensor representing the logits_per_text (rows: text description, cols: images).
        suptitle (str): subplot title.
        writer (SummaryWriter): writer instance to for logging.
    
    Returns:
        None
    '''
    if writer is None:
        raise ValueError("Missing tensorboard writer.")
    
    # Apply softmax to convert from logits to probabilities 
    probs_per_image = torch.softmax(logits_per_image[:8, :8], dim=1)
    probs_per_text = torch.softmax(logits_per_text[:8, :8], dim=1)

    # Convert tensors to numpy
    probs_per_image = probs_per_image.detach().cpu().numpy()
    probs_per_text = probs_per_text.detach().cpu().numpy()

    # Create a subplot grid
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # Plot heatmap 1
    sns.heatmap(probs_per_image, cmap='Blues', annot=True, fmt=".2f", xticklabels=False, yticklabels=False, ax=axes[0], cbar=False)
    axes[0].set_title('Probabilities/Image')
    axes[0].set_xlabel('Text Description')
    axes[0].set_ylabel('Images')
    axes[0].xaxis.tick_top()

    # Plot heatmap 2
    sns.heatmap(probs_per_text, cmap='Blues', annot=True, fmt=".2f", xticklabels=False, yticklabels=False, ax=axes[1], cbar=False)
    axes[1].set_title('Probabilities/Text')
    axes[1].set_xlabel('Images')
    axes[1].set_ylabel('Text Description')
    axes[1].xaxis.tick_top()
    
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()

    # Render the subplots
    plt.draw()

    # Convert the rendered subplots to a numpy array
    fig.canvas.flush_events()  # Ensure all pending events have been processed
    w, h = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))

    # Convert the numpy array to a PyTorch tensor
    tensor_image = torch.from_numpy(buffer.transpose(2, 0, 1))  # Transpose to have channels first
    
    # Log the image to TensorBoard
    writer.add_image('val/logits', tensor_image, global_step=global_step)

    # Close the plot to prevent displaying it
    plt.close()
