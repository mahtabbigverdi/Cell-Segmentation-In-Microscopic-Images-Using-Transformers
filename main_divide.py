from __future__ import absolute_import
from __future__ import division
import torch
import pickle
from PIL import Image
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from typing import Callable, Tuple, List, Iterable, Type, Optional
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.color import gray2rgb
from torchvision.utils import save_image
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
import copy
from typing import Union, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from datetime import datetime
import numpy as np
import setproctitle
import os
import setproctitle
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.color import gray2rgb
from torchvision.utils import save_image

def iterable_to_device(data: List[torch.Tensor], device: str = "cuda") -> List[torch.Tensor]:
    """
    Function maps data to a given device.
    :param data: (List[torch.Tensor]) List of torch tensors
    :param device: (str) Device to be used
    :return: (List[torch.Tensor]) Input data mapped to the given device
    """
    # Iterate over all tensors
    for index in range(len(data)):
        # Map tensors to device
        data[index] = data[index].to(device)
    return data

class Logger(object):
    """
    Class to log different metrics
    """

    def __init__(self) -> None:
        self.metrics = dict()
        self.hyperparameter = dict()

    def log(self, metric_name: str, value: float) -> None:
        """
        Method writes a given metric value into a dict including list for every metric
        :param metric_name: (str) Name of the metric
        :param value: (float) Value of the metric
        """
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
        else:
            self.metrics[metric_name] = [value]

    def save_metrics(self, path: str) -> None:
        """
        Static method to save dict of metrics
        :param metrics: (Dict[str, List[float]]) Dict including metrics
        :param path: (str) Path to save metrics
        :param add_time_to_file_name: (bool) True if time has to be added to filename of every metric
        """
        # Save dict of hyperparameter as json file
        with open(os.path.join(path, 'hyperparameter.txt'), 'w') as json_file:
            json.dump(self.hyperparameter, json_file)
        # Iterate items in metrics dict
        for metric_name, values in self.metrics.items():
            # Convert list of values to torch tensor to use build in save method from torch
            values = torch.tensor(values)
            # Save values
            torch.save(values, os.path.join(path, '{}.pt'.format(metric_name)))

    def get_average_metric_for_epoch(self, metric_name: str, epoch: int, epoch_name: str = 'epoch') -> float:
        """
        Method calculates the average of a metric for a given epoch
        :param metric_name: (str) Name of the metric
        :param epoch: (int) Epoch to average over
        :param epoch_name: (str) Name of epoch metric
        :return: (float) Average metric
        """
        # Convert lists to np.array
        metric = np.array(self.metrics[metric_name])
        epochs = np.array(self.metrics[epoch_name])
        # Calc mean
        metric_average = np.mean(metric[np.argwhere(epochs == epoch)])
        return float(metric_average)

def plot_instance_segmentation_overlay_instances_bb_classes(image: torch.Tensor, instances: torch.Tensor,
                                                            bounding_boxes: torch.Tensor,
                                                            class_labels: torch.Tensor, save: bool = False,
                                                            show: bool = False,
                                                            file_path: str = "", alpha: float = 0.3,
                                                            show_class_label: bool = True,
                                                            colors_nucs: Tuple[Tuple[float, float, float], ...] = (
                                                                    (0.05, 0.05, 0.05),
                                                                    (0.25, 0.25, 0.25)),
                                                            cell_classes: Tuple[int, ...] = (1),
                                                            colors_cells: Tuple[Tuple[float, float, float], ...] = (
                                                                    (1.0, 1.0, 0.0),
                                                                    (0.5, 1.0, 0.0),
                                                                    (0.0, 0.625, 1.0),
                                                                    (1.0, 0.0, 0.0),
                                                                    (0.125, 1.0, 0.0),
                                                                    (1.0, 0.375, 0.0),
                                                                    (1.0, 0.0, 0.375),
                                                                    (1.0, 0.0, 0.75),
                                                                    (0.5, 0.0, 1.0),
                                                                    (1.0, 0.75, 0.0),
                                                                    (0.125, 0.0, 1.0),
                                                                    (0.0, 1.0, 0.625),
                                                                    (0.0, 1.0, 0.25),
                                                                    (0.0, 0.25, 1.0),
                                                                    (0.875, 0.0, 1.0),
                                                                    (0.875, 1.0,
                                                                     0.0))) -> None:
    """
    Function produces an instance segmentation plot
    :param image: (torch.Tensor) Input image of shape (3, height, width) or (1, height, width)
    :param instances: (torch.Tensor) Instances masks of shape (instances, height, width)
    :param bounding_boxes: (torch.Tensor) Bounding boxes of shape (instances, 4 (x1, y1, x2, y2))
    :param class_labels: (torch.Tensor) Class labels of each instance (instances, )
    :param save: (bool) If true image will be stored
    :param show: (bool) If true plt.show() will be called
    :param file_path: (str) Path and name where image will be stored
    :param show_class_label: (bool) If true class label will be shown in plot
    :param alpha: (float) Transparency factor of the instances
    :param colors_cells: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each cell instances.
    :param colors_nucs: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each nuclei instances.
    :param cell_classes: (Tuple[int, ...]) Tuple of cell classes
    """
    # Normalize image to [0, 255]
    image = normalize_0_1(image)
    # Convert data to numpy
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    instances = instances.detach().cpu().numpy()
    bounding_boxes = bounding_boxes.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Convert grayscale image to rgb
    if image.shape[-1] == 1:
        image = gray2rgb(image=image[:, :, 0])
    # Init counters
    counter_cell_instance = 0
    counter_nuc_instance = 0
    # Add instances to image
    for index, instance in enumerate(instances):
        # Case of cell instances
        if (class_labels[index] == 1):
            for c in range(image.shape[-1]):
                image[:, :, c] = np.where(instance == 1,
                                          image[:, :, c] * (1 - alpha) + alpha *
                                          colors_cells[min(counter_cell_instance, len(colors_cells) - 1)][c],
                                          image[:, :, c])
            counter_cell_instance += 1
        # Case of nuclei class
        elif (class_labels[index] == 2):
            for c in range(image.shape[-1]):
                image[:, :, c] = np.where(instance == 1,
                                          image[:, :, c] * (1 - alpha) + alpha *
                                          colors_nucs[min(counter_nuc_instance, len(colors_nucs) - 1)][c],
                                          image[:, :, c])
            counter_nuc_instance += 1
    # Init figure
    fig, ax = plt.subplots()
    # Set size
    fig.set_size_inches(5, 5 * image.shape[0] / image.shape[1])
    # Plot image and instances
    ax.imshow(image)
    # Init counters
    counter_cell_instance = 0
    counter_nuc_instance = 0
    # Plot bounding_boxes and classes
    for index, bounding_box in enumerate(bounding_boxes):
        # Case if cell is present
        if (class_labels[index] == 1):
            rectangle = patches.Rectangle((float(bounding_box[0]), float(bounding_box[1])),
                                          float(bounding_box[2]) - float(bounding_box[0]),
                                          float(bounding_box[3]) - float(bounding_box[1]),
                                          linewidth=3,
                                          edgecolor=colors_cells[min(counter_cell_instance, len(colors_cells) - 1)],
                                          facecolor='none', ls='dashed')
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0]) - 2),
                        float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1]) - 2),
                        'Cell', horizontalalignment='right', verticalalignment='bottom',
                        color="white", size=15)
            # Increment counter
            counter_cell_instance += 1
        # Cas if nuclei is present
        elif (class_labels[index] == 2):
            rectangle = patches.Rectangle((float(bounding_box[0]), float(bounding_box[1])),
                                          float(bounding_box[2]) - float(bounding_box[0]),
                                          float(bounding_box[3]) - float(bounding_box[1]),
                                          linewidth=3,
                                          edgecolor=colors_nucs[min(counter_nuc_instance, len(colors_nucs) - 1)],
                                          facecolor='none', ls='dashed')
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0]) - 2),
                        float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1]) - 2),
                        'Nuclei', horizontalalignment='right', verticalalignment='bottom',
                        color="white", size=15)
            # Increment counter
            counter_nuc_instance += 1
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Save figure if utilized
    if save:
        plt.savefig(file_path, dpi=image.shape[1] * 4 / 3.845, transparent=True, bbox_inches='tight', pad_inches=0)
    # Show figure if utilized
    if show:
        plt.show(bbox_inches='tight', pad_inches=0)
    # Close figure
    plt.close()


def plot_instance_segmentation_overlay_instances(image: torch.Tensor, instances: torch.Tensor,
                                                 class_labels: torch.Tensor, save: bool = False, show: bool = False,
                                                 file_path: str = "",
                                                 alpha: float = 0.5,
                                                 colors_cells: Tuple[Tuple[float, float, float], ...] = (
                                                         (1., 0., 0.89019608),
                                                         (1., 0.5, 0.90980392),
                                                         (0.7, 0., 0.70980392),
                                                         (0.7, 0.5, 0.73333333),
                                                         (0.5, 0., 0.53333333),
                                                         (0.5, 0.2, 0.55294118),
                                                         (0.3, 0., 0.45),
                                                         (0.3, 0.2, 0.45)),
                                                 colors_nucs: Tuple[Tuple[float, float, float], ...] = (
                                                         (0.05, 0.05, 0.05),
                                                         (0.25, 0.25, 0.25)),
                                                 cell_classes: Tuple[int, ...] = (2, 3)) -> None:
    """
    Function produces an instance segmentation plot
    :param image: (torch.Tensor) Input image of shape (3, height, width) or (1, height, width)
    :param instances: (torch.Tensor) Instances masks of shape (instances, height, width)
    :param class_labels: (torch.Tensor) Class labels of each instance (instances, )
    :param save: (bool) If true image will be stored
    :param show: (bool) If true plt.show() will be called
    :param file_path: (str) Path and name where image will be stored
    :param alpha: (float) Transparency factor
    :param colors_cells: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each cell instances.
    :param colors_nucs: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each nuclei instances.
    :param cell_classes: (Tuple[int, ...]) Tuple of cell classes
    """
    # Normalize image to [0, 1]
    image = normalize_0_1(image)
    # Convert data to numpy
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    instances = instances.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Convert grayscale image to rgb
    if image.shape[-1] == 1:
        image = gray2rgb(image=image[:, :, 0])
    # Init counters
    counter_cell_instance = 0
    counter_nuc_instance = 0
    # Add instances to image
    for index, instance in enumerate(instances):
        # Case of cell instances
        if bool(class_labels[index] ==1 ):
            for c in range(image.shape[-1]):
                image[:, :, c] = np.where(instance == 1,
                                          image[:, :, c] * (1 - alpha) + alpha *
                                          colors_cells[min(counter_cell_instance, len(colors_cells) - 1)][c],
                                          image[:, :, c])
            counter_cell_instance += 1
        # Case of nuc class
        elif (class_labels[index] ==2 ):
            for c in range(image.shape[-1]):
                image[:, :, c] = np.where(instance == 1,
                                          image[:, :, c] * (1 - alpha) + alpha *
                                          colors_nucs[min(counter_nuc_instance, len(colors_nucs) - 1)][c],
                                          image[:, :, c])
            counter_nuc_instance += 1

    # Init figure
    fig, ax = plt.subplots()
    # Set size
    fig.set_size_inches(5, 5 * image.shape[0] / image.shape[1])
    # Plot image and instances
    ax.imshow(image)
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Save figure if utilized
    if save:
        plt.savefig(file_path, dpi=image.shape[1] * 4 / 3.845, transparent=True, bbox_inches='tight', pad_inches=0)
    # Show figure if utilized
    if show:
        plt.show(bbox_inches='tight', pad_inches=0)
    # Close figure
    plt.close()


def plot_instance_segmentation_labels(instances: torch.Tensor, bounding_boxes: torch.Tensor,
                                      class_labels: torch.Tensor, save: bool = False, show: bool = False,
                                      file_path: str = "",
                                      colors_cells: Tuple[Tuple[float, float, float], ...] = ((1., 0., 0.89019608),
                                                                                              (1., 0.5, 0.90980392),
                                                                                              (0.7, 0., 0.70980392),
                                                                                              (0.7, 0.5, 0.73333333),
                                                                                              (0.5, 0., 0.53333333),
                                                                                              (0.5, 0.2, 0.55294118),
                                                                                              (0.3, 0., 0.45),
                                                                                              (0.3, 0.2, 0.45)),
                                      colors_nucs: Tuple[Tuple[float, float, float], ...] = (
                                              (0.3, 0.3, 0.3),
                                              (0.5, 0.5, 0.5)),
                                      cell_classes: Tuple[int, ...] = (2, 3), white_background: bool = False,
                                      show_class_label: bool = True) -> None:
    """
    Function plots given instance segmentation labels including the pixel-wise segmentation maps, bounding boxes,
    and class labels
    :param instances: (torch.Tensor) Pixel-wise instance segmentation map
    :param bounding_boxes: (torch.Tensor) Bounding boxes of shape (instances, 4 (x1, y1, x2, y2))
    :param class_labels: (torch.Tensor) Class labels of each instance (instances, )
    :param save: (bool) If true image will be saved (matplotlib is used)
    :param show: (bool) If true matplotlib plot of the image will be shown
    :param file_path: (str) Path and name where image will be stored
    :param colors_cells: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each cell instances.
    :param colors_nucs: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each nuclei instances.
    :param white_background: (bool) If true a white background is utilized
    :param show_class_label: (bool) If true class name will be shown in the left bottom corner of each bounding box
    """
    # Convert data to numpy
    instances = instances.detach().cpu().numpy()
    bounding_boxes = bounding_boxes.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Init map to visualize instances
    instances_map = np.zeros((instances.shape[1], instances.shape[2], 3), dtype=np.float)
    # Init counters to track the number of cells and traps for different colours
    counter_cell_instance = 0
    counter_nuc_instance = 0
    # Instances to instances map
    for instance, class_label in zip(instances, class_labels):
        # Case if cell is present
        if bool(class_label == 1):
            # Add pixels of current instance, in the corresponding colour, to instances map
            instances_map += np.array(colors_cells[min(counter_cell_instance, len(colors_cells) - 1)]).reshape(1, 1, 3) \
                             * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_cell_instance += 1
        # Cas if nuclei is present
        elif (class_label == 2):
            # Add pixels of current instance, in the corresponding colour, to instances map
            instances_map += np.array(colors_nucs[min(counter_nuc_instance, len(colors_cells) - 1)]).reshape(1, 1, 3) \
                             * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_nuc_instance += 1
    # Init figure
    fig, ax = plt.subplots()
    # Set size
    fig.set_size_inches(5, 5 * instances_map.shape[0] / instances_map.shape[1])
    # Make background white if specified
    if white_background:
        for h in range(instances_map.shape[0]):
            for w in range(instances_map.shape[1]):
                if np.alltrue(instances_map[h, w, :] == np.array([0.0, 0.0, 0.0])):
                    instances_map[h, w, :] = np.array([1.0, 1.0, 1.0])
    # Plot image and instances
    ax.imshow(instances_map)
    # Init counters to track the number of cells and nucleis for different colours
    counter_cell_instance = 0
    counter_nuc_instance = 0
    # Plot bounding_boxes and classes
    for index, bounding_box in enumerate(bounding_boxes):
        # Case if cell is present
        if bool(class_labels[index] ==1 ):
            rectangle = patches.Rectangle((float(bounding_box[0]), float(bounding_box[1])),
                                          float(bounding_box[2]) - float(bounding_box[0]),
                                          float(bounding_box[3]) - float(bounding_box[1]),
                                          linewidth=3,
                                          edgecolor=colors_cells[min(counter_cell_instance, len(colors_cells) - 1)],
                                          facecolor='none', ls='dashed')
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0]) - 2),
                        float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1]) - 2),
                        'Cell', horizontalalignment='right', verticalalignment='bottom',
                        color="black" if white_background else "white", size=15)
            # Increment counter
            counter_cell_instance += 1
        # Cas if nuclei is present
        elif (class_labels[index] == 2 ):
            rectangle = patches.Rectangle((float(bounding_box[0]), float(bounding_box[1])),
                                          float(bounding_box[2]) - float(bounding_box[0]),
                                          float(bounding_box[3]) - float(bounding_box[1]),
                                          linewidth=3,
                                          edgecolor=colors_nucs[min(counter_nuc_instance, len(colors_nucs) - 1)],
                                          facecolor='none', ls='dashed')
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0]) - 2),
                        float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1]) - 2),
                        'Nuclei', horizontalalignment='right', verticalalignment='bottom',
                        color="black" if white_background else "white", size=15)
            # Increment counter
            counter_nuc_instance += 1
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Save figure if utilized
    if save:
        plt.savefig(file_path, dpi=instances_map.shape[1] * 4 / 3.845, transparent=True, bbox_inches='tight',
                    pad_inches=0)
    # Show figure if utilized
    if show:
        plt.show(bbox_inches='tight', pad_inches=0)
    # Close figure
    plt.close()


def plot_instance_segmentation_map_label(instances: torch.Tensor, class_labels: torch.Tensor, save: bool = False,
                                         show: bool = False, file_path: str = "",
                                         colors_cells: Tuple[Tuple[float, float, float], ...] = ((1., 0., 0.89019608),
                                                                                                 (1., 0.5, 0.90980392),
                                                                                                 (0.7, 0., 0.70980392),
                                                                                                 (0.7, 0.5, 0.73333333),
                                                                                                 (0.5, 0., 0.53333333),
                                                                                                 (0.5, 0.2, 0.55294118),
                                                                                                 (0.3, 0., 0.45),
                                                                                                 (0.3, 0.2, 0.45)),
                                         colors_nucs: Tuple[Tuple[float, float, float], ...] = (
                                                 (0.3, 0.3, 0.3),
                                                 (0.5, 0.5, 0.5)),
                                         cell_classes: Tuple[int, ...] = (2, 3),
                                         white_background: bool = False) -> None:
    """
    Function plots given instance segmentation labels including the pixel-wise segmentation maps, bounding boxes,
    and class labels
    :param instances: (torch.Tensor) Pixel-wise instance segmentation map
    :param class_labels: (torch.Tensor) Class labels of each instance (instances, )
    :param save: (bool) If true image will be saved (matplotlib is used)
    :param show: (bool) If true matplotlib plot of the image will be shown
    :param file_path: (str) File path to save the image
    :param colors_cells: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each cell instances.
    :param colors_traps: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each trap instances.
    :param white_background: (bool) If true a white background is utilized
    """
    # Convert data to numpy
    instances = instances.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Init map to visualize instances
    instances_map = np.zeros((instances.shape[1], instances.shape[2], 3), dtype=np.float)
    # Init counters to track the number of cells and traps for different colours
    counter_cell_instance = 0
    counter_nuc_instance = 0
    # Instances to instances map
    for instance, class_label in zip(instances, class_labels):
        # Case if cell is present
        if bool(class_label ==1):
            # Add pixels of current instance, in the corresponding colour, to instances map
            instances_map += np.array(colors_cells[min(counter_cell_instance, len(colors_cells) - 1)]).reshape(1, 1, 3) \
                             * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_cell_instance += 1
        # Cas if nuclei is present
        elif (class_label ==2):
            # Add pixels of current instance, in the corresponding colour, to instances map
            instances_map += np.array(colors_nuc[min(counter_nuc_instance, len(colors_cells) - 1)]).reshape(1, 1, 3) \
                             * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_nuc_instance += 1
    # Init figure
    fig, ax = plt.subplots()
    # Set size
    fig.set_size_inches(5, 5 * instances_map.shape[0] / instances_map.shape[1])
    # Make background white if specified
    if white_background:
        for h in range(instances_map.shape[0]):
            for w in range(instances_map.shape[1]):
                if np.alltrue(instances_map[h, w, :] == np.array([0.0, 0.0, 0.0])):
                    instances_map[h, w, :] = np.array([1.0, 1.0, 1.0])
    # Plot image and instances
    ax.imshow(instances_map)
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Save figure if utilized
    if save:
        plt.savefig(file_path, dpi=instances_map.shape[1] * 4 / 3.845, transparent=True, bbox_inches='tight',
                    pad_inches=0)
    # Show figure if utilized
    if show:
        plt.show(bbox_inches='tight', pad_inches=0)
    # Close figure
    plt.close()


def plot_image(image: torch.Tensor, save: bool = False, show: bool = False, file_path: str = "") -> None:
    """
    This function plots and saves an images
    :param image: (torch.Tensor) Image as a torch tensor
    :param save: (bool) If true image will be saved (torchvision save_image function is utilized)
    :param show: (bool) If true matplotlib plot of the image will be shown
    :param file_path: (str) File path to save the image
    """
    # Make sure image tensor is not on GPU an is not attached to graph
    image = image.cpu().detach()
    # Normalize image to [0, 255]
    image = normalize_0_1(image)
    # Save image if utilized
    if save:
        # Add batch dim to image if needed
        image_ = image.unsqueeze(dim=0) if image.ndim == 3 else image
        save_image(image_, file_path, nrow=1, padding=0, normalize=False)
    # Show matplotlib plot if utilized
    if show:
        # Change oder of dims to match matplotlib format and convert to numpy
        image = image.permute(1, 2, 0).numpy()
        # Init figure
        fig, ax = plt.subplots()
        # Set size
        fig.set_size_inches(5, 5 * image.shape[0] / image.shape[1])
        # Plot image and instances
        ax.imshow(image[:, :, 0], cmap="gray")
        # Axis off
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show(bbox_inches='tight', pad_inches=0)
        # Close figure
        plt.close()


def plot_instance_segmentation_overlay_bb_classes(image: torch.Tensor, bounding_boxes: torch.Tensor,
                                                  class_labels: torch.Tensor, save: bool = False, show: bool = False,
                                                  file_path: str = "",
                                                  show_class_label: bool = True,
                                                  colors_cells: Tuple[Tuple[float, float, float], ...] =
                                                  (1., 0., 0.89019608),
                                                  colors_nucs: Tuple[Tuple[float, float, float], ...] =
                                                  (0.0, 0.0, 0.0),
                                                  cell_classes: Tuple[int, ...] = (2, 3)) -> None:
    """
    Function produces an instance segmentation plot
    :param image: (torch.Tensor) Input image of shape (3, height, width) or (1, height, width)
    :param bounding_boxes: (torch.Tensor) Bounding boxes of shape (instances, 4 (x1, y1, x2, y2))
    :param class_labels: (torch.Tensor) Class labels of each instance (instances, )
    :param save: (bool) If true image will be stored
    :param show: (bool) If true plt.show() will be called
    :param file_path: (str) Path and name where image will be stored
    :param show_class_label: (bool) If true class label is show in plot
    :param alpha: (float) Transparency factor
    :param colors_cells: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each cell instances.
    :param colors_traps: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each trap instances.
    :param cell_classes: (Tuple[int, ...]) Tuple of cell classes
    """
    # Normalize image to [0, 255]
    image = normalize_0_1(image)
    # Convert data to numpy
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    bounding_boxes = bounding_boxes.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Convert grayscale image to rgb
    if image.shape[-1] == 1:
        image = gray2rgb(image=image[:, :, 0])
    # Init figure
    fig, ax = plt.subplots()
    # Set size
    fig.set_size_inches(5, 5 * image.shape[0] / image.shape[1])
    # Plot image and instances
    ax.imshow(image)
    # Plot bounding_boxes and classes
    for index, bounding_box in enumerate(bounding_boxes):
        # Case if cell is present
        if bool(class_labels[index] ==1):
            rectangle = patches.Rectangle((float(bounding_box[0]), float(bounding_box[1])),
                                          float(bounding_box[2]) - float(bounding_box[0]),
                                          float(bounding_box[3]) - float(bounding_box[1]),
                                          linewidth=3,
                                          edgecolor=colors_cells,
                                          facecolor='none', ls='dashed')
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0])) - 2,
                        float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1])) - 2,
                        'Cell', horizontalalignment='right', verticalalignment='bottom', color="white", size=15)
        # Cas if trap is present
        elif (class_labels[index] == 2):
            rectangle = patches.Rectangle((float(bounding_box[0]), float(bounding_box[1])),
                                          float(bounding_box[2]) - float(bounding_box[0]),
                                          float(bounding_box[3]) - float(bounding_box[1]),
                                          linewidth=3,
                                          edgecolor=colors_nucs,
                                          facecolor='none', ls='dashed')
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0])) - 2,
                        float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1])) - 2,
                        'Nuclei', horizontalalignment='right', verticalalignment='bottom', color="white", size=15)
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Save figure if utilized
    if save:
        plt.savefig(file_path, dpi=image.shape[1] * 4 / 3.845, transparent=True, bbox_inches='tight', pad_inches=0)
    # Show figure if utilized
    if show:
        plt.show(bbox_inches='tight', pad_inches=0)
    # Close figure
    plt.close()


def plot_instance_segmentation_instances(instances: torch.Tensor, class_labels: torch.Tensor, save: bool = False,
                                         show: bool = False, file_path: str = "",
                                         colors_cells: Tuple[Tuple[float, float, float], ...] = ((1., 0., 0.89019608),
                                                                                                 (1., 0.5, 0.90980392),
                                                                                                 (0.7, 0., 0.70980392),
                                                                                                 (0.7, 0.5, 0.73333333),
                                                                                                 (0.5, 0., 0.53333333),
                                                                                                 (0.5, 0.2, 0.55294118),
                                                                                                 (0.3, 0., 0.45),
                                                                                                 (0.3, 0.2, 0.45)),
                                         colors_nucs: Tuple[Tuple[float, float, float], ...] = (
                                                 (0.3, 0.3, 0.3),
                                                 (0.5, 0.5, 0.5)),
                                         cell_classes: Tuple[int, ...] = (2, 3),
                                         white_background: bool = False) -> None:
    """
    Function plots given instance segmentation labels including the pixel-wise segmentation maps, bounding boxes,
    and class labels
    :param instances: (torch.Tensor) Pixel-wise instance segmentation map
    :param bounding_boxes: (torch.Tensor) Bounding boxes of shape (instances, 4 (x1, y1, x2, y2))
    :param class_labels: (torch.Tensor) Class labels of each instance (instances, )
    :param save: (bool) If true image will be saved (matplotlib is used)
    :param show: (bool) If true matplotlib plot of the image will be shown
    :param file_path: (str) File path to save the image
    :param colors_cells: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each cell instances.
    :param colors_traps: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each trap instances.
    :param white_background: (bool) If true a white background is utilized
    """
    # Convert data to numpy
    instances = instances.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Init counters to track the number of cells and traps for different colours
    counter_cell_instance = 0
    counter_nuc_instance = 0
    # Instances to instances map
    for index, data in enumerate(zip(instances, class_labels)):
        # Unzip data
        instance, class_label = data
        # Case if cell is present
        if bool(class_label == 1):
            # Add pixels of current instance, in the corresponding colour, to instances map
            instance = np.array(colors_cells[min(counter_cell_instance, len(colors_cells) - 1)]).reshape(1, 1, 3) \
                       * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_cell_instance += 1
        # Cas if trap is present
        elif (class_label == 2):
            # Add pixels of current instance, in the corresponding colour, to instances map
            instance = np.array(colors_nucs[min(counter_nuc_instance, len(colors_cells) - 1)]).reshape(1, 1, 3) \
                       * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_nuc_instance += 1
        # Init figure
        fig, ax = plt.subplots()
        # Set size
        fig.set_size_inches(5, 5 * instance.shape[0] / instance.shape[1])
        # Make background white if specified
        if white_background:
            for h in range(instance.shape[0]):
                for w in range(instance.shape[1]):
                    if np.alltrue(instance[h, w, :] == np.array([0.0, 0.0, 0.0])):
                        instance[h, w, :] = np.array([1.0, 1.0, 1.0])
        # Plot image and instances
        ax.imshow(instance)
        # Axis off
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Save figure if utilized
        if save:
            plt.savefig(file_path.replace(".", "_{}.".format(index)), dpi=instance.shape[1] * 4 / 3.845,
                        transparent=True, bbox_inches='tight', pad_inches=0)
        # Show figure if utilized
        if show:
            plt.show(bbox_inches='tight', pad_inches=0)
        # Close figure
        plt.close()

def to_one_hot(input: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Converts a given tensor to a one hot encoded tensor
    :param input: (torch.Tensor) Class number tensor
    :param num_classes: (int) Number of classes
    :return: (torch.Tensor) One hot tensor
    """
    one_hot = torch.zeros([input.shape[0], num_classes], dtype=torch.float)
    one_hot.scatter_(1, input.view(-1, 1).long(), 1)
    return one_hot

def normalize(input: torch.Tensor) -> torch.Tensor:
    """
    Normalize a given tensor to var=1.0 and mean=0.0
    :param input: (torch.Tensor) Input tensor
    :return: (torch.Tensor) Normalized output tensor
    """
    return (input - input.mean()) / input.std()


def normalize_0_1(input: torch.Tensor) -> torch.Tensor:
    """
    Normalize a given tensor to a range of [0, 1]
    :param input: (Torch tensor) Input tensor
    :param inplace: (bool) If true normalization is performed inplace
    :return: (Torch tensor) Normalized output tensor
    """
    # Perform normalization not inplace
    return (input - input.min()) / (input.max() - input.min())

def bounding_box_xcycwh_to_x0y0x1y1(bounding_boxes: torch.Tensor) -> torch.Tensor:
    """
    This function converts a given bounding bix of the format
    [batch size, instances, 4 (x center, y center, width, height)] to [batch size, instances, 4 (x0, y0, x1, y1)].
    :param bounding_boxes: Bounding box of shape [batch size, instances, 4 (x center, y center, width, height)]
    :return: Converted bounding box of shape [batch size, instances, 4 (x0, y0, x1, y1)]
    """
    
    x_center, y_center, width, height = bounding_boxes.unbind(dim=-1)
    bounding_box_converted = [(x_center - 0.5 * width),
                              (y_center - 0.5 * height),
                              (x_center + 0.5 * width),
                              (y_center + 0.5 * height)]
    return torch.stack(tensors=bounding_box_converted, dim=-1)


def bounding_box_x0y0x1y1_to_xcycwh(bounding_boxes: torch.Tensor) -> torch.Tensor:
    """
    This function converts a given bounding bix of the format
    [batch size, instances, 4 (x0, y0, x1, y1)] to [batch size, instances, 4 (x center, y center, width, height)].
    :param bounding_boxes: Bounding box of shape [batch size, instances, 4 (x0, y0, x1, y1)]
    :return: Converted bounding box of shape [batch size, instances, 4 (x center, y center, width, height)]
    """
    x_0, y_0, x_1, y_1 = bounding_boxes.unbind(dim=-1)
    bounding_box_converted = [((x_0 + x_1) / 2),
                              ((y_0 + y_1) / 2),
                              (x_1 - x_0),
                              (y_1 - y_0)]
    return torch.stack(tensors=bounding_box_converted, dim=-1)


def relative_bounding_box_to_absolute(bounding_boxes: torch.Tensor, height: int, width: int,
                                      xcycwh: bool = False) -> torch.Tensor:
    """
    This function converts a relative bounding box to an absolute one for a given image shape. Inplace operation!
    :param bounding_boxes: (torch.Tensor) Bounding box with the format [batch size, instances, 4 (x0, y0, x1, y1)]
    :param height: (int) Height of the image
    :param width: (int) Width of the image
    :param xcycwh: (bool) True if the xcycwh format is given
    :return: (torch.Tensor) Absolute bounding box in the format [batch size, instances, 4 (x0, y0, x1, y1)]
    """
    # Case if xcycwh format is given
    if xcycwh:
        bounding_boxes = bounding_box_xcycwh_to_x0y0x1y1(bounding_boxes)
    # Apply height and width
    bounding_boxes[..., [0, 2]] = bounding_boxes[..., [0, 2]] * width
    bounding_boxes[..., [1, 3]] = bounding_boxes[..., [1, 3]] * height
    # Return bounding box in the original format
    if xcycwh:
        return bounding_box_x0y0x1y1_to_xcycwh(bounding_boxes).long()
    return bounding_boxes.long()


def absolute_bounding_box_to_relative(bounding_boxes: torch.Tensor, height: int, width: int,
                                      xcycwh: bool = False) -> torch.Tensor:
    """
    This function converts an absolute bounding box to a relative one for a given image shape. Inplace operation!
    :param bounding_boxes: (torch.Tensor) Bounding box with the format [batch size, instances, 4 (x0, y0, x1, y1)]
    :param height: (int) Height of the image
    :param width: (int) Width of the image
    :param xcycwh: (bool) True if the xcycwh format is given
    :return: (torch.Tensor) Absolute bounding box in the format [batch size, instances, 4 (x0, y0, x1, y1)]
    """
    # Case if xcycwh format is given
    if xcycwh:
        bounding_boxes = bounding_box_xcycwh_to_x0y0x1y1(bounding_boxes)
    # Apply height and width
    bounding_boxes[..., [0, 2]] = bounding_boxes[..., [0, 2]] / width
    bounding_boxes[..., [1, 3]] = bounding_boxes[..., [1, 3]] / height
    # Return bounding box in the original format
    if xcycwh:
        return bounding_box_x0y0x1y1_to_xcycwh(bounding_boxes)
    return bounding_boxes
def giou(bounding_box_1: torch.Tensor, bounding_box_2: torch.Tensor,
         return_iou: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Function computes the general IoU for two given bounding boxes
    :param bounding_box_1: (torch.Tensor) Bounding box prediction of shape (batch size, instances, 4)
    :param bounding_box_2: (torch.Tensor) Bounding box labels of shape (batch size, instances, 4)
    :param return_iou: (bool) If true the normal IoU is also returned
    :return: (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) GIoU loss value for each sample and iou optimal
    """
    # Get areas of bounding boxes
    prediction_area = (bounding_box_1[..., 2] - bounding_box_1[..., 0]) * (
            bounding_box_1[..., 3] - bounding_box_1[..., 1])
    label_area = (bounding_box_2[..., 2] - bounding_box_2[..., 0]) * (bounding_box_2[..., 3] - bounding_box_2[..., 1])
    # Calc anchors
    left_top_anchors = torch.max(bounding_box_1[..., None, :2], bounding_box_2[..., :2])
    right_bottom_anchors = torch.min(bounding_box_1[..., None, 2:], bounding_box_2[..., 2:])
    # Calc width and height and clamp if needed
    width_height = (right_bottom_anchors - left_top_anchors).clamp(min=0.0)
    # Calc intersection
    intersection = width_height[..., 0] * width_height[..., 1]
    # Calc union
    union = prediction_area + label_area - intersection
    # Calc IoU
    iou = (intersection / union)
    # Calc anchors for smallest convex hull
    left_top_anchors_convex_hull = torch.min(bounding_box_1[..., :2], bounding_box_2[..., :2])
    right_bottom_anchors_convex_hull = torch.max(bounding_box_1[..., 2:], bounding_box_2[..., 2:])
    # Calc width and height and clamp if needed
    width_height_convex_hull = (right_bottom_anchors_convex_hull - left_top_anchors_convex_hull).clamp(min=0.0)
    # Calc area of convex hull
    area_convex_hull = width_height_convex_hull[..., 0] * width_height_convex_hull[..., 1]
    # Calc gIoU
    giou = (iou - ((area_convex_hull - union) / area_convex_hull))
    # Return also the iou if needed
    if return_iou:
        return giou, iou
    return giou


def giou_for_matching(bounding_box_1: torch.Tensor, bounding_box_2: torch.Tensor) -> torch.Tensor:
    """
    Function computes the general IoU for two given bounding boxes
    :param bounding_box_1: (torch.Tensor) Bounding box prediction of shape (batch size, instances, 4)
    :param bounding_box_2: (torch.Tensor) Bounding box labels of shape (batch size, instances, 4)
    :return: (torch.Tensor) GIoU matrix for matching
    """
    # Get areas of bounding boxes
    bounding_box_1_area = (bounding_box_1[:, 2] - bounding_box_1[:, 0]) * (bounding_box_1[:, 3] - bounding_box_1[:, 1])
    bounding_box_2_area = (bounding_box_2[:, 2] - bounding_box_2[:, 0]) * (bounding_box_2[:, 3] - bounding_box_2[:, 1])
    # Calc anchors
    left_top_anchors = torch.max(bounding_box_1[:, None, :2], bounding_box_2[:, :2])
    right_bottom_anchors = torch.min(bounding_box_1[:, None, 2:], bounding_box_2[:, 2:])
    # Calc width and height and clamp if needed
    width_height = (right_bottom_anchors - left_top_anchors).clamp(min=0.0)
    # Calc intersection
    intersection = width_height[:, :, 0] * width_height[:, :, 1]
    # Calc union
    union = bounding_box_1_area[:, None] + bounding_box_2_area - intersection
    # Calc IoU
    iou = (intersection / union)
    # Calc anchors for smallest convex hull
    left_top_anchors_convex_hull = torch.min(bounding_box_1[:, None, :2], bounding_box_2[..., :2])
    right_bottom_anchors_convex_hull = torch.max(bounding_box_1[:, None, 2:], bounding_box_2[..., 2:])
    # Calc width and height and clamp if needed
    width_height_convex_hull = (right_bottom_anchors_convex_hull - left_top_anchors_convex_hull).clamp(min=0.0)
    # Calc area of convex hull
    area_convex_hull = width_height_convex_hull[:, :, 0] * width_height_convex_hull[:, :, 1]
    # Calc gIoU
    giou = (iou - ((area_convex_hull - union) / area_convex_hull))
    return giou

class Augmentation(object):
    """
    Super class for all augmentations.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        pass

    def need_labels(self) -> None:
        """
        Method should return if the labels are needed for the augmentation
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> None:
        """
        Call method is used to apply the augmentation
        :param args: Will be ignored
        :param kwargs: Will be ignored
        """
        raise NotImplementedError()


class VerticalFlip(Augmentation):
    """
    This class implements vertical flipping for instance segmentation.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(VerticalFlip, self).__init__()

    def need_labels(self) -> bool:
        """
        Method returns that the labels are needed for the augmentation
        :return: (Bool) True will be returned
        """
        return True

    def __call__(self, input: torch.tensor, instances: torch.tensor,
                 bounding_boxes: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Flipping augmentation (only horizontal)
        :param image: (torch.Tensor) Input image of shape [channels, height, width]
        :param instances: (torch.Tenor) Instances segmentation maps of shape [instances, height, width]
        :param bounding_boxes: (torch.Tensor) Bounding boxes of shape [instances, 4 (x1, y1, x2, y2)]
        :return: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) Input flipped, instances flipped & BBs flipped
        """
        # Flip input
        input_flipped = input.flip(dims=(2,))
        # Flip instances
        instances_flipped = instances.flip(dims=(2,))
        # Flip bounding boxes
        image_center = torch.tensor((input.shape[2] // 2, input.shape[1] // 2))
        bounding_boxes[:, [0, 2]] += 2 * (image_center - bounding_boxes[:, [0, 2]])
        bounding_boxes_w = torch.abs(bounding_boxes[:, 0] - bounding_boxes[:, 2])
        bounding_boxes[:, 0] -= bounding_boxes_w
        bounding_boxes[:, 2] += bounding_boxes_w
        return input_flipped, instances_flipped, bounding_boxes


class ElasticDeformation(Augmentation):
    """
    This class implement random elastic deformation of a given input image
    """

    def __init__(self, alpha: float = 125, sigma: float = 20) -> None:
        """
        Constructor method
        :param alpha: (float) Alpha coefficient which represents the scaling
        :param sigma: (float) Sigma coefficient which represents the elastic factor
        """
        # Call super constructor
        super(ElasticDeformation, self).__init__()
        # Save parameters
        self.alpha = alpha
        self.sigma = sigma

    def need_labels(self) -> bool:
        """
        Method returns that the labels are needed for the augmentation
        :return: (Bool) True will be returned
        """
        return False

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Method applies the random elastic deformation
        :param image: (torch.Tensor) Input image
        :return: (torch.Tensor) Transformed input image
        """
        # Convert torch tensor to numpy array for scipy
        image = image.numpy()
        # Save basic shape
        shape = image.shape[1:]
        # Sample offsets
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        # Perform deformation
        for index in range(image.shape[0]):
            image[index] = map_coordinates(image[index], indices, order=1).reshape(shape)
        return torch.from_numpy(image)


class NoiseInjection(Augmentation):
    """
    This class implements vertical flipping for instance segmentation.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.25) -> None:
        """
        Constructor method
        :param mean: (Optional[float]) Mean of the gaussian noise
        :param std: (Optional[float]) Standard deviation of the gaussian noise
        """
        # Call super constructor
        super(NoiseInjection, self).__init__()
        # Save parameter
        self.mean = mean
        self.std = std

    def need_labels(self) -> bool:
        """
        Method returns that the labels are needed for the augmentation
        :return: (Bool) False will be returned
        """
        return False

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """
        Method injects gaussian noise to the given input image
        :param image: (torch.Tensor) Input image
        :return: (torch.Tensor) Transformed input image
        """
        # Get noise
        noise = self.mean + torch.randn_like(input) * self.std
        # Apply nose to image
        input = input + noise
        return input

class CellInstanceSegmentation(Dataset):
    """
    This dataset implements the cell instance segmentation dataset for the DETR model.
    Dataset source: https://github.com/ChristophReich1996/BCS_Data/tree/master/Cell_Instance_Segmentation_Regular_Traps
    """

    def __init__(self, path: str = "",
                 normalize: bool = True,
                 normalization_function: Callable[[torch.Tensor], torch.Tensor] = normalize,
                 augmentation: Tuple[Augmentation, ...] = (
                         VerticalFlip(), NoiseInjection(), ElasticDeformation()),
                 augmentation_p: float = 0.5, return_absolute_bounding_box: bool = False,
                 downscale: bool = True, downscale_shape: Tuple[int, int] = (256, 256)) -> None:


        """
        Constructor method
        :param path: (str) Path to dataset
        :param normalize: (bool) If true normalization_function is applied
        :param normalization_function: (Callable[[torch.Tensor], torch.Tensor]) Normalization function
        :param augmentation: (Tuple[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]) Tuple of
        augmentation functions to be applied
        :param augmentation_p: (float) Probability that an augmentation is utilized
        :param downscale: (bool) If true images and segmentation maps will be downscaled to a size of 256 X 256
        :param downscale_shape: (Tuple[int, int]) Target shape is downscale is utilized
        :param return_absolute_bounding_box: (Bool) If true the absolute bb is returned else the relative bb is returned
        """
        # Save parameters
        self.normalize = normalize
        self.normalization_function = normalization_function
        self.augmentation = augmentation
        self.augmentation_p = augmentation_p
        self.return_absolute_bounding_box = return_absolute_bounding_box
        self.downscale = downscale
        self.downscale_shape = downscale_shape
        self.path = path

     
         



        # Get paths of input images
        self.inputs = []
        for file in sorted(os.listdir(os.path.join(path, "new_x"))):
            self.inputs.append(os.path.join(path, "new_x", file))
        self.inputs = sorted(self.inputs)

        
        # Get paths of class labels
        self.class_labels = []
        for file in sorted(os.listdir(os.path.join(path, "new_labels"))):
            self.class_labels.append(os.path.join(path, "new_labels", file))
        self.class_labels = sorted(self.class_labels)
        
        # Get paths of bounding boxes
        self.bounding_boxes = []
        for file in sorted(os.listdir(os.path.join(path, "new_bbox"))):
            self.bounding_boxes.append(os.path.join(path, "new_bbox", file))
        
        self.bounding_boxes = sorted(self.bounding_boxes)    

        # Get paths of instances
        self.instances = []
        for file in sorted(os.listdir(os.path.join(path, "new_instances"))):
            self.instances.append(os.path.join(path, "new_instances", file))
        
        self.instances = sorted(self.instances) 
        

    def __len__(self) -> int:
        """
        Method returns the length of the dataset
        :return: (int) Length of the dataset
        """
        return len(self.inputs)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item method
        :param item: (int) Item to be returned of the dataset
        :return: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) Tuple including input image,
        bounding box, class label and instances.
        """
        # Load data
        filename = self.inputs[item].split("/")[-1].split('_')[0] + ".bmp"
        s = torch.load(self.inputs[item]).tolist()
        input = Image.open(os.path.join(self.path, "x",filename))
        # input = Image.open(self.inputs[item])
        input = np.array(input)
        ####
        input = input[int(s[0]):int(s[1]),int(s[2]):int(s[3]),:]
        ####
        input = (0.2989 * input[:,:,0] + 0.5870 *input[:,:,1]+ 0.1140 *input[:,:,2])/3
        input = torch.Tensor(input)
        input = input.unsqueeze(dim=0)

        file_name = self.inputs[item].split("/")[-1].split('.')[0]
        
        
        instances = torch.load(self.instances[item])
        bounding_boxes = torch.load(self.bounding_boxes[item])
        # print(len(self.mydict[file_name]))
        # print(bounding_boxes)
        # print(bounding_boxes.shape)
        # print(file_name)
        class_labels = torch.load(self.class_labels[item])
        # Encode class labels as one-hot
        class_labels = to_one_hot(class_labels.clamp(max=2.0), num_classes=2 + 1)

        # Normalize input if utilized
        if self.normalize:
            input = self.normalization_function(input)
        # Apply augmentation if needed
        if np.random.random() < self.augmentation_p and self.augmentation is not None:
            # Get augmentation
            augmentation_to_be_applied = np.random.choice(self.augmentation)
            # Apply augmentation
            if augmentation_to_be_applied.need_labels():
                input, instances, bounding_boxes = augmentation_to_be_applied(input, instances, bounding_boxes)
            else:
                input = augmentation_to_be_applied(input)
        
        # Downscale data to 256 x 256 if utilized
        if self.downscale:
            # Apply height and width
            bounding_boxes[..., [0, 2]] = bounding_boxes[..., [0, 2]] * (self.downscale_shape[0] / input.shape[-1])
            bounding_boxes[..., [1, 3]] = bounding_boxes[..., [1, 3]] * (self.downscale_shape[1] / input.shape[-2])
            input = F.interpolate(input=input.unsqueeze(dim=0),
                                  size=self.downscale_shape, mode="bicubic", align_corners=False)[0]
            instances = (F.interpolate(input=instances.unsqueeze(dim=0),
                                       size=self.downscale_shape, mode="bilinear", align_corners=False)[
                             0] > 0.75).float()
        # Convert absolute bounding box to relative bounding box of utilized
        if not self.return_absolute_bounding_box:
            bounding_boxes = absolute_bounding_box_to_relative(bounding_boxes=bounding_boxes,
                                                                    height=input.shape[1], width=input.shape[2])
        return input, instances, bounding_box_x0y0x1y1_to_xcycwh(bounding_boxes), class_labels


def collate_function_cell_instance_segmentation(
        batch: List[Tuple[torch.Tensor]]) -> \
        Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Collate function of instance segmentation dataset.
    :param batch: (Tuple[Iterable[torch.Tensor], Iterable[torch.Tensor], Iterable[torch.Tensor], Iterable[torch.Tensor]])
    Batch of input data, instances maps, bounding boxes and class labels
    :return: (Tuple[torch.Tensor, Iterable[torch.Tensor], Iterable[torch.Tensor], Iterable[torch.Tensor]]) Batched input
    data, instances, bounding boxes and class labels are stored in a list due to the different instances.
    """
    return torch.stack([input_samples[0] for input_samples in batch], dim=0), \
           [input_samples[1] for input_samples in batch], \
           [input_samples[2] for input_samples in batch], \
           [input_samples[3] for input_samples in batch]

"""# Loss"""

class HungarianMatcher(nn.Module):
    """
    This class implements a hungarian algorithm based matcher for DETR.
    """

    def __init__(self, weight_classification: float = 1.0,
                 weight_bb_l1: float = 1.0,
                 weight_bb_giou: float = 1.0) -> None:
        # Call super constructor
        super(HungarianMatcher, self).__init__()
        # Save parameters
        self.weight_classification = weight_classification
        self.weight_bb_l1 = weight_bb_l1
        self.weight_bb_giou = weight_bb_giou

    def __repr__(self):
        """
        Get representation of the matcher module
        :return: (str) String including information
        """
        return "{}, W classification:{}, W BB L1:{}, W BB gIoU".format(self.__class__.__name__,
                                                                       self.weight_classification, self.weight_bb_l1,
                                                                       self.weight_bb_giou)

    @torch.no_grad()
    def forward(self, prediction_classification: torch.Tensor,
                prediction_bounding_box: torch.Tensor,
                label_classification: Tuple[torch.Tensor],
                label_bounding_box: Tuple[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass computes the permutation produced by the hungarian algorithm.
        :param prediction_classification: (torch.Tensor) Classification prediction (batch size, # queries, classes + 1)
        :param prediction_bounding_box: (torch.Tensor) BB predictions (batch size, # queries, 4)
        :param label_classification: (Tuple[torch.Tensor]) Classification label batched [(instances, classes + 1)]
        :param label_bounding_box: (Tuple[torch.Tensor]) BB label batched [(instances, 4)]
        :return: (torch.Tensor) Permutation of shape (batch size, instances)
        """
        # Save shapes
        batch_size, number_of_queries = prediction_classification.shape[:2]
        # Get number of instances in each training sample
        number_of_instances = [label_bounding_box_instance.shape[0] for label_bounding_box_instance in
                               label_bounding_box]
        # Flatten  to shape [batch size * # queries, classes + 1]
        prediction_classification = prediction_classification.flatten(start_dim=0, end_dim=1)
        # Flatten  to shape [batch size * # queries, 4]
        prediction_bounding_box = prediction_bounding_box.flatten(start_dim=0, end_dim=1)
        # Class label to index
        # Concat labels
        label_classification = torch.cat([instance.argmax(dim=-1) for instance in label_classification], dim=0)
        label_bounding_box = torch.cat([instance for instance in label_bounding_box], dim=0)
        # Compute classification cost
        cost_classification = -prediction_classification[:, label_classification.long()]
        # Compute the L1 cost of bounding boxes
        cost_bounding_boxes_l1 = torch.cdist(prediction_bounding_box, label_bounding_box, p=1)
        # Compute gIoU cost of bounding boxes
        cost_bounding_boxes_giou = -giou_for_matching(
            bounding_box_xcycwh_to_x0y0x1y1(prediction_bounding_box),
            bounding_box_xcycwh_to_x0y0x1y1(label_bounding_box))
        # Construct cost matrix
        cost_matrix = self.weight_classification * cost_classification \
                      + self.weight_bb_l1 * cost_bounding_boxes_l1 \
                      + self.weight_bb_giou * cost_bounding_boxes_giou
        cost_matrix = cost_matrix.view(batch_size, number_of_queries, -1).cpu().clamp(min=-1e20, max=1e20)
        # Get optimal indexes
        indexes = [linear_sum_assignment(cost_vector[index]) for index, cost_vector in
                   enumerate(cost_matrix.split(number_of_instances, dim=-1))]
        # Convert indexes to list of prediction index and label index
        return [(torch.as_tensor(index_prediction, dtype=torch.int), torch.as_tensor(index_label, dtype=torch.int)) for
                index_prediction, index_label in indexes]

class LovaszHingeLoss(nn.Module):
    """
    This class implements the lovasz hinge loss which is the continuous of the IoU for binary segmentation.
    Source: https://github.com/bermanmaxim/LovaszSoftmax
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(LovaszHingeLoss, self).__init__()

    def _calc_grad(self, label_sorted: torch.Tensor) -> torch.Tensor:
        """
        Method computes the gradients of the sorted and flattened label
        :param label_sorted: (torch.Tensor) Sorted and flattened label of shape [n]
        :return: (torch.Tensor) Gradient tensor
        """
        # Calc sum of labels
        label_sum = label_sorted.sum()
        # Calc intersection
        intersection = label_sum - label_sorted.cumsum(dim=0)
        # Calc union
        union = label_sum + (1 - label_sorted).cumsum(dim=0)
        # Calc iou
        iou = 1.0 - (intersection / union)
        # Calc grad
        iou[1:] = iou[1:] - iou[0:-1]
        return iou

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the dice loss
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Dice loss value
        """
        # Flatten both tensors
        prediction = prediction.flatten(start_dim=0)
        label = label.flatten(start_dim=0)
        # Get signs of the label
        signs = 2.0 * label - 1.0
        # Get error
        error = 1.0 - prediction * signs
        # Sort errors
        errors_sorted, permutation = torch.sort(error, dim=0, descending=True)
        # Apply permutation to label
        label_sorted = label[permutation]
        # Calc grad of permuted label
        grad = self._calc_grad(label_sorted)
        # Calc final loss
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss


class DiceLoss(nn.Module):
    """
    This class implements the dice loss for multiple instances
    """

    def __init__(self, smooth_factor: float = 1.0) -> None:
        # Call super constructor
        super(DiceLoss, self).__init__()
        # Save parameter
        self.smooth_factor = smooth_factor

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, smooth factor={}".format(self.__class__.__name__, self.smooth_factor)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the dice loss
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Dice loss value
        """
        # Flatten both tensors
        prediction = prediction.flatten(start_dim=0)
        label = label.flatten(start_dim=0)
        # Calc dice loss
        loss = torch.tensor(1.0, dtype=torch.float32, device=prediction.device) \
               - ((2.0 * torch.sum(torch.mul(prediction, label)) + self.smooth_factor)
                  / (torch.sum(prediction) + torch.sum(label) + self.smooth_factor))
        return loss


class FocalLoss(nn.Module):
    """
    This class implements the segmentation focal loss.
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        """
        Constructor method
        :param alpha: (float) Alpha constant
        :param gamma: (float) Gamma constant (see paper)
        """
        # Call super constructor
        super(FocalLoss, self).__init__()
        # Save parameters
        self.alpha = alpha
        self.gamma = gamma

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, alpha={}, gamma={}".format(self.__class__.__name__, self.alpha, self.gamma)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the binary cross entropy loss of segmentation masks
        :param prediction: (torch.Tensor) Prediction probability
        :param label: (torch.Tensor) Label one-hot encoded
        :return: (torch.Tensor) Loss value
        """
        # Calc binary cross entropy loss as normal
        binary_cross_entropy_loss = -(label * torch.log(prediction.clamp(min=1e-12))
                                      + (1.0 - label) * torch.log((1.0 - prediction).clamp(min=1e-12)))
        # Calc focal loss factor based on the label and the prediction
        focal_factor = prediction * label + (1.0 - prediction) * (1.0 - label)
        # Calc final focal loss
        loss = ((1.0 - focal_factor) ** self.gamma * binary_cross_entropy_loss * self.alpha).sum(dim=1).mean()
        return loss


class LovaszSoftmaxLoss(nn.Module):
    """
    Implementation of the Lovasz-Softmax loss.
    https://arxiv.org/pdf/1708.02002.pdf
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(LovaszSoftmaxLoss, self).__init__()

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the dice loss
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Dice loss value
        """
        # One hot to class num
        _, label = label.max(dim=0)
        # Flatten tensors
        classes, height, width = prediction.size()
        prediction = prediction.permute(1, 2, 0).contiguous().view(-1, classes)
        label = label.view(-1)
        # Allocate tensor for every class loss
        losses = torch.zeros(classes, dtype=torch.float, device=prediction.device)
        # Calc loss for every class
        for c in range(classes):
            # Foreground for c
            fg = (label == c).float()
            # Class prediction
            class_prediction = prediction[:, c]
            # Calc error
            errors = (Variable(fg) - class_prediction).abs()
            # Sort errors
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            # Sort foreground
            perm = perm.data
            fg_sorted = fg[perm]
            # Calc grad
            p = len(fg_sorted)
            gts = fg_sorted.sum()
            intersection = gts - fg_sorted.float().cumsum(0)
            union = gts + (1 - fg_sorted).float().cumsum(0)
            jaccard = 1. - intersection / union
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
            # Calc class loss
            losses[c] = torch.dot(errors_sorted, Variable(jaccard))
        return losses.mean()


class FocalLossMultiClass(nn.Module):
    """
    Implementation of the multi class focal loss.
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        """
        Constructor method
        :param alpha: (float) Alpha constant
        :param gamma: (float) Gamma constant (see paper)
        """
        # Call super constructor
        super(FocalLossMultiClass, self).__init__()
        # Save parameters
        self.alpha = alpha
        self.gamma = gamma

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, alpha={}, gamma={}".format(self.__class__.__name__, self.alpha, self.gamma)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the binary cross entropy loss of segmentation masks
        :param prediction: (torch.Tensor) Prediction probability
        :param label: (torch.Tensor) Label one-hot encoded
        :return: (torch.Tensor) Loss value
        """
        # Calc binary cross entropy loss as normal
        cross_entropy_loss = - (label * torch.log(prediction.clamp(min=1e-12))).sum(dim=0)
        # Calc focal loss factor based on the label and the prediction
        focal_factor = (prediction * label + (1.0 - prediction) * (1.0 - label))
        # Calc final focal loss
        loss = ((1.0 - focal_factor) ** self.gamma * cross_entropy_loss * self.alpha).sum(dim=0).mean()
        return loss


class MultiClassSegmentationLoss(nn.Module):
    """
    Multi class segmentation loss for the case if a softmax is utilized as the final activation.
    """

    def __init__(self, dice_loss: nn.Module = DiceLoss(),
                 focal_loss: nn.Module = FocalLossMultiClass(),
                 lovasz_softmax_loss: nn.Module = LovaszSoftmaxLoss(),
                 w_dice: float = 1.0, w_focal: float = 0.1, w_lovasz_softmax: float = 0.0) -> None:
        # Call super constructor
        super(MultiClassSegmentationLoss, self).__init__()
        # Save parameters
        self.dice_loss = dice_loss
        self.focal_loss = focal_loss
        self.lovasz_softmax_loss = lovasz_softmax_loss
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_lovasz_softmax = w_lovasz_softmax

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, {}, w_focal={}, {}, w_dice={}, " \
               "{}, w_lovasz_softmax={}".format(self.__class__.__name__,
                                                self.dice_loss.__class__.__name__,
                                                self.w_dice,
                                                self.focal_loss.__class__.__name__,
                                                self.w_focal,
                                                self.lovasz_softmax_loss.__class__.__name__,
                                                self.w_lovasz_softmax)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the segmentation loss
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Loss value
        """
        return self.w_dice * self.dice_loss(prediction, label) \
               + self.w_focal * self.focal_loss(prediction, label) \
               + self.w_lovasz_softmax * self.lovasz_softmax_loss(prediction, label)


class SegmentationLoss(nn.Module):
    """
    This class implement the segmentation loss.
    """

    def __init__(self, dice_loss: nn.Module = DiceLoss(),
                 focal_loss: nn.Module = FocalLoss(),
                 lovasz_hinge_loss: nn.Module = LovaszHingeLoss(),
                 w_dice: float = 1.0, w_focal: float = 0.2, w_lovasz_hinge: float = 0.0) -> None:
        # Call super constructor
        super(SegmentationLoss, self).__init__()
        # Save parameters
        self.dice_loss = dice_loss
        self.focal_loss = focal_loss
        self.lovasz_hinge_loss = lovasz_hinge_loss
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_lovasz_hinge = w_lovasz_hinge

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, {}, w_focal={}, {}, w_dice={}, " \
               "{}, w_lovasz_hinge={}".format(self.__class__.__name__,
                                              self.dice_loss.__class__.__name__,
                                              self.w_dice,
                                              self.focal_loss.__class__.__name__,
                                              self.w_focal,
                                              self.lovasz_hinge_loss.__class__.__name__,
                                              self.w_lovasz_hinge)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the segmentation loss
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Loss value
        """
        return self.w_dice * self.dice_loss(prediction, label) \
               + self.w_focal * self.focal_loss(prediction, label) \
               + self.w_lovasz_hinge * self.lovasz_hinge_loss(prediction, label)


class BoundingBoxGIoULoss(nn.Module):
    """
    This class implements the generalized bounding box iou proposed in:
    https://giou.stanford.edu/
    This implementation is highly based on the torchvision bb iou implementation and on:
    https://github.com/facebookresearch/detr/blob/be9d447ea3208e91069510643f75dadb7e9d163d/util/box_ops.py
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(BoundingBoxGIoULoss, self).__init__()

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}".format(self.__class__.__name__)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the GIoU
        :param prediction: (torch.Tensor) Bounding box prediction of shape (batch size, instances, 4)
        :param label: (torch.Tensor) Bounding box labels of shape (batch size, instances, 4)
        :return: (torch.Tensor) GIoU loss value
        """
        return 1.0 - giou(bounding_box_1=prediction, bounding_box_2=label).diagonal().mean()


class BoundingBoxLoss(nn.Module):
    """
    This class implements the bounding box loss proposed in:
    https://arxiv.org/abs/2005.12872
    """

    def __init__(self, iou_loss_function: nn.Module = BoundingBoxGIoULoss(),
                 l1_loss_function: nn.Module = nn.L1Loss(reduction="mean"), weight_iou: float = 0.4,
                 weight_l1: float = 0.6) -> None:
        """
        Constructor method
        :param iou_loss_function: (nn.Module) Loss function module of iou loss
        :param l1_loss_function: (nn.Module) Loss function module of l1 loss
        :param weight_iou: (float) Weights factor of the iou loss
        :param weight_l1: (float) Weights factor of the l1 loss
        """
        # Call super constructor
        super(BoundingBoxLoss, self).__init__()
        # Save parameters
        self.iou_loss_function = iou_loss_function
        self.l1_loss_function = l1_loss_function
        self.weight_iou = weight_iou
        self.weight_l1 = weight_l1

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, {}, w_iou={}, {}, w_l1={}".format(self.__class__.__name__,
                                                      self.iou_loss_function.__class__.__name__,
                                                      self.weight_l1, self.l1_loss_function.__class__.__name__,
                                                      self.weight_l1)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the combined loss
        :param prediction: (torch.Tensor) Bounding box prediction of shape (batch size, instances, 4)
        :param label: (torch.Tensor) Bounding box labels of shape (batch size, instances, 4)
        :return: (torch.Tensor) Loss value
        """
        return self.weight_iou * self.iou_loss_function(prediction, label) \
               + self.weight_l1 * self.l1_loss_function(prediction, label)


class ClassificationLoss(nn.Module):
    """
    This class implements a cross entropy classification loss
    """

    def __init__(self, class_weights=torch.tensor([0.5, 1.5, 1.5], dtype=torch.float)) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(ClassificationLoss, self).__init__()
        # Save parameter
        self.class_weights = class_weights

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, class weights:{}".format(self.__class__.__name__, self.class_weights)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor, ohem: bool = False) -> torch.Tensor:
        """
        Forward pass computes loss value
        :param prediction: (torch.Tensor) Prediction one hot encoded with shape (batch size, instances, classes + 1)
        :param label: (torch.Tensor) Label one hot encoded with shape (batch size, instances, classes + 1)
        :param ohem: (bool) If true batch size is not reduced for online hard example mining
        :return: (torch.Tensor) Loss value
        """
        # Compute loss
        if ohem:
            return (- label * torch.log(prediction.clamp(min=1e-12))
                    * self.class_weights.to(label.device)).sum(dim=-1).mean(dim=-1)
        return (- label * torch.log(prediction.clamp(min=1e-12))
                * self.class_weights.to(label.device)[:prediction.shape[-1]]).sum(dim=-1).mean()


class InstanceSegmentationLoss(nn.Module):
    """
    This class combines all losses for instance segmentation
    """

    def __init__(self, classification_loss: nn.Module = ClassificationLoss(),
                 bounding_box_loss: nn.Module = BoundingBoxLoss(),
                 segmentation_loss: nn.Module = SegmentationLoss(),
                 matcher: nn.Module = HungarianMatcher(),
                 w_classification: float = 1.0, w_bounding_box: float = 1.0, w_segmentation: float = 1.0,
                 ohem: bool = False, ohem_faction: float = 0.75) -> None:
        """
        Constructor method
        :param classification_loss: (nn.Module) Classification loss function
        :param bounding_box_loss: (nn.Module) Bounding box loss function
        :param segmentation_loss: (nn.Module) Segmentation loss function
        :param matcher: (nn.Module) Matcher module to estimate the best permutation of label prediction
        :param w_classification: (float) Weights factor of the classification loss
        :param w_bounding_box: (float) Weights factor of the bounding box loss
        :param w_segmentation: (float) Weights factor of the segmentation loss
        :param ohem: (bool) True if hard example mining should be utilized
        :param ohem_faction: (float) Fraction of the whole batch size which is returned after ohm
        """
        # Call super constructor
        super(InstanceSegmentationLoss, self).__init__()
        # Save parameters
        self.classification_loss = classification_loss
        self.bounding_box_loss = bounding_box_loss
        self.segmentation_loss = segmentation_loss
        self.matcher = matcher
        self.w_classification = w_classification
        self.w_bounding_box = w_bounding_box
        self.w_segmentation = w_segmentation
        self.ohem = ohem
        self.ohem_faction = ohem_faction

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, Classification Loss:{}, w_classification={}, Bounding Box Loss:{}, w_classification={}, " \
               "Segmentation Loss:{}, w_classification={}, Matcher:{}" \
            .format(self.__class__.__name__,
                    self.classification_loss, self.w_classification,
                    self.bounding_box_loss, self.w_bounding_box,
                    self.segmentation_loss, self.w_segmentation,
                    self.matcher)

    def _construct_full_classification_label(self, label: List[torch.Tensor],
                                             number_of_predictions: int) -> torch.Tensor:
        """
        Method fills a given label with one hot encoded no-object labels
        :param label: (Tuple[torch.Tensor]) Tuple of each batch instance with variable number of instances
        :param number_of_predictions: (int) Number of predictions from the network
        :return: (torch.Tensor) Filled tensor with no-object classes [batch size, # predictions, classes + 1]
        """
        # Init new label
        new_label = torch.zeros(len(label), number_of_predictions, label[0].shape[-1])
        # Set no-object class in new label
        no_object_vector = torch.zeros(number_of_predictions, label[0].shape[-1])
        no_object_vector[:, 0] = 1.0
        new_label[:, :] = no_object_vector
        # New label to device
        new_label = new_label.to(label[0].device)
        # Iterate over all batch instances
        for index, batch_instance in enumerate(label):
            # Add existing label to new label
            new_label[index, :batch_instance.shape[0]] = batch_instance
        return new_label

    def apply_permutation(self, prediction: torch.Tensor, label: List[torch.Tensor],
                          indexes: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Method applies a given permutation to the prediction and the label
        :param prediction: (torch.Tensor) Prediction tensor of shape [batch size, # predictions, ...]
        :param label: (Tuple[torch.Tensor]) Label of shape len([[instances, ...]])= batch size
        :param indexes: (List[Tuple[torch.Tensor, torch.Tensor]])) Permutation indexes for each instance
        :return: (Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]) Prediction and label with permutation
        """
        # Iterate over batch size
        for batch_index in range(len(label)):
            # Apply permutation to label
            label[batch_index] = label[batch_index][indexes[batch_index][1].long()]
            # Apply permutation to prediction
            prediction[batch_index, :] = prediction[batch_index, torch.unique(
                torch.cat([indexes[batch_index][0].long(), torch.arange(0, prediction[batch_index].shape[0]).long()],
                          dim=0), sorted=False).long().flip(dims=(0,))]
        return prediction, label

    def forward(self, prediction_classification: torch.Tensor,
                prediction_bounding_box: torch.Tensor,
                prediction_segmentation: torch.Tensor,
                label_classification: List[torch.Tensor],
                label_bounding_box: List[torch.Tensor],
                label_segmentation: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computes combined loss
        :param prediction_classification: (torch.Tensor) Classification prediction
        :param prediction_bounding_box: (torch.Tensor) Bounding box prediction
        :param prediction_segmentation: (torch.Tensor) Segmentation prediction
        :param label_classification: (List[torch.Tensor]) Classification label
        :param label_bounding_box: (List[torch.Tensor]) Bounding box label
        :param label_segmentation: (List[torch.Tensor]) Segmentation label
        :return: (torch.Tensor) Loss value
        """
        # print(label_classification[0].shape, prediction_classification.shape)
        # Get matching indexes
        matching_indexes = self.matcher(prediction_classification, prediction_bounding_box, label_classification,
                                        label_bounding_box)
        # Apply permutation to labels and predictions
        prediction_classification, label_classification = self.apply_permutation(prediction=prediction_classification,
                                                                                 label=label_classification,
                                                                                 indexes=matching_indexes)
        prediction_bounding_box, label_bounding_box = self.apply_permutation(prediction=prediction_bounding_box,
                                                                             label=label_bounding_box,
                                                                             indexes=matching_indexes)
        prediction_segmentation, label_segmentation = self.apply_permutation(prediction=prediction_segmentation,
                                                                             label=label_segmentation,
                                                                             indexes=matching_indexes)
        # Construct full classification label
        label_classification = self._construct_full_classification_label(label=label_classification,
                                                                         number_of_predictions=
                                                                         prediction_classification.shape[1])
        # Calc classification loss
        loss_classification = self.classification_loss(prediction_classification, label_classification, self.ohem)
        # Calc bounding box loss
        
        #print('##############################################\n'+ str(loss_classification)+ '\n##############################################\n')
        
        loss_bounding_box = torch.zeros(len(label_bounding_box), dtype=torch.float,
                                        device=prediction_segmentation.device)
        for batch_index in range(len(label_bounding_box)):
            # Calc loss for each batch instance
            loss_bounding_box[batch_index] = self.bounding_box_loss(
                bounding_box_xcycwh_to_x0y0x1y1(
                    prediction_bounding_box[batch_index, :label_bounding_box[batch_index].shape[0]]),
                bounding_box_xcycwh_to_x0y0x1y1(label_bounding_box[batch_index]) )
        #print('##############################################\n'+ str(loss_bounding_box)+ '\n##############################################\n')
        # Calc segmentation loss
        loss_segmentation = torch.zeros(len(label_bounding_box), dtype=torch.float,
                                        device=prediction_segmentation.device)
        for batch_index in range(len(label_segmentation)):
            # Calc loss for each batch instance
            loss_segmentation[batch_index] = self.segmentation_loss(
                prediction_segmentation[batch_index, :label_segmentation[batch_index].shape[0]],
                label_segmentation[batch_index])
        # Perform online hard example mining if utilized
        if self.ohem:
            # Calc full loss for each batch instance
            loss = self.w_classification * loss_classification + self.w_bounding_box * loss_bounding_box \
                   + self.w_segmentation * loss_segmentation
            # Perform arg sort to get highest losses
            sorted_indexes = torch.argsort(loss, descending=True)
            # Get indexes with the highest loss and apply ohem fraction
            sorted_indexes = sorted_indexes[:min(int(self.ohem_faction * len(label_segmentation)), 1)]
            # Get corresponding losses and perform mean reduction
            return self.w_classification * loss_classification[sorted_indexes].mean(), \
                   self.w_bounding_box * loss_bounding_box[sorted_indexes].mean(), \
                   self.w_segmentation * loss_segmentation[sorted_indexes].mean()
        return self.w_classification * loss_classification, self.w_bounding_box * loss_bounding_box.mean(), \
               self.w_segmentation * loss_segmentation.mean()

"""# DETR

## backbone
"""

class ResNetBlock(nn.Module):
    """
    This class implements a simple Res-Net block with two convolutions, each followed by a normalization step and an
    activation function, and a residual mapping.
    """

    def __init__(self, in_channels: int, out_channels: int, convolution: Type = nn.Conv2d,
                 normalization: Type = nn.InstanceNorm2d, activation: Type = nn.PReLU, pooling: Type = nn.AvgPool2d) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param convolution: (Type) Type of convolution to be utilized
        :param normalization: (Type) Type of normalization to be utilized
        :param activation: (Type) Type of activation function to be utilized
        :param pooling: (Type) Type of pooling operation to be utilized
        """
        # Call super constructor
        super(ResNetBlock, self).__init__()
        # Init main mapping
        self.main_mapping = nn.Sequential(
            convolution(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            normalization(num_features=in_channels, affine=True, track_running_stats=True),
            activation(),
            convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            normalization(num_features=out_channels, affine=True, track_running_stats=True),
            activation()
        )
        # Init residual mapping
        self.residual_mapping = convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                            stride=(1, 1), padding=(0, 0),
                                            bias=True) if in_channels != out_channels else nn.Identity()
        # Init pooling
        self.pooling = pooling(kernel_size=(2, 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        :param input: (torch.Tensor) Input tensor of shape (batch size, input channels, height, width)
        :return: (torch.Tensor) Output tensor of shape (batch size, output channels, height // 2, width // 2)
        """
        # Perform main mapping
        output = self.main_mapping(input)
        # Perform residual mapping
        output = output + self.residual_mapping(input)
        # Perform pooling
        output = self.pooling(output)
        return output


class StandardBlock(nn.Module):
    """
    This class implements a standard convolution block including two convolutions, each followed by a normalization and
    an activation function.
    """

    def __init__(self, in_channels: int, out_channels: int, convolution: Type = nn.Conv2d,
                 normalization: Type = nn.InstanceNorm2d, activation: Type = nn.PReLU,
                 pooling: Type = nn.AvgPool2d) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param convolution: (Type) Type of convolution to be utilized
        :param normalization: (Type) Type of normalization to be utilized
        :param activation: (Type) Type of activation function to be utilized
        :param pooling: (Type) Type of pooling operation to be utilized
        """
        # Call super constructor
        super(StandardBlock, self).__init__()
        # Init main mapping
        self.main_mapping = nn.Sequential(
            convolution(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            normalization(num_features=in_channels, affine=True, track_running_stats=True),
            activation(),
            convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            normalization(num_features=out_channels, affine=True, track_running_stats=True),
            activation()
        )
        # Init pooling
        self.pooling = pooling(kernel_size=(2, 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        :param input: (torch.Tensor) Input tensor of shape (batch size, input channels, height, width)
        :return: (torch.Tensor) Output tensor of shape (batch size, output channels, height // 2, width // 2)
        """
        # Perform main mapping
        output = self.main_mapping(input)
        # Perform pooling
        output = self.pooling(output)
        return output


class DenseNetBlock(nn.Module):
    """
    This class implements a Dense-Net block including two convolutions, each followed by a normalization and
    an activation function, and skip connections for each convolution
    """

    def __init__(self, in_channels: int, out_channels: int, convolution: Type = nn.Conv2d,
                 normalization: Type = nn.InstanceNorm2d, activation: Type = nn.PReLU,
                 pooling: Type = nn.AvgPool2d) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param convolution: (Type) Type of convolution to be utilized
        :param normalization: (Type) Type of normalization to be utilized
        :param activation: (Type) Type of activation function to be utilized
        :param pooling: (Type) Type of pooling operation to be utilized
        """
        # Call super constructor
        super(DenseNetBlock, self).__init__()
        # Calc convolution filters
        filters, additional_filters = divmod(out_channels - in_channels, 2)
        # Init fist mapping
        self.first_mapping = nn.Sequential(
            convolution(in_channels=in_channels, out_channels=filters, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            normalization(num_features=filters, affine=True, track_running_stats=True),
            activation()
        )
        # Init second mapping
        self.second_mapping = nn.Sequential(
            convolution(in_channels=in_channels + filters, out_channels=filters + additional_filters,
                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            normalization(num_features=filters + additional_filters, affine=True, track_running_stats=True),
            activation()
        )
        # Init pooling
        self.pooling = pooling(kernel_size=(2, 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        :param input: (torch.Tensor) Input tensor of shape (batch size, input channels, height, width)
        :return: (torch.Tensor) Output tensor of shape (batch size, output channels, height // 2, width // 2)
        """
        # Perform main mapping
        output = torch.cat((input, self.first_mapping(input)), dim=1)
        # Perform main mapping
        output = torch.cat((output, self.second_mapping(output)), dim=1)
        # Perform pooling
        output = self.pooling(output)
        return output


class Backbone(nn.Module):
    """
    This class implements the backbone network.
    """

    def __init__(self, channels: Tuple[Tuple[int, int], ...] = ((1, 16), (16, 32), (32, 64), (64, 128), (128, 256)),
                 block: Type = StandardBlock, convolution: Type = nn.Conv2d, normalization: Type = nn.InstanceNorm2d,
                 activation: Type = nn.PReLU, pooling: Type = nn.AvgPool2d) -> None:
        """
        Constructor method
        :param channels: (Tuple[Tuple[int, int]]) In and output channels of each block
        :param block: (Type) Basic block to be used
        :param convolution: (Type) Type of convolution to be utilized
        :param normalization: (Type) Type of normalization to be utilized
        :param activation: (Type) Type of activation function to be utilized
        :param pooling: (Type) Type of pooling operation to be utilized
        """
        # Call super constructor
        super(Backbone, self).__init__()
        # Init input convolution
        self.input_convolution = nn.Sequential(convolution(in_channels=channels[0][0], out_channels=channels[0][1],
                                                           kernel_size=(7, 7), stride=(1, 1), padding=(3, 3),
                                                           bias=True),
                                               pooling(kernel_size=(2, 2)))
        # Init blocks
        self.blocks = nn.ModuleList([
            block(in_channels=channel[0], out_channels=channel[1], convolution=convolution, normalization=normalization,
                  activation=activation, pooling=pooling) for channel in channels])
        # Init weights
        for module in self.modules():
            # Case if module is convolution
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=1)
                nn.init.constant_(module.bias, 0)
            # Deformable convolution is already initialized in the right way
            # Init PReLU
            elif isinstance(module, nn.PReLU):
                nn.init.constant_(module.weight, 0.2)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass
        :param input: (torch.Tensor) Input image of shape (batch size, input channels, height, width)
        :return: (torch.Tensor) Output tensor (batch size, output channels, height // 2 ^ depth, width // 2 ^ depth) and
        features of each stage of the backbone network
        """
        # Init list to store feature maps
        feature_maps = []
        # Forward pass of all blocks
        for index, block in enumerate(self.blocks):
            if index == 0:
                input = block(input) + self.input_convolution(input)
                feature_maps.append(input)
            else:
                input = block(input)
                feature_maps.append(input)
        return input, feature_maps

"""## transformer

"""

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation=nn.LeakyReLU, normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=nn.LeakyReLU, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=nn.LeakyReLU, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "leaky relu":
        return F.leaky_relu
    if activation == "selu":
        return F.selu
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu, gelu, glu, leaky relu or selu, not {activation}.")

"""## bounding-box-head"""

class BoundingBoxHead(nn.Module):
    """
    This class implements the feed forward bounding box head as proposed in:
    https://arxiv.org/abs/2005.12872
    """

    def __init__(self, features: Tuple[Tuple[int, int]] = ((256, 64), (64, 16), (16, 4)),
                 activation: Type = nn.PReLU) -> None:
        """
        Constructor method
        :param features: (Tuple[Tuple[int, int]]) Number of input and output features in each layer
        :param activation: (Type) Activation function to be utilized
        """
        # Call super constructor
        super(BoundingBoxHead, self).__init__()
        # Init layers
        self.layers = []
        for index, feature in enumerate(features):
            if index < len(features) - 1:
                self.layers.extend([nn.Linear(in_features=feature[0], out_features=feature[1]), activation()])
            else:
                self.layers.append(nn.Linear(in_features=feature[0], out_features=feature[1]))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of shape (batch size, instances, features)
        :return: (torch.Tensor) Output tensor of shape (batch size, instances, classes + 1 (no object))
        """
        return self.layers(input)

"""## segmentation"""

"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""


__all__ = ['PacConv2d', 'PacConvTranspose2d', 'PacPool2d',
           'pacconv2d', 'pacconv_transpose2d', 'pacpool2d', 'packernel2d', 'nd2col']

import math
from numbers import Number
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function, once_differentiable
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair

try:
    import pyinn as P

    has_pyinn = True
except ImportError:
    P = None
    has_pyinn = False
    pass


def _neg_idx(idx):
    return None if idx == 0 else -idx


def np_gaussian_2d(width, sigma=-1):
    '''Truncated 2D Gaussian filter'''
    assert width % 2 == 1
    if sigma <= 0:
        sigma = float(width) / 4

    r = np.arange(-(width // 2), (width // 2) + 1, dtype=np.float32)
    gaussian_1d = np.exp(-0.5 * r * r / (sigma * sigma))
    gaussian_2d = gaussian_1d.reshape(-1, 1) * gaussian_1d
    gaussian_2d /= gaussian_2d.sum()

    return gaussian_2d


def nd2col(input_nd, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, transposed=False,
           use_pyinn_if_possible=False):
    """
    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, *kernel_size, *L_{out})` where
          :math:`L_{out} = floor((L_{in} + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)` for non-transposed
          :math:`L_{out} = (L_{in} - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding` for transposed
    """
    n_dims = len(input_nd.shape[2:])
    kernel_size = (kernel_size,) * n_dims if isinstance(kernel_size, Number) else kernel_size
    stride = (stride,) * n_dims if isinstance(stride, Number) else stride
    padding = (padding,) * n_dims if isinstance(padding, Number) else padding
    output_padding = (output_padding,) * n_dims if isinstance(output_padding, Number) else output_padding
    dilation = (dilation,) * n_dims if isinstance(dilation, Number) else dilation

    if transposed:
        assert n_dims == 2, 'Only 2D is supported for fractional strides.'
        w_one = input_nd.new_ones(1, 1, 1, 1)
        pad = [(k - 1) * d - p for (k, d, p) in zip(kernel_size, dilation, padding)]
        input_nd = F.conv_transpose2d(input_nd, w_one, stride=stride)
        input_nd = F.pad(input_nd, (pad[1], pad[1] + output_padding[1], pad[0], pad[0] + output_padding[0]))
        stride = _pair(1)
        padding = _pair(0)

    (bs, nch), in_sz = input_nd.shape[:2], input_nd.shape[2:]
    out_sz = tuple([((i + 2 * p - d * (k - 1) - 1) // s + 1)
                    for (i, k, d, p, s) in zip(in_sz, kernel_size, dilation, padding, stride)])
    # Use PyINN if possible (about 15% faster) TODO confirm the speed-up
    if n_dims == 2 and dilation == 1 and has_pyinn and torch.cuda.is_available() and use_pyinn_if_possible:
        output = P.im2col(input_nd, kernel_size, stride, padding)
    else:
        output = F.unfold(input_nd, kernel_size, dilation, padding, stride)
        out_shape = (bs, nch) + tuple(kernel_size) + out_sz
        output = output.view(*out_shape).contiguous()
    return output


class GaussKernel2dFn(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding, dilation, channel_wise):
        ctx.kernel_size = _pair(kernel_size)
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        bs, ch, in_h, in_w = input.shape
        out_h = (in_h + 2 * ctx.padding[0] - ctx.dilation[0] * (ctx.kernel_size[0] - 1) - 1) // ctx.stride[0] + 1
        out_w = (in_w + 2 * ctx.padding[1] - ctx.dilation[1] * (ctx.kernel_size[1] - 1) - 1) // ctx.stride[1] + 1
        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_h, out_w)
        center_y, center_x = ctx.kernel_size[0] // 2, ctx.kernel_size[1] // 2
        feat_0 = cols.contiguous()[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :]
        diff_sq = (cols - feat_0).pow(2)
        if not channel_wise:
            diff_sq = diff_sq.sum(dim=1, keepdim=True)
        output = torch.exp(-0.5 * diff_sq)
        ctx.save_for_backward(input, output)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        bs, ch, in_h, in_w = input.shape
        out_h, out_w = output.shape[-2:]
        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_h, out_w)
        center_y, center_x = ctx.kernel_size[0] // 2, ctx.kernel_size[1] // 2
        feat_0 = cols.contiguous()[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :]
        diff = cols - feat_0
        grad = -0.5 * grad_output * output
        grad_diff = grad.expand_as(cols) * (2 * diff)
        grad_diff[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :] -= \
            grad_diff.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

        grad_input = F.fold(grad_diff.view(bs, ch * ctx.kernel_size[0] * ctx.kernel_size[1], -1),
                            (in_h, in_w), ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)

        return grad_input, None, None, None, None, None


class PacConv2dFn(Function):
    @staticmethod
    def forward(ctx, input, kernel, weight, bias=None, stride=1, padding=0, dilation=1, shared_filters=False):
        (bs, ch), in_sz = input.shape[:2], input.shape[2:]
        if kernel.size(1) > 1:
            raise ValueError('Non-singleton channel is not allowed for kernel.')
        ctx.input_size = in_sz
        ctx.in_ch = ch
        ctx.kernel_size = tuple(weight.shape[-2:])
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        ctx.shared_filters = shared_filters
        ctx.save_for_backward(input if (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]) else None,
                              kernel if (ctx.needs_input_grad[0] or ctx.needs_input_grad[2]) else None,
                              weight if (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]) else None)

        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)

        in_mul_k = cols.view(bs, ch, *kernel.shape[2:]) * kernel

        # matrix multiplication, written as an einsum to avoid repeated view() and permute()
        if shared_filters:
            output = torch.einsum('ijklmn,zykl->ijmn', (in_mul_k, weight))
        else:
            output = torch.einsum('ijklmn,ojkl->iomn', (in_mul_k, weight))

        if bias is not None:
            output += bias.view(1, -1, 1, 1)

        return output.clone()  # TODO understand why a .clone() is needed here

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_input = grad_kernel = grad_weight = grad_bias = None
        (bs, out_ch), out_sz = grad_output.shape[:2], grad_output.shape[2:]
        in_ch = ctx.in_ch

        input, kernel, weight = ctx.saved_tensors
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            if ctx.shared_filters:
                grad_in_mul_k = grad_output.view(bs, out_ch, 1, 1, out_sz[0], out_sz[1]) \
                                * weight.view(ctx.kernel_size[0], ctx.kernel_size[1], 1, 1)
            else:
                grad_in_mul_k = torch.einsum('iomn,ojkl->ijklmn', (grad_output, weight))
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            in_cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
            in_cols = in_cols.view(bs, in_ch, ctx.kernel_size[0], ctx.kernel_size[1], out_sz[0], out_sz[1])
        if ctx.needs_input_grad[0]:
            grad_im2col_output = grad_in_mul_k * kernel
            grad_im2col_output = grad_im2col_output.view(bs, -1, out_sz[0] * out_sz[1])

            grad_input = F.fold(grad_im2col_output,
                                ctx.input_size[:2], ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        if ctx.needs_input_grad[1]:
            grad_kernel = in_cols * grad_in_mul_k
            grad_kernel = grad_kernel.sum(dim=1, keepdim=True)
        if ctx.needs_input_grad[2]:
            in_mul_k = in_cols * kernel
            if ctx.shared_filters:
                grad_weight = torch.einsum('ijmn,ijklmn->kl', (grad_output, in_mul_k))
                grad_weight = grad_weight.view(1, 1, ctx.kernel_size[0], ctx.kernel_size[1]).contiguous()
            else:
                grad_weight = torch.einsum('iomn,ijklmn->ojkl', (grad_output, in_mul_k))
        if ctx.needs_input_grad[3]:
            grad_bias = torch.einsum('iomn->o', (grad_output,))

        return grad_input, grad_kernel, grad_weight, grad_bias, None, None, None, None


class PacConvTranspose2dFn(Function):
    @staticmethod
    def forward(ctx, input, kernel, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1,
                shared_filters=False):
        (bs, ch), in_sz = input.shape[:2], input.shape[2:]
        if kernel.size(1) > 1:
            raise ValueError('Non-singleton channel is not allowed for kernel.')
        ctx.in_ch = ch
        ctx.kernel_size = tuple(weight.shape[-2:])
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.output_padding = _pair(output_padding)
        ctx.stride = _pair(stride)
        ctx.shared_filters = shared_filters
        ctx.save_for_backward(input if (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]) else None,
                              kernel if (ctx.needs_input_grad[0] or ctx.needs_input_grad[2]) else None,
                              weight if (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]) else None)

        w = input.new_ones((ch, 1, 1, 1))
        x = F.conv_transpose2d(input, w, stride=stride, groups=ch)
        pad = [(k - 1) * d - p for (k, d, p) in zip(ctx.kernel_size, ctx.dilation, ctx.padding)]
        x = F.pad(x, (pad[1], pad[1] + ctx.output_padding[1], pad[0], pad[0] + ctx.output_padding[0]))

        cols = F.unfold(x, ctx.kernel_size, ctx.dilation, _pair(0), _pair(1))

        in_mul_k = cols.view(bs, ch, *kernel.shape[2:]) * kernel

        # matrix multiplication, written as an einsum to avoid repeated view() and permute()
        if shared_filters:
            output = torch.einsum('ijklmn,jokl->iomn', (in_mul_k, weight))
        else:
            output = torch.einsum('ijklmn,jokl->iomn', (in_mul_k, weight))

        if bias is not None:
            output += bias.view(1, -1, 1, 1)

        return output.clone()  # TODO understand why a .clone() is needed here

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_input = grad_kernel = grad_weight = grad_bias = None
        (bs, out_ch), out_sz = grad_output.shape[:2], grad_output.shape[2:]
        in_ch = ctx.in_ch
        pad = [(k - 1) * d - p for (k, d, p) in zip(ctx.kernel_size, ctx.dilation, ctx.padding)]
        pad = [(p, p + op) for (p, op) in zip(pad, ctx.output_padding)]

        input, kernel, weight = ctx.saved_tensors
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            if ctx.shared_filters:
                grad_in_mul_k = grad_output.view(bs, out_ch, 1, 1, out_sz[0], out_sz[1]) \
                                * weight.view(ctx.kernel_size[0], ctx.kernel_size[1], 1, 1)
            else:
                grad_in_mul_k = torch.einsum('iomn,jokl->ijklmn', (grad_output, weight))
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            w = input.new_ones((in_ch, 1, 1, 1))
            x = F.conv_transpose2d(input, w, stride=ctx.stride, groups=in_ch)
            x = F.pad(x, (pad[1][0], pad[1][1], pad[0][0], pad[0][1]))
            in_cols = F.unfold(x, ctx.kernel_size, ctx.dilation, _pair(0), _pair(1))
            in_cols = in_cols.view(bs, in_ch, ctx.kernel_size[0], ctx.kernel_size[1], out_sz[0], out_sz[1])
        if ctx.needs_input_grad[0]:
            grad_im2col_output = grad_in_mul_k * kernel
            grad_im2col_output = grad_im2col_output.view(bs, -1, out_sz[0] * out_sz[1])
            im2col_input_sz = [o + (k - 1) * d for (o, k, d) in zip(out_sz, ctx.kernel_size, ctx.dilation)]

            grad_input = F.fold(grad_im2col_output,
                                im2col_input_sz[:2], ctx.kernel_size, ctx.dilation, 0, 1)
            grad_input = grad_input[:, :, pad[0][0]:-pad[0][1]:ctx.stride[0], pad[1][0]:-pad[1][1]:ctx.stride[1]]
        if ctx.needs_input_grad[1]:
            grad_kernel = in_cols * grad_in_mul_k
            grad_kernel = grad_kernel.sum(dim=1, keepdim=True)
        if ctx.needs_input_grad[2]:
            in_mul_k = in_cols * kernel
            if ctx.shared_filters:
                grad_weight = torch.einsum('ijmn,ijklmn->kl', (grad_output, in_mul_k))
                grad_weight = grad_weight.view(1, 1, ctx.kernel_size[0], ctx.kernel_size[1]).contiguous()
            else:
                grad_weight = torch.einsum('iomn,ijklmn->jokl', (grad_output, in_mul_k))
        if ctx.needs_input_grad[3]:
            grad_bias = torch.einsum('iomn->o', (grad_output,))
        return grad_input, grad_kernel, grad_weight, grad_bias, None, None, None, None, None


class PacPool2dFn(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_size, stride=1, padding=0, dilation=1):
        (bs, ch), in_sz = input.shape[:2], input.shape[2:]
        if kernel.size(1) > 1 and kernel.size(1) != ch:
            raise ValueError('Incompatible input and kernel sizes.')
        ctx.input_size = in_sz
        ctx.kernel_size = _pair(kernel_size)
        ctx.kernel_ch = kernel.size(1)
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        ctx.save_for_backward(input if ctx.needs_input_grad[1] else None,
                              kernel if ctx.needs_input_grad[0] else None)

        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)

        output = cols.view(bs, ch, *kernel.shape[2:]) * kernel
        output = torch.einsum('ijklmn->ijmn', (output,))

        return output.clone()  # TODO check whether a .clone() is needed here

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, kernel = ctx.saved_tensors
        grad_input = grad_kernel = None
        (bs, ch), out_sz = grad_output.shape[:2], grad_output.shape[2:]
        if ctx.needs_input_grad[0]:
            grad_im2col_output = torch.einsum('ijmn,izklmn->ijklmn', (grad_output, kernel))
            grad_im2col_output = grad_im2col_output.view(bs, -1, out_sz[0] * out_sz[1])

            grad_input = F.fold(grad_im2col_output,
                                ctx.input_size[:2], ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        if ctx.needs_input_grad[1]:
            cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
            cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_sz[0], out_sz[1])
            grad_kernel = torch.einsum('ijmn,ijklmn->ijklmn', (grad_output, cols))
            if ctx.kernel_ch == 1:
                grad_kernel = grad_kernel.sum(dim=1, keepdim=True)

        return grad_input, grad_kernel, None, None, None, None


def packernel2d(input, mask=None, kernel_size=0, stride=1, padding=0, output_padding=0, dilation=1,
                kernel_type='gaussian', smooth_kernel_type='none', smooth_kernel=None, inv_alpha=None, inv_lambda=None,
                channel_wise=False, normalize_kernel=False, transposed=False, native_impl=False):
    kernel_size = _pair(kernel_size)
    dilation = _pair(dilation)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    stride = _pair(stride)
    output_mask = False if mask is None else True
    norm = None

    if mask is not None and mask.dtype != input.dtype:
        mask = torch.tensor(mask, dtype=input.dtype, device=input.device)

    if transposed:
        in_sz = tuple(int((o - op - 1 - (k - 1) * d + 2 * p) // s) + 1 for (o, k, s, p, op, d) in
                      zip(input.shape[-2:], kernel_size, stride, padding, output_padding, dilation))
    else:
        in_sz = input.shape[-2:]

    if mask is not None or normalize_kernel:
        mask_pattern = input.new_ones(1, 1, *in_sz)
        mask_pattern = nd2col(mask_pattern, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                              dilation=dilation, transposed=transposed)
        if mask is not None:
            mask = nd2col(mask, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                          dilation=dilation, transposed=transposed)
            if not normalize_kernel:
                norm = mask.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) \
                       / mask_pattern.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        else:
            mask = mask_pattern

    if transposed:
        stride = _pair(1)
        padding = tuple((k - 1) * d // 2 for (k, d) in zip(kernel_size, dilation))

    if native_impl:
        bs, k_ch, in_h, in_w = input.shape

        x = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)
        x = x.view(bs, k_ch, -1, *x.shape[-2:]).contiguous()

        if smooth_kernel_type == 'none':
            self_idx = kernel_size[0] * kernel_size[1] // 2
            feat_0 = x[:, :, self_idx:self_idx + 1, :, :]
        else:
            smooth_kernel_size = smooth_kernel.shape[2:]
            smooth_padding = (int(padding[0] - (kernel_size[0] - smooth_kernel_size[0]) / 2),
                              int(padding[1] - (kernel_size[1] - smooth_kernel_size[1]) / 2))
            crop = tuple(-1 * np.minimum(0, smooth_padding))
            input_for_kernel_crop = input.view(-1, 1, in_h, in_w)[:, :,
                                    crop[0]:_neg_idx(crop[0]), crop[1]:_neg_idx(crop[1])]
            smoothed = F.conv2d(input_for_kernel_crop, smooth_kernel,
                                stride=stride, padding=tuple(np.maximum(0, smooth_padding)))
            feat_0 = smoothed.view(bs, k_ch, 1, *x.shape[-2:])
        x = x - feat_0
        if kernel_type.find('_asym') >= 0:
            x = F.relu(x, inplace=True)
        # x.pow_(2)  # this causes an autograd issue in pytorch>0.4
        x = x * x
        if not channel_wise:
            x = torch.sum(x, dim=1, keepdim=True)
        if kernel_type == 'gaussian':
            x = torch.exp_(x.mul_(-0.5))  # TODO profiling for identifying the culprit of 5x slow down
            # x = torch.exp(-0.5 * x)
        elif kernel_type.startswith('inv_'):
            epsilon = 1e-4
            x = inv_alpha.view(1, -1, 1, 1, 1) \
                + torch.pow(x + epsilon, 0.5 * inv_lambda.view(1, -1, 1, 1, 1))
        else:
            raise ValueError()
        output = x.view(*(x.shape[:2] + tuple(kernel_size) + x.shape[-2:])).contiguous()
    else:
        assert (smooth_kernel_type == 'none' and
                kernel_type == 'gaussian')
        output = GaussKernel2dFn.apply(input, kernel_size, stride, padding, dilation, channel_wise)

    if mask is not None:
        output = output * mask  # avoid numerical issue on masked positions

    if normalize_kernel:
        norm = output.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

    if norm is not None:
        empty_mask = (norm == 0)
        # output = output / (norm + torch.tensor(empty_mask, dtype=input.dtype, device=input.device))
        output = output / (norm + empty_mask.clone().detach())
        output_mask = (1 - empty_mask) if output_mask else None
    else:
        output_mask = None

    return output, output_mask


def pacconv2d(input, kernel, weight, bias=None, stride=1, padding=0, dilation=1, shared_filters=False,
              native_impl=False):
    kernel_size = tuple(weight.shape[-2:])
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    if native_impl:
        # im2col on input
        im_cols = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)

        # main computation
        if shared_filters:
            output = torch.einsum('ijklmn,zykl->ijmn', (im_cols * kernel, weight))
        else:
            output = torch.einsum('ijklmn,ojkl->iomn', (im_cols * kernel, weight))

        if bias is not None:
            output += bias.view(1, -1, 1, 1)
    else:
        output = PacConv2dFn.apply(input, kernel, weight, bias, stride, padding, dilation, shared_filters)

    return output


def pacconv_transpose2d(input, kernel, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1,
                        shared_filters=False, native_impl=False):
    kernel_size = tuple(weight.shape[-2:])
    stride = _pair(stride)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    dilation = _pair(dilation)

    if native_impl:
        ch = input.shape[1]
        w = input.new_ones((ch, 1, 1, 1))
        x = F.conv_transpose2d(input, w, stride=stride, groups=ch)
        pad = [(kernel_size[i] - 1) * dilation[i] - padding[i] for i in range(2)]
        x = F.pad(x, (pad[1], pad[1] + output_padding[1], pad[0], pad[0] + output_padding[0]))
        output = pacconv2d(x, kernel, weight.permute(1, 0, 2, 3), bias, dilation=dilation,
                           shared_filters=shared_filters, native_impl=True)
    else:
        output = PacConvTranspose2dFn.apply(input, kernel, weight, bias, stride, padding, output_padding, dilation,
                                            shared_filters)

    return output


def pacpool2d(input, kernel, kernel_size, stride=1, padding=0, dilation=1, native_impl=False):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    if native_impl:
        bs, in_ch, in_h, in_w = input.shape
        out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

        # im2col on input
        im_cols = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)

        # main computation
        im_cols *= kernel
        output = im_cols.view(bs, in_ch, -1, out_h, out_w).sum(dim=2, keepdim=False)
    else:
        output = PacPool2dFn.apply(input, kernel, kernel_size, stride, padding, dilation)

    return output


class _PacConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, bias,
                 pool_only, kernel_type, smooth_kernel_type,
                 channel_wise, normalize_kernel, shared_filters, filler):
        super(_PacConvNd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.pool_only = pool_only
        self.kernel_type = kernel_type
        self.smooth_kernel_type = smooth_kernel_type
        self.channel_wise = channel_wise
        self.normalize_kernel = normalize_kernel
        self.shared_filters = shared_filters
        self.filler = filler
        if any([k % 2 != 1 for k in kernel_size]):
            raise ValueError('kernel_size only accept odd numbers')
        if smooth_kernel_type.find('_') >= 0 and int(smooth_kernel_type[smooth_kernel_type.rfind('_') + 1:]) % 2 != 1:
            raise ValueError('smooth_kernel_type only accept kernels of odd widths')
        if shared_filters:
            assert in_channels == out_channels, 'when specifying shared_filters, number of channels should not change'
        if any([p > d * (k - 1) / 2 for (p, d, k) in zip(padding, dilation, kernel_size)]):
            # raise ValueError('padding ({}) too large'.format(padding))
            pass  # TODO verify that this indeed won't cause issues
        if not pool_only:
            if self.filler in {'pool', 'crf_pool'}:
                assert shared_filters
                self.register_buffer('weight', torch.ones(1, 1, *kernel_size))
                if self.filler == 'crf_pool':
                    self.weight[(0, 0) + tuple(k // 2 for k in kernel_size)] = 0  # Eq.5, DenseCRF
            elif shared_filters:
                self.weight = Parameter(torch.Tensor(1, 1, *kernel_size))
            elif transposed:
                self.weight = Parameter(torch.Tensor(in_channels, out_channels, *kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            if bias:
                self.bias = Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)
        if kernel_type.startswith('inv_'):
            self.inv_alpha_init = float(kernel_type.split('_')[1])
            self.inv_lambda_init = float(kernel_type.split('_')[2])
            if self.channel_wise and kernel_type.find('_fixed') < 0:
                if out_channels <= 0:
                    raise ValueError('out_channels needed for channel_wise {}'.format(kernel_type))
                inv_alpha = self.inv_alpha_init * torch.ones(out_channels)
                inv_lambda = self.inv_lambda_init * torch.ones(out_channels)
            else:
                inv_alpha = torch.tensor(float(self.inv_alpha_init))
                inv_lambda = torch.tensor(float(self.inv_lambda_init))
            if kernel_type.find('_fixed') < 0:
                self.register_parameter('inv_alpha', Parameter(inv_alpha))
                self.register_parameter('inv_lambda', Parameter(inv_lambda))
            else:
                self.register_buffer('inv_alpha', inv_alpha)
                self.register_buffer('inv_lambda', inv_lambda)
        elif kernel_type != 'gaussian':
            raise ValueError('kernel_type set to invalid value ({})'.format(kernel_type))
        if smooth_kernel_type.startswith('full_'):
            smooth_kernel_size = int(smooth_kernel_type.split('_')[-1])
            self.smooth_kernel = Parameter(torch.Tensor(1, 1, *repeat(smooth_kernel_size, len(kernel_size))))
        elif smooth_kernel_type == 'gaussian':
            smooth_1d = torch.tensor([.25, .5, .25])
            smooth_kernel = smooth_1d
            for d in range(1, len(kernel_size)):
                smooth_kernel = smooth_kernel * smooth_1d.view(-1, *repeat(1, d))
            self.register_buffer('smooth_kernel', smooth_kernel.unsqueeze(0).unsqueeze(0))
        elif smooth_kernel_type.startswith('average_'):
            smooth_kernel_size = int(smooth_kernel_type.split('_')[-1])
            smooth_1d = torch.tensor((1.0 / smooth_kernel_size,) * smooth_kernel_size)
            smooth_kernel = smooth_1d
            for d in range(1, len(kernel_size)):
                smooth_kernel = smooth_kernel * smooth_1d.view(-1, *repeat(1, d))
            self.register_buffer('smooth_kernel', smooth_kernel.unsqueeze(0).unsqueeze(0))
        elif smooth_kernel_type != 'none':
            raise ValueError('smooth_kernel_type set to invalid value ({})'.format(smooth_kernel_type))

        self.reset_parameters()

    def reset_parameters(self):
        if not (self.pool_only or self.filler in {'pool', 'crf_pool'}):
            if self.filler == 'uniform':
                n = self.in_channels
                for k in self.kernel_size:
                    n *= k
                stdv = 1. / math.sqrt(n)
                if self.shared_filters:
                    stdv *= self.in_channels
                self.weight.data.uniform_(-stdv, stdv)
                if self.bias is not None:
                    self.bias.data.uniform_(-stdv, stdv)
            elif self.filler == 'linear':
                effective_kernel_size = tuple(2 * s - 1 for s in self.stride)
                pad = tuple(int((k - ek) // 2) for k, ek in zip(self.kernel_size, effective_kernel_size))
                assert self.transposed and self.in_channels == self.out_channels
                assert all(k >= ek for k, ek in zip(self.kernel_size, effective_kernel_size))
                w = 1.0
                for i, (p, s, k) in enumerate(zip(pad, self.stride, self.kernel_size)):
                    d = len(pad) - i - 1
                    w = w * (np.array((0.0,) * p + tuple(range(1, s)) + tuple(range(s, 0, -1)) + (0,) * p) / s).reshape(
                        (-1,) + (1,) * d)
                    if self.normalize_kernel:
                        w = w * np.array(tuple(((k - j - 1) // s) + (j // s) + 1.0 for j in range(k))).reshape(
                            (-1,) + (1,) * d)
                self.weight.data.fill_(0.0)
                for c in range(1 if self.shared_filters else self.in_channels):
                    self.weight.data[c, c, :] = torch.tensor(w)
                if self.bias is not None:
                    self.bias.data.fill_(0.0)
            elif self.filler in {'crf', 'crf_perturbed'}:
                assert len(self.kernel_size) == 2 and self.kernel_size[0] == self.kernel_size[1] \
                       and self.in_channels == self.out_channels
                perturb_range = 0.001
                n_classes = self.in_channels
                gauss = np_gaussian_2d(self.kernel_size[0]) * self.kernel_size[0] * self.kernel_size[0]
                gauss[self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
                if self.shared_filters:
                    self.weight.data[0, 0, :] = torch.tensor(gauss)
                else:
                    compat = 1.0 - np.eye(n_classes, dtype=np.float32)
                    self.weight.data[:] = torch.tensor(compat.reshape(n_classes, n_classes, 1, 1) * gauss)
                if self.filler == 'crf_perturbed':
                    self.weight.data.add_((torch.rand_like(self.weight.data) - 0.5) * perturb_range)
                if self.bias is not None:
                    self.bias.data.fill_(0.0)
            else:
                raise ValueError('Initialization method ({}) not supported.'.format(self.filler))
        if hasattr(self, 'inv_alpha') and isinstance(self.inv_alpha, Parameter):
            self.inv_alpha.data.fill_(self.inv_alpha_init)
            self.inv_lambda.data.fill_(self.inv_lambda_init)
        if hasattr(self, 'smooth_kernel') and isinstance(self.smooth_kernel, Parameter):
            self.smooth_kernel.data.fill_(1.0 / np.multiply.reduce(self.smooth_kernel.shape))

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', kernel_type={kernel_type}')
        if self.stride != (1,) * len(self.stride):
            s += ', stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.bias is None:
            s += ', bias=False'
        if self.smooth_kernel_type != 'none':
            s += ', smooth_kernel_type={smooth_kernel_type}'
        if self.channel_wise:
            s += ', channel_wise=True'
        if self.normalize_kernel:
            s += ', normalize_kernel=True'
        if self.shared_filters:
            s += ', shared_filters=True'
        return s.format(**self.__dict__)


class PacConv2d(_PacConvNd):
    r"""
    Args (in addition to those of Conv2d):
        kernel_type (str): 'gaussian' | 'inv_{alpha}_{lambda}[_asym][_fixed]'. Default: 'gaussian'
        smooth_kernel_type (str): 'none' | 'gaussian' | 'average_{sz}' | 'full_{sz}'. Default: 'none'
        normalize_kernel (bool): Default: False
        shared_filters (bool): Default: False
        filler (str): 'uniform'. Default: 'uniform'
    Note:
        - kernel_size only accepts odd numbers
        - padding should not be larger than :math:`dilation * (kernel_size - 1) / 2`
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 kernel_type='gaussian', smooth_kernel_type='none', normalize_kernel=False, shared_filters=False,
                 filler='uniform', native_impl=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(PacConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, False, _pair(0), bias,
            False, kernel_type, smooth_kernel_type, False, normalize_kernel, shared_filters, filler)

        self.native_impl = native_impl

    def compute_kernel(self, input_for_kernel, input_mask=None):
        return packernel2d(input_for_kernel, input_mask,
                           kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                           dilation=self.dilation, kernel_type=self.kernel_type,
                           smooth_kernel_type=self.smooth_kernel_type,
                           smooth_kernel=self.smooth_kernel if hasattr(self, 'smooth_kernel') else None,
                           inv_alpha=self.inv_alpha if hasattr(self, 'inv_alpha') else None,
                           inv_lambda=self.inv_lambda if hasattr(self, 'inv_lambda') else None,
                           channel_wise=False, normalize_kernel=self.normalize_kernel, transposed=False,
                           native_impl=self.native_impl)

    def forward(self, input_2d, input_for_kernel, kernel=None, mask=None):
        output_mask = None
        if kernel is None:
            kernel, output_mask = self.compute_kernel(input_for_kernel, mask)

        output = pacconv2d(input_2d, kernel, self.weight, self.bias, self.stride, self.padding, self.dilation,
                           self.shared_filters, self.native_impl)

        return output if output_mask is None else (output, output_mask)


class PacConvTranspose2d(_PacConvNd):
    r"""
    Args (in addition to those of ConvTranspose2d):
        kernel_type (str): 'gaussian' | 'inv_{alpha}_{lambda}[_asym][_fixed]'. Default: 'gaussian'
        smooth_kernel_type (str): 'none' | 'gaussian' | 'average_{sz}' | 'full_{sz}'. Default: 'none'
        normalize_kernel (bool): Default: False
        shared_filters (bool): Default: False
        filler (str): 'uniform' | 'linear'. Default: 'uniform'
    Note:
        - kernel_size only accepts odd numbers
        - padding should not be larger than :math:`dilation * (kernel_size - 1) / 2`
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 bias=True, kernel_type='gaussian', smooth_kernel_type='none', normalize_kernel=False,
                 shared_filters=False, filler='uniform', native_impl=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        output_padding = _pair(output_padding)
        dilation = _pair(dilation)
        super(PacConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, True, output_padding, bias,
            False, kernel_type, smooth_kernel_type, False, normalize_kernel, shared_filters, filler)

        self.native_impl = native_impl

    def compute_kernel(self, input_for_kernel, input_mask=None):
        return packernel2d(input_for_kernel, input_mask,
                           kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                           output_padding=self.output_padding, dilation=self.dilation, kernel_type=self.kernel_type,
                           smooth_kernel_type=self.smooth_kernel_type,
                           smooth_kernel=self.smooth_kernel if hasattr(self, 'smooth_kernel') else None,
                           inv_alpha=self.inv_alpha if hasattr(self, 'inv_alpha') else None,
                           inv_lambda=self.inv_lambda if hasattr(self, 'inv_lambda') else None,
                           channel_wise=False, normalize_kernel=self.normalize_kernel, transposed=True,
                           native_impl=self.native_impl)

    def forward(self, input_2d, input_for_kernel, kernel=None, mask=None):
        output_mask = None
        if kernel is None:
            kernel, output_mask = self.compute_kernel(input_for_kernel, mask)

        output = pacconv_transpose2d(input_2d, kernel, self.weight, self.bias, self.stride, self.padding,
                                     self.output_padding, self.dilation, self.shared_filters, self.native_impl)

        return output if output_mask is None else (output, output_mask)


class PacPool2d(_PacConvNd):
    r"""
    Args:
        kernel_size, stride, padding, dilation
        kernel_type (str): 'gaussian' | 'inv_{alpha}_{lambda}[_asym][_fixed]'. Default: 'gaussian'
        smooth_kernel_type (str): 'none' | 'gaussian' | 'average_{sz}' | 'full_{sz}'. Default: 'none'
        channel_wise (bool): Default: False
        normalize_kernel (bool): Default: False
        out_channels (int): needs to be specified for channel_wise 'inv_*' (non-fixed) kernels. Default: -1
    Note:
        - kernel_size only accepts odd numbers
        - padding should not be larger than :math:`dilation * (kernel_size - 1) / 2`
    """

    def __init__(self, kernel_size, stride=1, padding=0, dilation=1,
                 kernel_type='gaussian', smooth_kernel_type='none',
                 channel_wise=False, normalize_kernel=False, out_channels=-1, native_impl=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(PacPool2d, self).__init__(
            -1, out_channels, kernel_size, stride,
            padding, dilation, False, _pair(0), False,
            True, kernel_type, smooth_kernel_type, channel_wise, normalize_kernel, False, None)

        self.native_impl = native_impl

    def compute_kernel(self, input_for_kernel, input_mask=None):
        return packernel2d(input_for_kernel, input_mask,
                           kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                           dilation=self.dilation, kernel_type=self.kernel_type,
                           smooth_kernel_type=self.smooth_kernel_type,
                           smooth_kernel=self.smooth_kernel if hasattr(self, 'smooth_kernel') else None,
                           inv_alpha=self.inv_alpha if hasattr(self, 'inv_alpha') else None,
                           inv_lambda=self.inv_lambda if hasattr(self, 'inv_lambda') else None,
                           channel_wise=self.channel_wise, normalize_kernel=self.normalize_kernel, transposed=False,
                           native_impl=self.native_impl)

    def forward(self, input_2d, input_for_kernel, kernel=None, mask=None):
        output_mask = None
        if kernel is None:
            kernel, output_mask = self.compute_kernel(input_for_kernel, mask)

        bs, in_ch, in_h, in_w = input_2d.shape
        if self.channel_wise and (kernel.shape[1] != in_ch):
            raise ValueError('input and kernel must have the same number of channels when channel_wise=True')
        assert self.out_channels <= 0 or self.out_channels == in_ch

        output = pacpool2d(input_2d, kernel, self.kernel_size, self.stride, self.padding, self.dilation,
                           self.native_impl)

        return output if output_mask is None else (output, output_mask)

class MultiHeadAttention(nn.Module):
    """
    This class implements a multi head attention module like proposed in:
    https://arxiv.org/abs/2005.12872
    """

    def __init__(self, query_dimension: int = 64, hidden_features: int = 64, number_of_heads: int = 16,
                 dropout: float = 0.0) -> None:
        """
        Constructor method
        :param query_dimension: (int) Dimension of query tensor
        :param hidden_features: (int) Number of hidden features in detr
        :param number_of_heads: (int) Number of prediction heads
        :param dropout: (float) Dropout factor to be utilized
        """
        # Call super constructor
        super(MultiHeadAttention, self).__init__()
        # Save parameters
        self.hidden_features = hidden_features
        self.number_of_heads = number_of_heads
        self.dropout = dropout
        # Init layer
        self.layer_box_embedding = nn.Linear(in_features=query_dimension, out_features=hidden_features, bias=True)
        # Init convolution layer
        self.layer_image_encoding = nn.Conv2d(in_channels=query_dimension, out_channels=hidden_features,
                                              kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        # Init normalization factor
        self.normalization_factor = torch.tensor(self.hidden_features / self.number_of_heads, dtype=torch.float).sqrt()

    def forward(self, input_box_embeddings: torch.Tensor, input_image_encoding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input_box_embeddings: (torch.Tensor) Bounding box embeddings
        :param input_image_encoding: (torch.Tensor) Encoded image of the transformer encoder
        :return: (torch.Tensor) Attention maps of shape (batch size, n, m, height, width)
        """
        # Map box embeddings
        output_box_embeddings = self.layer_box_embedding(input_box_embeddings)
        # Map image features
        output_image_encoding = self.layer_image_encoding(input_image_encoding)
        # Reshape output box embeddings
        output_box_embeddings = output_box_embeddings.view(output_box_embeddings.shape[0],
                                                           output_box_embeddings.shape[1],
                                                           self.number_of_heads,
                                                           self.hidden_features // self.number_of_heads)
        # Reshape output image encoding
        output_image_encoding = output_image_encoding.view(output_image_encoding.shape[0],
                                                           self.number_of_heads,
                                                           self.hidden_features // self.number_of_heads,
                                                           output_image_encoding.shape[-2],
                                                           output_image_encoding.shape[-1])
        # Combine tensors and normalize
        output = torch.einsum("bqnc,bnchw->bqnhw",
                              output_box_embeddings * self.normalization_factor,
                              output_image_encoding)
        # Apply softmax
        output = F.softmax(output.flatten(start_dim=2), dim=-1).view_as(output)
        # Perform dropout if utilized
        if self.dropout > 0.0:
            output = F.dropout(input=output, p=self.dropout, training=self.training)
        return output.contiguous()


class ResFeaturePyramidBlock(nn.Module):
    """
    This class implements a residual feature pyramid block.
    """

    def __init__(self, in_channels: int, out_channels: int, feature_channels: int, convolution: Type = nn.Conv2d,
                 normalization: Type = nn.InstanceNorm2d, activation: Type = nn.PReLU, dropout: float = 0.0) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param feature_channels: (int) Number of channels present in the feature map
        :param convolution: (Type) Type of convolution to be utilized
        :param normalization: (Type) Type of normalization to be used
        :param activation: (Type) Type of activation function to be utilized
        :param dropout: (float) Dropout factor to be applied after upsampling is performed
        """
        # Call super constructor
        super(ResFeaturePyramidBlock, self).__init__()
        # Save parameter
        self.dropout = dropout
        # Init main mapping
        self.main_mapping = nn.Sequential(
            convolution(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            normalization(num_features=out_channels // 2, affine=True, track_running_stats=True),
            activation(),
            convolution(in_channels=out_channels // 2, out_channels=out_channels // 2, kernel_size=(3, 3),
                        stride=(1, 1), padding=(1, 1), bias=True),
            normalization(num_features=out_channels // 2, affine=True, track_running_stats=True),
            activation()
        )
        # Init residual mapping
        self.residual_mapping = convolution(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=(1, 1),
                                            stride=(1, 1), padding=(0, 0),
                                            bias=True) if in_channels != out_channels // 2 else nn.Identity()
        # Init upsampling
        self.upsampling = nn.Upsample(scale_factor=(2, 2), mode='bicubic', align_corners=False)
        # Init feature mapping
        self.feature_mapping = convolution(in_channels=feature_channels, out_channels=out_channels // 2,
                                           kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

    def forward(self, input: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of shape (batch size * number of heads, in channels, height, width)
        :param feature: (torch.Tensor) Feature tensor of backbone of shape (batch size, channels, height, width)
        :return: (torch.Tensor) Output tensor (batch size * number of heads, out channels, height * 2, width * 2)
        """
        # Perform main mapping
        output = self.main_mapping(input)
        # Perform residual mapping
        output = output + self.residual_mapping(input)
        # Perform upsampling
        output = self.upsampling(output)
        # Perform dropout if utilized
        if self.dropout > 0.0:
            output = F.dropout(output, p=self.dropout, training=self.training)
        # Add mapped feature
        output = torch.cat((output, self.feature_mapping(feature).unsqueeze(dim=1).repeat(1, int(
            output.shape[0] / feature.shape[0]), 1, 1, 1).flatten(0, 1).contiguous()), dim=1)
        return output


class ResPACFeaturePyramidBlock(nn.Module):
    """
    This class implements a residual feature pyramid block.
    """

    def __init__(self, in_channels: int, out_channels: int, feature_channels: int, convolution: Type = nn.Conv2d,
                 normalization: Type = nn.InstanceNorm2d, activation: Type = nn.PReLU, dropout: float = 0.0) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param feature_channels: (int) Number of channels present in the feature map
        :param convolution: (Type) Type of convolution to be utilized
        :param normalization: (Type) Type of normalization to be used
        :param activation: (Type) Type of activation function to be utilized
        :param dropout: (float) Dropout factor to be applied after upsampling is performed
        """
        # Call super constructor
        super(ResPACFeaturePyramidBlock, self).__init__()
        # Save parameter
        self.dropout = dropout
        # Init main mapping
        self.main_mapping = nn.Sequential(
            convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            normalization(num_features=out_channels, affine=True, track_running_stats=True),
            activation(),
            convolution(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            normalization(num_features=out_channels, affine=True, track_running_stats=True),
            activation()
        )
        # Init residual mapping
        self.residual_mapping = convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                            stride=(1, 1), padding=(0, 0),
                                            bias=True) if in_channels != out_channels else nn.Identity()
        # Init upsampling
        self.upsampling = nn.Upsample(scale_factor=(2, 2), mode='bicubic', align_corners=False)
        # Init feature mapping
        self.feature_mapping = convolution(in_channels=feature_channels, out_channels=out_channels,
                                           kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        # Init pixel adaptive convolution
        self.pixel_adaptive_convolution = PacConv2d(in_channels=out_channels, out_channels=out_channels,
                                                    kernel_size=(5, 5), padding=(2, 2), stride=(1, 1), bias=True,
                                                    normalize_kernel=True)

    def forward(self, input: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of shape (batch size * number of heads, in channels, height, width)
        :param feature: (torch.Tensor) Feature tensor of backbone of shape (batch size, channels, height, width)
        :return: (torch.Tensor) Output tensor (batch size * number of heads, out channels, height * 2, width * 2)
        """
        # Perform main mapping
        output = self.main_mapping(input)
        # Perform residual mapping
        output = output + self.residual_mapping(input)
        # Perform upsampling
        output = self.upsampling(output)
        # Perform dropout if utilized
        if self.dropout > 0.0:
            output = F.dropout(output, p=self.dropout, training=self.training)
        # Perform PAC
        output = self.pixel_adaptive_convolution(output, self.feature_mapping(feature).unsqueeze(dim=1).repeat(1, int(
            output.shape[0] / feature.shape[0]), 1, 1, 1).flatten(0, 1).contiguous())
        return output


class FinalBlock(nn.Module):
    """
    This class implements the final block of the segmentation head
    """

    def __init__(self, in_channels: int, out_channels: int, convolution: Type = nn.Conv2d,
                 normalization: Type = nn.InstanceNorm2d, activation: Type = nn.PReLU,
                 number_of_query_positions: int = None) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param convolution: (Type) Type of convolution to be utilized
        :param normalization: (Type) Type of normalization to be utilized
        :param activation: (Type) Type of activation function to be utilized
        :param number_of_query_positions: (int) Number of query positions utilized
        """
        # Call super constructor
        super(FinalBlock, self).__init__()
        # Init main mapping
        self.main_mapping = nn.Sequential(
            convolution(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            normalization(num_features=in_channels, affine=True, track_running_stats=True),
            activation(),
            convolution(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            normalization(num_features=in_channels, affine=True, track_running_stats=True),
            activation()
        )
        # Init upsampling
        self.upsampling = nn.Upsample(scale_factor=(2, 2), mode='bicubic', align_corners=False)
        # Init final mapping
        self.final_mapping = nn.Sequential(
            convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            activation(),
            convolution(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1),
                        padding=(0, 0), bias=True)
        )

    def forward(self, input: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of shape (batch size * number of heads, in channels, height, width)
        :param batch_size: (torch.Tensor) Batch size
        :return: (torch.Tensor) Output tensor (batch size * number of heads, out channels, height * 2, width * 2)
        """
        # Perform main mapping
        output = self.main_mapping(input)
        # Perform residual mapping
        output = output + input
        # Perform upsampling
        output = self.upsampling(output)
        # Perform final mapping
        output = self.final_mapping(output)
        return output.view(batch_size, -1, output.shape[2], output.shape[3])


class FinalBlockReshaped(nn.Module):
    """
    This class implements the final block of the segmentation head
    """

    def __init__(self, in_channels: int, out_channels: int, convolution: Type = nn.Conv2d,
                 normalization: Type = nn.InstanceNorm2d, activation: Type = nn.PReLU,
                 number_of_query_positions: int = 12) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param convolution: (Type) Type of convolution to be utilized
        :param normalization: (Type) Type of normalization to be utilized
        :param activation: (Type) Type of activation function to be utilized
        :param number_of_query_positions: (int) Number of query positions utilized
        """
        # Call super constructor
        super(FinalBlockReshaped, self).__init__()
        # Init main mapping
        self.main_mapping = nn.Sequential(
            convolution(in_channels=in_channels * number_of_query_positions,
                        out_channels=in_channels * number_of_query_positions // 2,
                        kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            normalization(num_features=in_channels * number_of_query_positions // 2, affine=True,
                          track_running_stats=True),
            activation(),
            convolution(in_channels=in_channels * number_of_query_positions // 2,
                        out_channels=in_channels * number_of_query_positions // 8, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            normalization(num_features=in_channels * number_of_query_positions // 8, affine=True,
                          track_running_stats=True),
            activation()
        )
        # Init residual mapping
        self.residual_mapping = convolution(in_channels=in_channels * number_of_query_positions,
                                            out_channels=in_channels * number_of_query_positions // 8,
                                            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        # Init upsampling
        self.upsampling = nn.Upsample(scale_factor=(2, 2), mode='bicubic', align_corners=False)
        # Init final mapping
        self.final_mapping = nn.Sequential(
            convolution(in_channels=in_channels * number_of_query_positions // 8,
                        out_channels=in_channels, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            activation(),
            convolution(in_channels=in_channels,
                        out_channels=out_channels * number_of_query_positions, kernel_size=(1, 1), stride=(1, 1),
                        padding=(0, 0), bias=True)
        )

    def forward(self, input: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of shape (batch size * number of heads, in channels, height, width)
        :param batch_size: (torch.Tensor) Batch size
        :return: (torch.Tensor) Output tensor (batch size * number of heads, out channels, height * 2, width * 2)
        """
        # Reshape input
        input = input.view(batch_size, -1, input.shape[2], input.shape[3])
        # Perform main mapping
        output = self.main_mapping(input)
        # Perform residual mapping
        output = output + self.residual_mapping(input)
        # Perform upsampling
        output = self.upsampling(output)
        # Perform final mapping
        output = self.final_mapping(output)
        return output


class SegmentationHead(nn.Module):
    """
    This class implements a feature pyramid decoder network for prediction each binary instance mask.
    """

    def __init__(self, channels: Tuple[Tuple[int, int], ...] = ((80, 32), (32, 16), (16, 8), (8, 4)),
                 feature_channels: Tuple[int, ...] = (128, 64, 32, 16), convolution: Type = nn.Conv2d,
                 normalization: Type = nn.InstanceNorm2d, activation: Type = nn.PReLU,
                 block: Type = ResPACFeaturePyramidBlock, dropout: float = 0.0,
                 number_of_query_positions: int = 12, softmax: bool = True) -> None:
        """
        Constructor method
        :param channels: (Tuple[Tuple[int, int], ...]) Tuple of input and output channels of each block
        :param feature_channels: (Tuple[int, ...]) Tuple of channels present in each feature map
        :param convolution: (Type) Convolution to be utilized
        :param normalization: (Type) Normalization to be utilized
        :param activation: (Type) Activation function to be utilized
        :param block: (Type) Type of main convolution block to be utilized
        :param dropout: (float) Dropout factor to be applied
        :param number_of_query_positions: (int) Number of query positions utilized
        :param softmax: (bool) If softmax is utilized true and than final block with reshaping is utilized
        """
        # Call super constructor
        super(SegmentationHead, self).__init__()
        # Init blocks
        self.blocks = nn.ModuleList()
        for channel, feature_channel in zip(channels, feature_channels):
            self.blocks.append(block(in_channels=channel[0], out_channels=channel[1],
                                     feature_channels=feature_channel, convolution=convolution,
                                     normalization=normalization, activation=activation, dropout=dropout))
        # Init final block
        if softmax:
            self.final_block = FinalBlockReshaped(in_channels=channels[-1][-1], out_channels=1, convolution=convolution,
                                                  normalization=normalization, activation=activation,
                                                  number_of_query_positions=number_of_query_positions)
        else:
            self.final_block = FinalBlock(in_channels=channels[-1][-1], out_channels=1, convolution=convolution,
                                          normalization=normalization, activation=activation,
                                          number_of_query_positions=number_of_query_positions)

    def forward(self, features: torch.Tensor, segmentation_attention_head: torch.Tensor,
                backbone_features: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        :param features: (torch.Tensor) Input features of the transformer module
        :param segmentation_attention_head: (torch.Tensor) Output tensor of the multi head attention module
        :param backbone_features: (torch.Tensor) List of backbone feature maps
        :return: (torch.Tensor) Instance segmentation prediction maps
        """
        # Construct input to convolutions
        input = torch.cat(
            [features.unsqueeze(dim=1).repeat(1, segmentation_attention_head.shape[1], 1, 1, 1).flatten(0, 1),
             segmentation_attention_head.flatten(0, 1)], dim=1).contiguous()
        # Forward pass of all blocks
        for block, feature in zip(self.blocks, backbone_features):
            input = block(input, feature)
        # Forward pass of final block
        output = self.final_block(input, features.shape[0])
        return output

class CellDETR(nn.Module):
    """
    This class implements a DETR (Facebook AI) like instance segmentation model.
    """

    def __init__(self,
                 num_classes: int = 2,
                 number_of_query_positions: int = 26,
                 hidden_features=128,
                # hidden_features=32,
                 backbone_channels: Tuple[Tuple[int, int], ...] = (
                          (1, 64), (64, 128), (128, 256), (256, 256)),
                # backbone_channels: Tuple[Tuple[int, int], ...] = (
                 #        (1, 16), (16, 32), (32, 64), (64, 64)),
                 backbone_block: Type = ResNetBlock, backbone_convolution: Type = nn.Conv2d,
                 backbone_normalization: Type = nn.InstanceNorm2d, backbone_activation: Type = nn.LeakyReLU ,
                 backbone_pooling: Type = nn.AvgPool2d,
                 bounding_box_head_features: Tuple[Tuple[int, int], ...] = ((128, 64), (64, 16), (16, 4)),
                # bounding_box_head_features: Tuple[Tuple[int, int], ...] = ((32, 16), (16, 4)),
                 bounding_box_head_activation: Type = nn.LeakyReLU ,
                 classification_head_activation: Type = nn.LeakyReLU ,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 2,
                 dropout: float = 0.0,
                 transformer_attention_heads: int = 8,
                # transformer_attention_heads: int = 4,
                 transformer_activation: Type = nn.LeakyReLU ,
                 segmentation_attention_heads: int = 8,
                 #segmentation_attention_heads: int = 4,
                 segmentation_head_channels: Tuple[Tuple[int, int], ...] = (
                          (128 + 8, 128), (128, 64), (64, 32)),
                # segmentation_head_channels: Tuple[Tuple[int, int], ...] = (
                 #        (32 + 4, 32),(32, 32) , (32, 16)),
                 segmentation_head_feature_channels: Tuple[int, ...] = (256, 128, 64),
                # segmentation_head_feature_channels: Tuple[int, ...] = (64, 32, 16),
                 segmentation_head_block: Type = ResFeaturePyramidBlock,
                 segmentation_head_convolution: Type = nn.Conv2d,
                 segmentation_head_normalization: Type = nn.InstanceNorm2d,
                 segmentation_head_activation: Type = nn.LeakyReLU,
                 segmentation_head_final_activation: Type = nn.Sigmoid) -> None:
        """
        Constructor method
        :param num_classes: (int) Number of classes in the dataset
        :param number_of_query_positions: (int) Number of query positions
        :param hidden_features: (int) Number of hidden features in the transformer module
        :param backbone_channels: (Tuple[Tuple[int, int], ...]) In and output channels of each block in the backbone
        :param backbone_block: (Type) Type of block to be utilized in backbone
        :param backbone_convolution: (Type) Type of convolution to be utilized in the backbone
        :param backbone_normalization: (Type) Type of normalization to be used in the backbone
        :param backbone_activation: (Type) Type of activation function used in the backbone
        :param backbone_pooling: (Type) Type of pooling operation utilized in the backbone
        :param bounding_box_head_features: (Tuple[Tuple[int, int], ...]) In and output features of each layer in BB head
        :param bounding_box_head_activation: (Type) Type of activation function utilized in BB head
        :param classification_head_activation: (Type) Type of activation function utilized in classification head
        :param num_encoder_layers: (int) Number of layers in encoder part of the transformer module
        :param num_decoder_layers: (int) Number of layers in decoder part of the transformer module
        :param dropout: (float) Dropout factor used in transformer module and segmentation head
        :param transformer_attention_heads: (int) Number of attention heads in the transformer module
        :param transformer_activation: (Type) Type of activation function to be utilized in the transformer module
        :param segmentation_attention_heads: (int) Number of attention heads in the 2d multi head attention module
        :param segmentation_head_channels: (Tuple[Tuple[int, int], ...]) Number of in and output channels in seg. head
        :param segmentation_head_feature_channels: (Tuple[int, ...]) Backbone feature channels used in seg. head
        :param segmentation_head_block: (Type) Type of block to be utilized in segmentation head
        :param segmentation_head_convolution: (Type) Type of convolution utilized in segmentation head
        :param segmentation_head_normalization: (Type) Type of normalization used in segmentation head
        :param segmentation_head_activation: (Type) Type of activation used in segmentation head
        :param segmentation_head_final_activation: (Type) Type of activation function to be applied to the output pred
        """
        # Call super constructor
        super(CellDETR, self).__init__()
        # Init backbone
        self.backbone = Backbone(channels=backbone_channels, block=backbone_block, convolution=backbone_convolution,
                                 normalization=backbone_normalization, activation=backbone_activation,
                                 pooling=backbone_pooling)
        # Init convolution mapping to match transformer dims
        self.convolution_mapping = nn.Conv2d(in_channels=backbone_channels[-1][-1], out_channels=hidden_features,
                                             kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        # Init query positions
        self.query_positions = nn.Parameter(
            data=torch.randn(number_of_query_positions, hidden_features, dtype=torch.float),
            requires_grad=True)
        # Init embeddings
        self.row_embedding = nn.Parameter(data=torch.randn(50, hidden_features // 2, dtype=torch.float),
                                          requires_grad=True)
        self.column_embedding = nn.Parameter(data=torch.randn(50, hidden_features // 2, dtype=torch.float),
                                             requires_grad=True)
        # Init transformer
        self.transformer = Transformer(d_model=hidden_features, nhead=transformer_attention_heads,
                                       num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                       dropout=dropout, dim_feedforward=4 * hidden_features,
                                       activation=transformer_activation)
        # Init bounding box head
        self.bounding_box_head = BoundingBoxHead(features=bounding_box_head_features,
                                                 activation=bounding_box_head_activation)
        # Init class head
        self.class_head = nn.Sequential(
            nn.Linear(in_features=hidden_features, out_features=hidden_features // 2, bias=True),
            classification_head_activation(),
            nn.Linear(in_features=hidden_features // 2, out_features=num_classes + 1, bias=True))
        # Init segmentation attention head
        self.segmentation_attention_head = MultiHeadAttention(query_dimension=hidden_features,
                                                              hidden_features=hidden_features,
                                                              number_of_heads=segmentation_attention_heads,
                                                              dropout=dropout)
        # Init segmentation head
        self.segmentation_head = SegmentationHead(channels=segmentation_head_channels,
                                                  feature_channels=segmentation_head_feature_channels,
                                                  convolution=segmentation_head_convolution,
                                                  normalization=segmentation_head_normalization,
                                                  activation=segmentation_head_activation,
                                                  block=segmentation_head_block,
                                                  number_of_query_positions=number_of_query_positions,
                                                  softmax=isinstance(segmentation_head_final_activation(), nn.Softmax))
        # Init final segmentation activation
        self.segmentation_final_activation = segmentation_head_final_activation(dim=1) if isinstance(
            segmentation_head_final_activation(), nn.Softmax) else segmentation_head_final_activation()

    def get_parameters(self, lr_main: float = 1e-04, lr_backbone: float = 1e-05) -> Iterable:
        """
        Method returns all parameters of the model with different learning rates
        :param lr_main: (float) Leaning rate of all parameters which are not included in the backbone
        :param lr_backbone: (float) Leaning rate of the backbone parameters
        :return: (Iterable) Iterable object including the main parameters of the generator network
        """
        return [{'params': self.backbone.parameters(), 'lr': lr_backbone},
                {'params': self.convolution_mapping.parameters(), 'lr': lr_main},
                {'params': [self.row_embedding], 'lr': lr_main},
                {'params': [self.column_embedding], 'lr': lr_main},
                {'params': self.transformer.parameters(), 'lr': lr_main},
                {'params': self.bounding_box_head.parameters(), 'lr': lr_main},
                {'params': self.class_head.parameters(), 'lr': lr_main},
                {'params': self.segmentation_attention_head.parameters(), 'lr': lr_main},
                {'params': self.segmentation_head.parameters(), 'lr': lr_main}]

    def get_segmentation_head_parameters(self, lr: float = 1e-05) -> Iterable:
        """
        Method returns all parameter of the segmentation head and the 2d multi head attention module
        :param lr: (float) Learning rate to be utilized
        :return: (Iterable) Iterable object including the parameters of the segmentation head
        """
        return [{'params': self.segmentation_attention_head.parameters(), 'lr': lr},
                {'params': self.segmentation_head.parameters(), 'lr': lr}]

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        :param input: (torch.Tensor) Input image of shape (batch size, channels, height, width)
        :return: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) Class prediction, bounding box predictions and
        segmentation maps
        """
        # Get features from backbone
        features, feature_list = self.backbone(input)
        # Map features to the desired shape
        features = self.convolution_mapping(features)
        # Get height and width of features
        height, width = features.shape[2:]
        # Get batch size
        batch_size = features.shape[0]
        # Make positional embeddings
        positional_embeddings = torch.cat([self.column_embedding[:height].unsqueeze(dim=0).repeat(height, 1, 1),
                                           self.row_embedding[:width].unsqueeze(dim=1).repeat(1, width, 1)],
                                          dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        latent_tensor, features_encoded = self.transformer(features, None, self.query_positions, positional_embeddings)
        latent_tensor = latent_tensor.permute(2, 0, 1)
        # Get class prediction
        class_prediction = F.softmax(self.class_head(latent_tensor), dim=2).clone()
        # Get bounding boxes
        bounding_box_prediction = self.bounding_box_head(latent_tensor)
        # Get bounding box attention masks for segmentation
        bounding_box_attention_masks = self.segmentation_attention_head(
            latent_tensor, features_encoded.contiguous())
        # Get instance segmentation prediction
        instance_segmentation_prediction = self.segmentation_head(features.contiguous(),
                                                                  bounding_box_attention_masks.contiguous(),
                                                                  feature_list[-2::-1])
        return class_prediction, \
               bounding_box_prediction.sigmoid().clone(), \
               self.segmentation_final_activation(instance_segmentation_prediction).clone()

class IoU(nn.Module):
    """
    This class implements the IoU for validation. Not gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(IoU, self).__init__()
        # Save parameter
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the IoU score
        :param prediction: (torch.Tensor) Prediction of all shapes
        :param label: (torch.Tensor) Label of all shapes
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) IoU score
        """
        # Apply threshold to prediction
        prediction = (prediction > self.threshold).float()
        # Compute intersection
        intersection = ((prediction + label) == 2.0).sum()
        # Compute union
        union = ((prediction + label) >= 1.0).sum()
        # Compute iou
        return intersection / (union + 1e-10)


class CellIoU(nn.Module):
    """
    This class implements the IoU metric for cell instances.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(CellIoU, self).__init__()
        # Save parameter
        self.threshold = threshold

    def forward(self, prediction: torch.Tensor, label: torch.Tensor, class_label: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        Forward pass
        :param prediction: (torch.Tensor) Instance segmentation prediction
        :param label: (torch.Tensor) Instance segmentation label
        :param class_label: (torch.Tensor) Class label of each instance segmentation map
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) Mean cell IoU metric
        """
        # Apply threshold to prediction
        prediction = (prediction > self.threshold).float()
        # Get segmentation maps belonging to the cell class
        indexes = np.argwhere(class_label.cpu().numpy() >= 2)[:, 0]
        # Case if no cells are present
        if indexes.shape == (0,):
            return torch.tensor(np.nan)
        prediction = prediction[indexes].sum(dim=0)
        label = label[indexes].sum(dim=0)
        # Compute intersection
        intersection = ((prediction + label) == 2.0).sum(dim=(-2, -1))
        # Compute union
        union = ((prediction + label) >= 1.0).sum(dim=(-2, -1))
        # Compute iou
        return intersection / (union + 1e-10)


class MIoU(nn.Module):
    """
    This class implements the mean IoU for validation. Not gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(MIoU, self).__init__()
        # Save parameter
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the IoU score
        :param prediction: (torch.Tensor) Prediction of shape [..., height, width]
        :param label: (torch.Tensor) Label of shape [..., height, width]
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) IoU score
        """
        # Apply threshold to prediction
        prediction = (prediction > self.threshold).float()
        # Compute intersection
        intersection = ((prediction + label) == 2.0).sum(dim=(-2, -1))
        # Compute union
        union = ((prediction + label) >= 1.0).sum(dim=(-2, -1))
        # Compute iou
        return (intersection / (union + 1e-10)).mean()


class Dice(nn.Module):
    """
    This class implements the dice score for validation. No gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(Dice, self).__init__()
        # Save parameter
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the dice coefficient
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) Dice coefficient
        """
        # Apply threshold to prediction
        prediction = (prediction > self.threshold).float()
        # Compute intersection
        intersection = ((prediction + label) == 2.0).sum()
        # Compute dice score
        return (2 * intersection) / (prediction.sum() + label.sum() + 1e-10)


class ClassificationAccuracy(nn.Module):
    """
    This class implements the classification accuracy computation. No gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(ClassificationAccuracy, self).__init__()
        # Save parameter
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the accuracy score
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Accuracy
        """
        # Calc correct classified elements
        correct_classified_elements = (prediction == label).float().sum()
        # Calc accuracy
        return correct_classified_elements / prediction.numel()


class InstancesAccuracy(nn.Module):
    """
    This class implements the accuracy computation. No gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(InstancesAccuracy, self).__init__()
        # Save parameter
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the accuracy score
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) Accuracy
        """
        # Apply threshold to prediction
        prediction = (prediction > self.threshold).float()
        # Calc correct classified elements
        correct_classified_elements = (prediction == label).float().sum()
        # Calc accuracy
        return correct_classified_elements / prediction.numel()


class Accuracy(nn.Module):
    """
    This class implements the accuracy computation. No gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(Accuracy, self).__init__()
        # Save parameter
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the accuracy score
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) Accuracy
        """
        # Apply threshold to prediction
        prediction = (prediction > self.threshold).float()
        # Get instance map
        prediction = (prediction
                      * torch.arange(1, prediction.shape[0] + 1, device=prediction.device).view(-1, 1, 1)).sum(dim=0)
        label = (label * torch.arange(1, label.shape[0] + 1, device=label.device).view(-1, 1, 1)).sum(dim=0)
        # Calc correct classified elements
        correct_classified_elements = (prediction == label).float().sum()
        # Calc accuracy
        return correct_classified_elements / prediction.numel()


class Recall(nn.Module):
    """
    This class implements the recall score. No gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(Recall, self).__init__()
        # Save parameter
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the recall score
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) Recall score
        """
        # Apply threshold to prediction
        prediction = (prediction > self.threshold).float()
        # Calc true positive elements
        true_positive_elements = (((prediction == 1.0).float() + (label == 1.0)) == 2.0).float()
        # Calc false negative elements
        false_negative_elements = (((prediction == 0.0).float() + (label == 1.0)) == 2.0).float()
        # Calc recall scale
        return true_positive_elements.sum() / ((true_positive_elements + false_negative_elements).sum() + 1e-10)


class Precision(nn.Module):
    """
    This class implements the precision score. No gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(Precision, self).__init__()
        # Save parameter
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the precision score
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) Precision score
        """
        # Apply threshold to prediction
        prediction = (prediction > self.threshold).float()
        # Calc true positive elements
        true_positive_elements = (((prediction == 1.0).float() + (label == 1.0)) == 2.0).float()
        # Calc false positive elements
        false_positive_elements = (((prediction == 1.0).float() + (label == 0.0)) == 2.0).float()
        # Calc precision
        return true_positive_elements.sum() / ((true_positive_elements + false_positive_elements).sum() + 1e-10)


class F1(nn.Module):
    """
    This class implements the F1 score. No gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(F1, self).__init__()
        # Init recall and precision module
        self.recall = Recall(threshold=threshold)
        self.precision = Precision(threshold=threshold)

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the F1 score
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) F1 score
        """
        # Calc recall
        recall = self.recall(prediction, label)
        # Calc precision
        precision = self.precision(prediction, label)
        # Calc F1 score
        return (2.0 * recall * precision) / (recall + precision + 1e-10)


class BoundingBoxIoU(nn.Module):
    """
    This class implements the bounding box IoU.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(BoundingBoxIoU, self).__init__()

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the bounding box iou score
        :param prediction: (torch.Tensor) Bounding box predictions [batch size, instances, 4 (x0, y0, x1, y1)]
        :param label: (torch.Tensor) Bounding box labels in the format [batch size, instances, 4 (x0, y0, x1, y1)]
        :return: (torch.Tensor) Bounding box iou
        """
        return giou(bounding_box_1=prediction, bounding_box_2=label, return_iou=True)[1].diagonal().mean()


class BoundingBoxGIoU(nn.Module):
    """
    This class implements the bounding box IoU.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(BoundingBoxGIoU, self).__init__()

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the bounding box iou score
        :param prediction: (torch.Tensor) Bounding box predictions [batch size, instances, 4 (x0, y0, x1, y1)]
        :param label: (torch.Tensor) Bounding box labels in the format [batch size, instances, 4 (x0, y0, x1, y1)]
        :return: (torch.Tensor) Bounding box iou
        """
        return giou(bounding_box_1=prediction, bounding_box_2=label).diagonal().mean()


class MeanAveragePrecision(nn.Module):
    """
    This class implements the mean average precision metric for instance segmentation.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(MeanAveragePrecision, self).__init__()

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the accuracy score
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) Accuracy
        """
        # Flatten tensors and convert to numpy
        prediction_flatten = prediction.detach().cpu().view(-1).numpy()
        label_flatten = label.detach().cpu().view(-1).numpy()
        # Calc accuracy
        return torch.tensor(average_precision_score(label_flatten, prediction_flatten, average="macro"),
                            dtype=torch.float, device=label.device)

class ModelWrapper(object):
    """
    This class implements a wrapper for Cell.DETR, optimizer, datasets, and loss functions. This class implements also
    the training, validation and test method.
    """

    def __init__(self,
                 detr: Union[nn.DataParallel, CellDETR],
                 detr_optimizer: torch.optim.Optimizer,
                 detr_segmentation_optimizer: torch.optim.Optimizer,
                 training_dataset: DataLoader,
                 validation_dataset: DataLoader,
                 test_dataset: DataLoader,
                 loss_function: nn.Module,
                 learning_rate_schedule: torch.optim.lr_scheduler.MultiStepLR = None,
                 device: str = "cuda",
                 save_data_path: str = "saved_data",
                 use_telegram: bool = True) -> None:
        """
        Constructor method
        :param detr: (Union[nn.DataParallel, DETR]) DETR model
        :param detr_optimizer: (torch.optim.Optimizer) DETR model optimizer
        :param detr_segmentation_optimizer: (torch.optim.Optimizer) DETR segmentation head optimizer
        :param training_dataset: (DataLoader) Training dataset
        :param validation_dataset: (DataLoader) Validation dataset
        :param test_dataset: (DataLoader) Test dataset
        :param loss_function: (nn.Module) Loss function
        :param learning_rate_schedule: (torch.optim.lr_scheduler.MultiStepLR) Learning rate schedule
        :param device: (str) Device to be utilized
        :param save_data_path: (str) Path to store log data
        :param use_telegram: (bool) If true telegram_send is used
        """
        # Save parameters
        self.detr = detr
        self.detr_optimizer = detr_optimizer
        self.detr_segmentation_optimizer = detr_segmentation_optimizer
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.loss_function = loss_function
        self.learning_rate_schedule = learning_rate_schedule
        self.device = device
        self.save_data_path = save_data_path
        self.use_telegram = use_telegram
        # Init logger
        self.logger = Logger()
        # Make directories to save logs, plots and models during training
        time_and_date = str(datetime.now())
        save_data_path = os.path.join(save_data_path, time_and_date)
        os.makedirs(save_data_path, exist_ok=True)
        self.path_save_models = os.path.join(save_data_path, "models")
        os.makedirs(self.path_save_models, exist_ok=True)
        self.path_save_plots = os.path.join(save_data_path, "plots")
        os.makedirs(self.path_save_plots, exist_ok=True)
        self.path_save_metrics = os.path.join(save_data_path, "metrics")
        os.makedirs(self.path_save_metrics, exist_ok=True)
        # Init variable to store best mIoU
        self.best_miou = 0.0

    def train(self, epochs: int = 40, validate_after_n_epochs: int = 5, save_model_after_n_epochs: int = 20,
              optimize_only_segmentation_head_after_epoch: int = 150) -> None:
        """
        Training method
        :param epochs: (int) Number of epochs to perform
        :param validate_after_n_epochs: (int) Number of epochs after the validation is performed
        :param save_model_after_n_epochs: (int) Number epochs after the current models is saved
        :param optimize_only_segmentation_head_after_epoch: (int) Number of epochs after only the seg. head is trained
        """
        # Model into train mode
        self.detr.train()
        # Model to device
        self.detr.to(self.device)
        # Init progress bar
        self.progress_bar = tqdm(total=epochs * len(self.training_dataset.dataset))
        # Main trainings loop
        for epoch in range(epochs):
            for input, instance_labels, bounding_box_labels, class_labels in self.training_dataset:
                # Update progress bar
                self.progress_bar.update(n=input.shape[0])
                # Data to device
                input = input.to(self.device)
                instance_labels = iterable_to_device(instance_labels, device=self.device)
                bounding_box_labels = iterable_to_device(bounding_box_labels, device=self.device)
                class_labels = iterable_to_device(class_labels, device=self.device)
                # Reset gradients
                self.detr.zero_grad()
                # Get prediction
                
                class_predictions, bounding_box_predictions, instance_predictions = self.detr(input)
                # Calc loss
                
                loss_classification, loss_bounding_box, loss_segmentation = self.loss_function(class_predictions,
                                                                                               bounding_box_predictions,
                                                                                               instance_predictions,
                                                                                               class_labels,
                                                                                               bounding_box_labels,
                                                                                               instance_labels)
                
                # Case if the whole network is optimized
                if epoch < optimize_only_segmentation_head_after_epoch:
                    # Perform backward pass to compute the gradients
                    (loss_classification + loss_bounding_box + loss_segmentation).backward()
                    # Optimize detr
                    self.detr_optimizer.step()
                else:
                    # Perform backward pass to compute the gradients
                    loss_segmentation.backward()
                    # Optimize detr
                    self.detr_segmentation_optimizer.step()
                
                # Show losses in progress bar
                self.progress_bar.set_description(
                    "Epoch {}/{} Best val. mIoU={:.4f} Loss C.={:.4f} Loss BB.={:.4f} Loss Seg.={:.4f}".format(
                        epoch + 1, epochs, self.best_miou, loss_classification.item(), loss_bounding_box.item(),
                        loss_segmentation.item()))
                # Log losses
                self.logger.log(metric_name="loss_classification", value=loss_classification.item())
                self.logger.log(metric_name="loss_bounding_box", value=loss_bounding_box.item())
                self.logger.log(metric_name="loss_segmentation", value=loss_segmentation.item())
            # Learning rate schedule step
            if self.learning_rate_schedule is not None:
                self.learning_rate_schedule.step()
            # Validate
            if (epoch + 1) % validate_after_n_epochs == 0:
                self.validate(epoch=epoch, train=True)
            # Save model
            if (epoch + 1) % save_model_after_n_epochs == 0:
                torch.save(
                    self.detr.module.state_dict() if isinstance(self.detr, nn.DataParallel) else self.detr.state_dict(),
                    os.path.join(self.path_save_models, "detr_{}.pt".format(epoch)))
    

        self.validate(epoch=epoch, number_of_plots=30)
        # Close progress bar
        self.progress_bar.close()
        # Load best model
        self.detr.state_dict(torch.load(os.path.join(self.path_save_models, "detr_best_model.pt")))

    @torch.no_grad()
    def validate(self, validation_metrics_classification: Tuple[nn.Module, ...] = (
            ClassificationAccuracy(),),
                 validation_metrics_bounding_box: Tuple[nn.Module, ...] = (
                         nn.L1Loss(), nn.MSELoss(), BoundingBoxIoU(),
                         BoundingBoxGIoU()),
                 validation_metrics_segmentation: Tuple[nn.Module, ...] = (
                         Accuracy(), Precision(), Recall(),
                         F1(), IoU(), MIoU(),
                         Dice(),
                         MeanAveragePrecision(), InstancesAccuracy()),
                 epoch: int = -1, number_of_plots: int = 5, train: bool = False) -> None:
        """
        Validation method
        :param validation_metrics_classification: (Tuple[nn.Module, ...]) Validation modules for classification
        :param validation_metrics_bounding_box: (Tuple[nn.Module, ...]) Validation modules for bounding boxes
        :param validation_metrics_segmentation: (Tuple[nn.Module, ...]) Validation modules for segmentation
        :param epoch: (int) Current epoch
        :param number_of_plots: (int) Number of validation plot to be produced
        :param train: (bool) Train flag if set best model is saved based on val iou
        """
        # DETR to device
        self.detr.to(self.device)
        # DETR into eval mode
        self.detr.eval()
        # Init dicts to store metrics
        metrics_classification = dict()
        metrics_bounding_box = dict()
        metrics_segmentation = dict()
        # Init indexes of elements to be plotted
        plot_indexes = np.random.choice(np.arange(0, len(self.validation_dataset)), number_of_plots, replace=False)
        # Main loop over the validation set
        for index, batch in enumerate(self.validation_dataset):
            # Get data from batch
            input, instance_labels, bounding_box_labels, class_labels = batch
            # Data to device
            input = input.to(self.device)
            instance_labels = iterable_to_device(instance_labels, device=self.device)
            bounding_box_labels = iterable_to_device(bounding_box_labels, device=self.device)
            class_labels = iterable_to_device(class_labels, device=self.device)
            # Get prediction
            class_predictions, bounding_box_predictions, instance_predictions = self.detr(input)
            # Perform matching
            matching_indexes = self.loss_function.matcher(class_predictions, bounding_box_predictions,
                                                          class_labels, bounding_box_labels)
            # Apply permutation to labels and predictions
            class_predictions, class_labels = self.loss_function.apply_permutation(prediction=class_predictions,
                                                                                   label=class_labels,
                                                                                   indexes=matching_indexes)
            bounding_box_predictions, bounding_box_labels = self.loss_function.apply_permutation(
                prediction=bounding_box_predictions,
                label=bounding_box_labels,
                indexes=matching_indexes)
            instance_predictions, instance_labels = self.loss_function.apply_permutation(
                prediction=instance_predictions,
                label=instance_labels,
                indexes=matching_indexes)
            for batch_index in range(len(class_labels)):
                # Calc validation metrics for classification
                for validation_metric_classification in validation_metrics_classification:
                    # Calc metric
                    metric = validation_metric_classification(
                        class_predictions[batch_index, :class_labels[batch_index].shape[0]].argmax(dim=-1),
                        class_labels[batch_index].argmax(dim=-1)).item()
                    # Save metric and name of metric
                    if validation_metric_classification.__class__.__name__ in metrics_classification.keys():
                        metrics_classification[validation_metric_classification.__class__.__name__].append(metric)
                    else:
                        metrics_classification[validation_metric_classification.__class__.__name__] = [metric]
                # Calc validation metrics for bounding boxes
                for validation_metric_bounding_box in validation_metrics_bounding_box:
                    # Calc metric
                    metric = validation_metric_bounding_box(
                        bounding_box_predictions[batch_index, :bounding_box_labels[batch_index].shape[0]],
                        bounding_box_labels[batch_index]).item()
                    # Save metric and name of metric
                    if validation_metric_bounding_box.__class__.__name__ in metrics_bounding_box.keys():
                        metrics_bounding_box[validation_metric_bounding_box.__class__.__name__].append(metric)
                    else:
                        metrics_bounding_box[validation_metric_bounding_box.__class__.__name__] = [metric]
                # Calc validation metrics for bounding boxes
                for validation_metric_segmentation in validation_metrics_segmentation:
                    # Calc metric
                    metric = validation_metric_segmentation(
                        instance_predictions[batch_index, :instance_labels[batch_index].shape[0]],
                        instance_labels[batch_index], class_label=class_labels[batch_index].argmax(dim=-1)).item()
                    # Save metric and name of metric
                    if validation_metric_segmentation.__class__.__name__ in metrics_segmentation.keys():
                        metrics_segmentation[validation_metric_segmentation.__class__.__name__].append(metric)
                    else:
                        metrics_segmentation[validation_metric_segmentation.__class__.__name__] = [metric]
            if index in plot_indexes:
                # Plot
                object_classes = class_predictions[0].argmax(dim=-1).cpu().detach()
                # Case the no objects are detected
                if object_classes.shape[0] > 0:
                    object_indexes = torch.from_numpy(np.argwhere(object_classes.numpy() > 0)[:, 0])
                    bounding_box_predictions = relative_bounding_box_to_absolute(
                        bounding_box_xcycwh_to_x0y0x1y1(
                            bounding_box_predictions[0, object_indexes].cpu().clone().detach()), height=input.shape[-2],
                        width=input.shape[-1])
                    plot_instance_segmentation_overlay_instances_bb_classes(image=input[0],
                                                                                 instances=(instance_predictions[0][
                                                                                                object_indexes] > 0.5).float(),
                                                                                 bounding_boxes=bounding_box_predictions,
                                                                                 class_labels=object_classes[
                                                                                     object_indexes],
                                                                                 show=False, save=True,
                                                                                 file_path=os.path.join(
                                                                                     self.path_save_plots,
                                                                                     "validation_plot_is_bb_c_{}_{}.png".format(
                                                                                         epoch + 1, index)))
        # Average metrics and save them in logs
        for metric_name in metrics_classification:
            self.logger.log(metric_name=metric_name + "_classification_val",
                            value=float(np.mean(metrics_classification[metric_name])))
        for metric_name in metrics_bounding_box:
            self.logger.log(metric_name=metric_name + "_bounding_box_val",
                            value=float(np.mean(metrics_bounding_box[metric_name])))
        for metric_name in metrics_segmentation:
            metric_values = np.array(metrics_segmentation[metric_name])
            # Save best mIoU model if training is utilized
            if train and "MIoU" in metric_name and float(np.mean(metrics_segmentation[metric_name])) > self.best_miou:
                # Save current mIoU
                self.best_miou = float(np.mean(metric_values[~np.isnan(metric_values)]))
                # Show best MIoU as process name
                setproctitle.setproctitle("Cell-DETR best MIoU={:.4f}".format(self.best_miou))
                # Save model
                torch.save(
                    self.detr.module.state_dict() if isinstance(self.detr, nn.DataParallel) else self.detr.state_dict(),
                    os.path.join(self.path_save_models, "detr_best_model.pt"))
            self.logger.log(metric_name=metric_name + "_segmentation_val",
                            value=float(np.mean(metric_values[~np.isnan(metric_values)])))
        # Save metrics
        self.logger.save_metrics(path=self.path_save_metrics)

    # @torch.no_grad()
    # def test(self, test_metrics_classification: Tuple[nn.Module, ...] = (validation_metric.ClassificationAccuracy(),),
    #          test_metrics_bounding_box: Tuple[nn.Module, ...] = (
    #                  nn.L1Loss(), nn.MSELoss(), validation_metric.BoundingBoxIoU(),
    #                  validation_metric.BoundingBoxGIoU()),
    #          test_metrics_segmentation: Tuple[nn.Module, ...] = (
    #                  validation_metric.Accuracy(), validation_metric.Precision(), validation_metric.Recall(),
    #                  validation_metric.F1(), validation_metric.IoU(), validation_metric.MIoU(),
    #                  validation_metric.Dice(),
    #                  validation_metric.MeanAveragePrecision(), validation_metric.InstancesAccuracy())) -> None:
    #     """
    #     Test method
    #     :param test_metrics_classification: (Tuple[nn.Module, ...]) Test modules for classification
    #     :param test_metrics_bounding_box: (Tuple[nn.Module, ...]) Test modules for bounding boxes
    #     :param test_metrics_segmentation: (Tuple[nn.Module, ...]) Test modules for segmentation
    #     """
    #     # DETR to device
    #     self.detr.to(self.device)
    #     # DETR into eval mode
    #     self.detr.eval()
    #     # Init dicts to store metrics
    #     metrics_classification = dict()
    #     metrics_bounding_box = dict()
    #     metrics_segmentation = dict()
    #     # Main loop over the test set
    #     for index, batch in enumerate(self.test_dataset):
    #         # Get data from batch
    #         input, instance_labels, bounding_box_labels, class_labels = batch
    #         # Data to device
    #         input = input.to(self.device)
    #         instance_labels = iterable_to_device(instance_labels, device=self.device)
    #         bounding_box_labels = iterable_to_device(bounding_box_labels, device=self.device)
    #         class_labels = iterable_to_device(class_labels, device=self.device)
    #         # Get prediction
    #         class_predictions, bounding_box_predictions, instance_predictions = self.detr(input)
    #         # Perform matching
    #         matching_indexes = self.loss_function.matcher(class_predictions, bounding_box_predictions,
    #                                                       class_labels, bounding_box_labels)
    #         # Apply permutation to labels and predictions
    #         class_predictions, class_labels = self.loss_function.apply_permutation(prediction=class_predictions,
    #                                                                                label=class_labels,
    #                                                                                indexes=matching_indexes)
    #         bounding_box_predictions, bounding_box_labels = self.loss_function.apply_permutation(
    #             prediction=bounding_box_predictions,
    #             label=bounding_box_labels,
    #             indexes=matching_indexes)
    #         instance_predictions, instance_labels = self.loss_function.apply_permutation(
    #             prediction=instance_predictions,
    #             label=instance_labels,
    #             indexes=matching_indexes)
    #         for batch_index in range(len(class_labels)):
    #             # Calc test metrics for classification
    #             for test_metric_classification in test_metrics_classification:
    #                 # Calc metric
    #                 metric = test_metric_classification(
    #                     class_predictions[batch_index, :class_labels[batch_index].shape[0]].argmax(dim=-1),
    #                     class_labels[batch_index].argmax(dim=-1)).item()
    #                 # Save metric and name of metric
    #                 if test_metric_classification.__class__.__name__ in metrics_classification.keys():
    #                     metrics_classification[test_metric_classification.__class__.__name__].append(metric)
    #                 else:
    #                     metrics_classification[test_metric_classification.__class__.__name__] = [metric]
    #             # Calc test metrics for bounding boxes
    #             for test_metric_bounding_box in test_metrics_bounding_box:
    #                 # Calc metric
    #                 metric = test_metric_bounding_box(
    #                     bounding_box_xcycwh_to_x0y0x1y1(
    #                         bounding_box_predictions[batch_index, :bounding_box_labels[batch_index].shape[0]]),
    #                     bounding_box_xcycwh_to_x0y0x1y1(bounding_box_labels[batch_index])).item()
    #                 # Save metric and name of metric
    #                 if test_metric_bounding_box.__class__.__name__ in metrics_bounding_box.keys():
    #                     metrics_bounding_box[test_metric_bounding_box.__class__.__name__].append(metric)
    #                 else:
    #                     metrics_bounding_box[test_metric_bounding_box.__class__.__name__] = [metric]
    #             # Calc test metrics for bounding boxes
    #             for test_metric_segmentation in test_metrics_segmentation:
    #                 # Calc metric
    #                 metric = test_metric_segmentation(
    #                     instance_predictions[batch_index, :instance_labels[batch_index].shape[0]],
    #                     instance_labels[batch_index], class_label=class_labels[batch_index].argmax(dim=-1)).item()
    #                 # Save metric and name of metric
    #                 if test_metric_segmentation.__class__.__name__ in metrics_segmentation.keys():
    #                     metrics_segmentation[test_metric_segmentation.__class__.__name__].append(metric)
    #                 else:
    #                     metrics_segmentation[test_metric_segmentation.__class__.__name__] = [metric]
    #         # Plot
    #         object_classes = class_predictions[0].argmax(dim=-1).cpu().detach()
    #         # Case the no objects are detected
    #         if object_classes.shape[0] > 0:
    #             object_indexes = torch.from_numpy(np.argwhere(object_classes.numpy() > 0)[:, 0])
                # bounding_box_predictions = relative_bounding_box_to_absolute(
                #     bounding_box_xcycwh_to_x0y0x1y1(
                #         bounding_box_predictions[0, object_indexes].cpu().clone().detach()), height=input.shape[-2],
                #     width=input.shape[-1])
                # plot_instance_segmentation_overlay_instances_bb_classes(image=input[0],
                #                                                              instances=(instance_predictions[0][
                #                                                                             object_indexes] > 0.5).float(),
                #                                                              bounding_boxes=bounding_box_predictions,
                #                                                              class_labels=object_classes[
                #                                                                  object_indexes],
                #                                                              show=False, save=True,
                #                                                              file_path=os.path.join(
                #                                                                  self.path_save_plots,
                #                                                                  "test_plot_{}_is_bb_c.png".format(
                #                                                                      index)))
                # plot_instance_segmentation_overlay_instances_bb_classes(image=input[0],
                # #                                                              instances=(instance_predictions[0][
                #                                                                             object_indexes] > 0.5).float(),
                #                                                              bounding_boxes=bounding_box_predictions,
                #                                                              class_labels=object_classes[
                #                                                                  object_indexes],
                #                                                              show=False, save=True,
                #                                                              show_class_label=False,
                #                                                              file_path=os.path.join(
                #                                                                  self.path_save_plots,
                #                                                                  "test_plot_{}_is_bb.png".format(
                #                                                                      index)))
                # plot_instance_segmentation_overlay_instances(image=input[0],
                #                                                   instances=(instance_predictions[0][
                #                                                                  object_indexes] > 0.5).float(),
                #                                                   class_labels=object_classes[object_indexes],
                #                                                   show=False, save=True,
                #                                                   file_path=os.path.join(
                #                                                       self.path_save_plots,
                #                                                       "test_plot_{}_is.png".format(index)))
                # plot_instance_segmentation_overlay_bb_classes(image=input[0],
        #         #                                                    bounding_boxes=bounding_box_predictions,
        #                                                            class_labels=object_classes[
        #                                                                object_indexes],
        #                                                            show=False, save=True,
        #                                                            file_path=os.path.join(
        #                                                                self.path_save_plots,
        #                                                                "test_plot_{}_bb_c.png".format(
        #                                                                    index)))
        #         plot_instance_segmentation_labels(
        #             instances=(instance_predictions[0][object_indexes] > 0.5).float(),
        #             bounding_boxes=bounding_box_predictions,
        #             class_labels=object_classes[object_indexes], show=False, save=True,
        #             file_path=os.path.join(self.path_save_plots, "test_plot_{}_bb_no_overlay_.png".format(index)),
        #             show_class_label=False, white_background=True)
        #         plot_instance_segmentation_map_label(
        #             instances=(instance_predictions[0][object_indexes] > 0.5).float(),
        #             class_labels=object_classes[object_indexes], show=False, save=True,
        #             file_path=os.path.join(self.path_save_plots, "test_plot_{}_no_overlay.png".format(index)),
        #             white_background=True)
        # # Average metrics and save them in logs
        # for metric_name in metrics_classification:
        #     print(metric_name + "_classification_test=", float(np.mean(metrics_classification[metric_name])))
        #     self.logger.log(metric_name=metric_name + "_classification_test",
        #                     value=float(np.mean(metrics_classification[metric_name])))
        # for metric_name in metrics_bounding_box:
        #     print(metric_name + "_bounding_box_test=", float(np.mean(metrics_bounding_box[metric_name])))
        #     self.logger.log(metric_name=metric_name + "_bounding_box_test",
        #                     value=float(np.mean(metrics_bounding_box[metric_name])))
        # for metric_name in metrics_segmentation:
        #     metric_values = np.array(metrics_segmentation[metric_name])
        #     print(metric_name + "_segmentation_test=", float(np.mean(metric_values[~np.isnan(metric_values)])))
        #     self.logger.log(metric_name=metric_name + "_segmentation_test",
        #                     value=float(np.mean(metric_values[~np.isnan(metric_values)])))
        # # Save metrics
        # self.logger.save_metrics(path=self.path_save_metrics)

class Arg():
    def __init__(self):
        self.cuda_devices = 0
        self.train = True
        self.test = False
        self.val = False
        self.data_parallel = False
        self.cpu = False
        self.epochs = 200
        self.lr_schedule = False
        self.ohem = False
        self.ohem_fraction = 0.75
        self.batch_size =4
        self.path_to_data = 'data'
        self.augmentation_p = 0.6
        self.lr_main = 1e-4
        self.lr_backbone = 1e-5
        self.lr_segmentation_head =1e-6
        self.no_pac = True
        self.load_model =""
        self.dropout = 0.05
        self.softmax = False
        self.no_deform_conv = True
        self.no_pau = True
        self.only_train_segmentation_head_after_epoch = 1000

args = Arg()
device = "cpu" if args.cpu else "cuda"
        

if True:
    # Init detr
    detr = CellDETR(num_classes=2 ,
                    segmentation_head_block=ResPACFeaturePyramidBlock if args.no_pac else ResFeaturePyramidBlock,
                    segmentation_head_final_activation=nn.Softmax if args.softmax else nn.Sigmoid,
                    backbone_convolution=nn.Conv2d if args.no_deform_conv else ModulatedDeformConvPack,
                    segmentation_head_convolution=nn.Conv2d if args.no_deform_conv else ModulatedDeformConvPack,
                    transformer_activation=nn.LeakyReLU if args.no_pau else PAU,
                    backbone_activation=nn.LeakyReLU if args.no_pau else PAU,
                    bounding_box_head_activation=nn.LeakyReLU if args.no_pau else PAU,
                    classification_head_activation=nn.LeakyReLU if args.no_pau else PAU,
                    segmentation_head_activation=nn.LeakyReLU if args.no_pau else PAU)
    if args.load_model != "":
        detr.load_state_dict(torch.load(args.load_model))
    # Print network
    # print(detr)
    # Print number of parameters
    print("# DETR parameters", sum([p.numel() for p in detr.parameters()]))
    # Init optimizer
    detr_optimizer = torch.optim.AdamW(detr.get_parameters(lr_main=args.lr_main, lr_backbone=args.lr_backbone),
                                       weight_decay=1e-06)
    detr_segmentation_optimizer = torch.optim.AdamW(detr.get_segmentation_head_parameters(lr=args.lr_segmentation_head),
                                                    weight_decay=1e-06)
    # Init data parallel if utilized
    if args.data_parallel:
        detr = torch.nn.DataParallel(detr)
    # Init learning rate schedule if utilized
    if args.lr_schedule:
        learning_rate_schedule = torch.optim.lr_scheduler.MultiStepLR(detr_optimizer, milestones=[50, 100], gamma=0.1)
    else:
        learning_rate_schedule = None
    # Init datasets
    training_dataset = DataLoader(
        CellInstanceSegmentation(path=os.path.join(args.path_to_data, "train"),
                                 augmentation_p=args.augmentation_p ),
        collate_fn=collate_function_cell_instance_segmentation, batch_size=1,
        shuffle=True)
    validation_dataset = DataLoader(
        CellInstanceSegmentation(path=os.path.join(args.path_to_data, "validation"),
                                 augmentation_p=0.0),
        collate_fn=collate_function_cell_instance_segmentation, batch_size=1, num_workers=1, shuffle=False)
    test_dataset = None
    # test_dataset = DataLoader(
    #     CellInstanceSegmentation(path=os.path.join(args.path_to_data, "test"),
    #                              augmentation_p=0.0),
    #     collate_fn=collate_function_cell_instance_segmentation, batch_size=1, num_workers=1, shuffle=False)
    # Model wrapper
    model_wrapper = ModelWrapper(detr=detr,
                                 detr_optimizer=detr_optimizer,
                                 detr_segmentation_optimizer=detr_segmentation_optimizer,
                                 training_dataset=training_dataset,
                                 validation_dataset=validation_dataset,
                                 test_dataset=test_dataset,
                                 loss_function=InstanceSegmentationLoss(
                                     segmentation_loss=SegmentationLoss(),
                                     ohem=args.ohem,
                                     ohem_faction=args.ohem_fraction),
                                 device=device,
                                 save_data_path=os.path.join(args.path_to_data, "saved_data"))
    # Perform training
    if args.train:
        model_wrapper.train(epochs=args.epochs,
                            optimize_only_segmentation_head_after_epoch=args.only_train_segmentation_head_after_epoch)
    # Perform validation
    if args.val:
        model_wrapper.validate(number_of_plots=30)
    # Perform testing
    if args.test:
        model_wrapper.test()
