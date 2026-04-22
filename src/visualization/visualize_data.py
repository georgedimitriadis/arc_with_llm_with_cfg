
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from numpy._typing import NDArray

from structure.geometry.basic_geometry import Colour
from load_data import load_task_data_as_pixels

def data_to_colour(pixels: NDArray):
    """
    Takes an NxM NDArray that is assumed to have integers corresponding to the basic_geometry.Colour and turns it into
    an NxMx4 NDArray that can be shown with a plt.imshow with the correct Colours.
    :param pixels: The NDArray of pixels.
    :return: The NxMx4 NDArray to be passed to the plt.imshow().
    """
    pixels = np.array(pixels, np.uint8)
    size = pixels.shape
    result = np.zeros((size[0], size[1], 4)).astype(float)
    for i in range(size[0]):
        for j in range(size[1]):
            result[i, j, :] = np.array(Colour.map_int_to_colour[pixels[i, j]], float)
    return result


def add_vhlines_to_plot(axis: plt.Axes, data: NDArray, extent: List, thin_lines: bool = False):
    """
    Adds vertical and horizontal lines to a plot to demarcate the individual pixels.
    :param axis: The axis of the plot.
    :param data: The NxMx4 NDArray that is being plotted.
    :param extent: The extent of the vertical and horizontal lines. That is a 4 element List usually given as
    [-0.5, data.shape[1]+0.5, -0.5, data.shape[0]+0.5]. The 0.5s are there to make the lines go all the way to the edge of the plot.
    :param thin_lines: If True, make the lines really thin.
    :return: The axis with the lines drawn.
    """
    data = np.array(data)
    num_y = data.shape[0]
    num_x = data.shape[1]
    if not thin_lines:
        linewidths = 0.5 if np.max([num_y, num_x]) > 6 else 1.5
    else:
        linewidths = 0.1
    axis.vlines(np.arange(extent[0], extent[1]), ymin=extent[2], ymax=extent[3], colors='#BFBFBF', linewidths=linewidths)
    axis.hlines(np.arange(extent[2], extent[3]), xmin=extent[0], xmax=extent[1], colors='#BFBFBF', linewidths=linewidths)

    return axis


def plot_pixels(pixels:NDArray, extent: None | List = None, axes: plt.Axes | None = None,
                thin_lines: bool = False, save=None, show_interactive=True):
    """
    It plots a canvas of pixels (i.e. an NxM NDArray whose integers correspond to the basic_geometry.Colour scheme).
    :param pixels: The NxM NDArray of pixels.
    :param extent: The extent of the vertical and horizontal lines demarcating the pixels on the plot. If None then
    the function has as default extent=[0, data.shape[1], 0, data.shape[0]].
    :param axes: If given then plot on this axis. Otherwise create a new figure.
    :param thin_lines: If True, make the vertical and horizontal lines thin.
    :return: The figure and the axes if a new figure is created, otherwise just the axes.
    """
    if axes is None:
        fig = plt.figure()
        ax = fig.add_subplot()
        if not show_interactive:
            plt.switch_backend('Agg')
    else:
        ax = axes
    pixels_for_visualisation = data_to_colour(pixels)
    #pixels_for_visualisation = np.flipud(pixels_for_visualisation)
    num_y = pixels.shape[0]
    num_x = pixels.shape[1]
    if extent is None:
        extent = (0, num_x, 0, num_y)
    ax.imshow(pixels_for_visualisation, origin='lower', extent=extent, interpolation='None', aspect='equal')
    if thin_lines:
        for axes in ['top', 'bottom', 'left', 'right']:
            ax.spines[axes].set_linewidth(0.1)
        ax.tick_params(axis='both', labelsize=2, grid_linewidth=0.1)

    ax = add_vhlines_to_plot(axis=ax, data=pixels, extent=extent, thin_lines=thin_lines)

    if save is not None:
        fig.savefig(save)
    if not show_interactive:
        plt.switch_backend('Qt5Agg')

    if axes is not None:
        return ax

    return fig, ax


def plot_task(task_name:str, with_white_background:bool=False, save=None, show_interactive=True):
    """
    Plots all canvases of a Task as pairs from top to bottom. Each pair is an input (left) -> output (right) canvas of one
    example. The top examples are the training ones and the bottom the test one(s).
    :param task_name: The name of the task
    :param with_white_background: If True it shows each canvas within the 32x32 white background
    :param save: Save in the file specified by the save if save is not None.
    :param show_interactive: If True then show the generated plot. Otherwise do not show.
    :return:
    """
    if not show_interactive:
       plt.switch_backend('Agg')

    task = load_task_data_as_pixels(task_name, with_white_background=with_white_background)
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(n, 2, figsize=(4,  2 *n), dpi=100)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = t["input"], t["output"]
        axs[fig_num][0] = plot_pixels(t_in, axes=axs[fig_num][0])
        axs[fig_num][0].set_title(f'Train {i} Input')
        axs[fig_num][1] = plot_pixels(t_out, axes=axs[fig_num][1])
        axs[fig_num][1].set_title(f'Train {i} Output')
        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = t["input"], t["output"]
        axs[fig_num][0] = plot_pixels(t_in, axes=axs[fig_num][0])
        axs[fig_num][0].set_title(f'Test {i} Input')
        axs[fig_num][1] = plot_pixels(t_out, axes=axs[fig_num][1])
        axs[fig_num][1].set_title(f'Test {i} Output')
        fig_num += 1
    fig.subplots_adjust(left=0.06, bottom=0.04, right=0.98, top=0.96, wspace=0.09, hspace=0.4)
    fig.tight_layout()

    if save is not None:
        fig.savefig(save)
    if not show_interactive:
        plt.switch_backend('TKAgg')

