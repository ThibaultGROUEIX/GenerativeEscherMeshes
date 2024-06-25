import sys
import torch

from escher.rendering.render_tiling_core import get_color_function
from escher.rendering.render_tiling_from_pkl import render_from_pkl

"""
This function is used to render a tile from a pkl file of results.
The figure specific functions should be called from within the directory of the pkl file.
Site to get color palettes : https://coolors.co/177e89-a4b0f5-f58f29-9cfc97-4b3b47
"""


def render_poppy_overview():
    colors = torch.Tensor([[25, 88, 181], [218, 88, 28], [176, 39, 34]])
    col = get_color_function(colors / 255.0)
    render_from_pkl("results.pkl", color=col)
    return


# render_poppy_overview()


def render_dragon_teaser():
    colors = torch.Tensor([[25, 88, 181], [75, 59, 71], [23, 126, 137]])
    colors = torch.Tensor([[199, 164, 37], [77, 142, 158], [181, 52, 35]])
    col = get_color_function(colors / 255.0)
    render_from_pkl("results.pkl", color=col)
    return


render_dragon_teaser()


def render_balletdancer_teaser():
    colors = torch.Tensor([[203, 167, 81], [221, 162, 187], [119, 185, 185]])
    col = get_color_function(colors / 255.0)
    render_from_pkl("results.pkl", color=col)
    return


render_balletdancer_teaser()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        colors = torch.Tensor([[25, 88, 181], [218, 88, 28], [176, 39, 34]])
        col = get_color_function(colors / 255.0)
        render_from_pkl(sys.argv[1], color=col)
