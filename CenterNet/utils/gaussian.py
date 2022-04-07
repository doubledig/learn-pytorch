import math

import torch


def gaussian_radius(weight, height, min_overlap=0.7):
    r_w = weight * 0.5 * (math.sqrt(1 / min_overlap) - 1)
    r_h = height * 0.5 * (math.sqrt(1 / min_overlap) - 1)
    return torch.tensor([math.floor(r_w), math.floor(r_h)])


def gaussian2D(shape, sigma):
    m = torch.arange(-shape[0], shape[0] + 1)
    n = torch.arange(-shape[1], shape[1] + 1)
    x, y = torch.meshgrid(m, n, indexing='xy')

    h = torch.exp(-0.5 * ((x / sigma[0]) ** 2 + (y / sigma[1]) ** 2))
    return h


def draw_gaussian(heatmap, center, radius):
    diameter = (2 * radius + 1) / 6
    gaussian = gaussian2D(radius, sigma=diameter)

    left, right = int(center[0] - radius[0]), int(center[0] + radius[0])
    top, bottom = int(center[1] - radius[1]), int(center[1] + radius[1])

    heatmap[top:bottom + 1, left:right + 1] = torch.maximum(heatmap[top:bottom + 1, left:right + 1], gaussian)
    return heatmap
