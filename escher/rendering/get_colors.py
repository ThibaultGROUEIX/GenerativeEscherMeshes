import torch
import numpy as np


class ColorIndex:
    def __init__(self, n_orientation_index, n_sub_tile_index, color_strategy="RANDOM", list_fixed_colors=None):
        """Spits a color for each tile, given its orientation and subtile index

        Args:
            list_fixed_colors: _description_. Is a list of list of 3 elements. Defaults to None.
        """
        self.n_orientation_index = n_orientation_index
        self.n_sub_tile_index = n_sub_tile_index
        self.color_strategy = color_strategy
        self.n_hyper_tile_index = 2
        scale = 0.35
        self.list_rand_colors = 0.8 * torch.rand(n_sub_tile_index, n_orientation_index, 3).unsqueeze(2).unsqueeze(2) * (1 - scale) + scale
        if list_fixed_colors is not None:
            self.list_fixed_colors = torch.Tensor(list_fixed_colors).unsqueeze(2).unsqueeze(2) / 255.0
        else:
            # default args
            colors = torch.Tensor(
                [
                    [244, 67, 54],
                    [232, 30, 99],
                    [156, 39, 176],
                    [103, 58, 183],
                    [63, 81, 181],
                    [33, 150, 243],
                    [3, 169, 244],
                    [0, 188, 212],
                    [63, 81, 181],
                    [33, 150, 243],
                    [3, 169, 244],
                    [0, 188, 212],
                    [56, 142, 60],
                    [76, 175, 80],
                    [139, 195, 74],
                    [205, 220, 57],
                    [0, 150, 136],
                    [76, 175, 80],
                    [139, 195, 74],
                    [205, 220, 57],
                    [34, 198, 246],
                    [0, 188, 212],
                    [255, 235, 59],
                    [255, 193, 7],
                    [255, 235, 59],
                    [255, 193, 7],
                    [255, 152, 0],
                    [255, 87, 34],
                    [145, 0, 0],
                    [255, 87, 34],
                    [145, 0, 0],
                    [255, 87, 34],
                ]
            )
            self.list_fixed_colors = torch.Tensor(colors).reshape(4, 8, 3)

        scale = 0.23
        self.list_fixed_colors = torch.rand(4, 8, 3) * (1 - scale) + scale
        print(self.list_fixed_colors)
        self.list_fixed_colors = torch.Tensor(
            [
                [
                    [0.4057, 0.7126, 0.8807],
                    [0.8827, 0.2997, 0.3478],
                    [0.7737, 0.7287, 0.4159],
                    [0.2584, 0.5218, 0.4774],
                    [0.9963, 0.2959, 0.8406],
                    [0.5456, 0.6760, 0.6091],
                    [0.8328, 0.3799, 0.8677],
                    [0.8230, 0.7307, 0.9964],
                ],
                [
                    [0.4194, 0.4104, 0.8866],
                    [0.5989, 0.4359, 0.6686],
                    [0.5566, 0.9203, 0.7844],
                    [0.6645, 0.9656, 0.3679],
                    [0.9414, 0.8075, 0.5348],
                    [0.3156, 0.4085, 0.9848],
                    [0.4909, 0.7760, 0.9607],
                    [0.2577, 0.2552, 0.4422],
                ],
                [
                    [0.4092, 0.5829, 0.4209],
                    [0.5791, 0.7780, 0.5849],
                    [0.9434, 0.7266, 0.5538],
                    [0.5475, 0.4846, 0.6204],
                    [0.9851, 0.5118, 0.9489],
                    [0.7625, 0.8143, 0.3936],
                    [0.9361, 0.6184, 0.4833],
                    [0.2558, 0.4155, 0.7561],
                ],
                [
                    [0.8853, 0.5041, 0.5565],
                    [0.7124, 0.5660, 0.6036],
                    [0.4091, 0.4681, 0.9452],
                    [0.3813, 0.8336, 0.7960],
                    [0.9130, 0.7268, 0.6472],
                    [0.4099, 0.6425, 0.9548],
                    [0.5865, 0.2500, 0.2758],
                    [0.8418, 0.9932, 0.8398],
                ],
            ]
        )

        self.list_fixed_colors = self.list_fixed_colors.unsqueeze(2).unsqueeze(2) 
        self.list_fixed_colors = self.list_fixed_colors  # * (1 - scale) + scale

        self.identity = torch.Tensor([[1, 1, 1]]).unsqueeze(1).unsqueeze(1)

    def get_index(self, orientation_index, sub_tile_index=0, hyper_tile_index=(0, 0)):
        return orientation_index * self.n_sub_tile_index + sub_tile_index
        # return  hyper_tile_index[0] * self.n_sub_tile_index * self.n_hyper_tile_index**2 + \
        #         hyper_tile_index[1] * self.n_sub_tile_index * self.n_hyper_tile_index + \
        #         orientation_index * self.n_sub_tile_index + \
        #         sub_tile_index

    def random_color_func(
        self,
        orientation_index,
        sub_tile_index=0,
        hyper_tile_index=None,
    ):
        scale = 0.23
        return torch.rand(1, 1, 3) * (1 - scale) + scale
        return self.list_rand_colors[sub_tile_index][orientation_index]

    def periodic_colors(self, orientation_index, sub_tile_index=0, hyper_tile_index=None):
        list_fixed_colors = self.list_fixed_colors
        if orientation_index >= self.list_fixed_colors.shape[1]:
            list_fixed_colors = self.list_fixed_colors.view(1, -1)
        return list_fixed_colors[sub_tile_index][orientation_index]

    def get_color(self, orientation_index, sub_tile_index=0, hyper_tile_index=0):
        if self.color_strategy == "RANDOM":
            print(sub_tile_index, sub_tile_index)
            return self.periodic_colors(orientation_index, sub_tile_index)
            return self.random_color_func(orientation_index, sub_tile_index)
        elif self.color_strategy == "PERIODIC":
            return self.periodic_colors(orientation_index, sub_tile_index)
        elif self.color_strategy == "ALREADY_HAS_COLOR":
            return self.identity
        else:
            raise NotImplementedError(f"color strategy {self.color_strategy} not implemented")
