import torch
import numpy as np
from escher.OTE.core.Constraints import Constraints
from escher.OTE.core.SparseSystem import SparseSystem
from escher.OTE.core.AffineTrans import AffineTrans
from escher.OTE.core.GlobalDeformation import MapType
import math

""" 
torus: o
"""


class TorusHexagonConstraints(Constraints):
    """
    Orbifold signature "o"
    """

    def __init__(self, vertices, sides):
        self.sides = sides
        P = [sides[i][0] for i in sides]
        edges_with_corners = [sides[i] for i in range(6)]
        sp = SparseSystem(vertices.shape[0])
        angle = 2 * math.pi / 3

        P_locations = [vertices[P[i], :] for i in range(6)]
        P_locations = [P_locations[i][np.newaxis, :] for i in range(6)]

        translation_0 = (P_locations[4] - P_locations[0]).squeeze()
        translation_1 = (P_locations[5] - P_locations[1]).squeeze()
        translation_2 = (P_locations[0] - P_locations[2]).squeeze()
        
        
        
        self.translation_0 = translation_0
        self.translation_1 = translation_1
        self.translation_2 = translation_2
        # reverse order of points for 5,1,3
        edges_with_corners[3] = edges_with_corners[3][::-1]
        edges_with_corners[4] = edges_with_corners[4][::-1]
        edges_with_corners[5] = edges_with_corners[5][::-1]

        # loose the center of rotation
        edges_without_corners = [edges_with_corners[i][1:-1] for i in range(6)]

        sp = SparseSystem(vertices.shape[0])
        sp.generate_translation_constraint(edges_without_corners[0], edges_without_corners[3], translation_0)
        sp.generate_translation_constraint(edges_without_corners[1], edges_without_corners[4], translation_1)
        sp.generate_translation_constraint(edges_without_corners[2], edges_without_corners[5], translation_2)

        [sp.generate_fixed_constraints(np.array([P[i]]), P_locations[i]) for i in range(6)]

        super(TorusHexagonConstraints, self).__init__(sp)

    def update_scaling(self):
        self.ad_hoc_scaling = 1

    def get_global_transformation_type(self):
        return MapType.SKEW

    def get_torus_directions(self):
        # This needs a drawing to understand
        return 2 * self.translation_0, 2 * self.translation_1

    def get_torus_cover(self, vertices, sides):
        I = np.identity(2)
        return [
            AffineTrans(I, np.array([0, 0]), 0),
            AffineTrans(I, self.translation_0, 1),
            AffineTrans(I, self.translation_1, 2),
            AffineTrans(I, self.translation_2, 3),
        ]

    def get_boundary(self):
        sides = self.sides
        return [(sides["top"], sides["bottom"][-1::-1]), (sides["left"], sides["right"][-1::-1])]

    def tiling_coloring_number(self):
        return 8

    def get_horizontal_symmetry_orientation(self):
        theta = np.pi / 4
        return torch.Tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
