import math
import torch
import numpy as np
from escher.OTE.core.Constraints import Constraints
from escher.OTE.core.SparseSystem import SparseSystem
from escher.OTE.core.AffineTrans import AffineTrans
from escher.OTE.core.GlobalDeformation import MapType

### Orbifold notation "2222"


class OrbifoldIVConstraints(Constraints):
    """
    Orbifold signature "2222"
    """

    def __init__(self, vertices, sides):
        ###### constraint matrix
        self.sides = sides
        left = sides["left"]
        top = sides["top"]
        bottom = np.flip(sides["bottom"])
        right = np.flip(sides["right"])

        topleft, topright = np.split(top, 2)
        bottomleft, bottomright = np.split(bottom, 2)
        topright = np.flip(topright)
        bottomright = np.flip(bottomright)

        sp = SparseSystem(vertices.shape[0])
        sp.generate_rotation_constraints(topleft[1:], topright[1:], math.pi, np.array([0.0, 1.0]))
        sp.generate_rotation_constraints(bottomleft[1:], bottomright[1:], math.pi, np.array([0.0, -1.0]))
        sp.generate_translation_constraint(left, right, np.array([2.0, 0.0]))
        sp.generate_fixed_constraints(np.array([left[0]]), np.array([[-1.0, -1.0]]))
        sp.generate_fixed_constraints(np.array([left[-1]]), np.array([[-1.0, 1.0]]))

        super(OrbifoldIVConstraints, self).__init__(sp)

    def update_scaling(self):
        self.ad_hoc_scaling = 1

    def get_global_transformation_type(self):
        return MapType.SKEW

    def get_torus_directions(self):
        # This needs a drawing to understand
        return np.array([0, 4]), np.array([4, 0])

    def get_torus_cover(self, vertices, sides):
        R180 = np.array([[-1, 0], [0, -1]])
        first = AffineTrans(np.eye(2), np.array([0, 0]), 0, (0, 0))
        second = AffineTrans(R180, np.array([0, 2]), 1, 0)
        # second = AffineTrans(R180, np.array([0, -2]), 2, 0)
        third = AffineTrans(np.eye(2), np.array([2, 0]), 2, 1)
        fourth = AffineTrans(R180, np.array([2, 2]), 3, 0)
        return [first, second, third, fourth]  # , third, fourth]
        # A = [first]
        # second = AffineTrans(R90, np.array([0, 0]))
        # for i in range(3):
        #     A.append(second.compose(A[-1]))
        # return A

    def get_boundary(self):
        sides = self.sides
        return [(sides["top"], sides["left"]), (sides["bottom"], sides["right"])]

    def tiling_coloring_number(self):
        return 8

    def get_horizontal_symmetry_orientation(self):
        theta = np.pi / 4
        return torch.Tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
