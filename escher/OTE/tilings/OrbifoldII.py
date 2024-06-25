import math
import torch
import numpy as np
from escher.OTE.core.Constraints import Constraints
from escher.OTE.core.SparseSystem import SparseSystem
from escher.OTE.core.AffineTrans import AffineTrans
from escher.OTE.core.GlobalDeformation import MapType

### Orbifold signature "333"


class OrbifoldIIConstraints(Constraints):
    """
    Orbifold signature "333"
    """

    def __init__(self, vertices, sides):
        ###### constraint matrix
        self.sides = sides
        topleft = sides["top"][0]
        topright = sides["top"][-1]
        bottomleft = sides["bottom"][-1]
        bottomright = sides["bottom"][0]

        left = sides["left"][1:-1]
        top = np.flip(sides["top"])[1:-1]
        bottom = np.flip(sides["bottom"])[1:-1]
        right = sides["right"][1:-1]

        sp = SparseSystem(vertices.shape[0])

        sp.generate_rotation_constraints(left, top, 2 * math.pi / 3, np.array([0.0, 1.0]))
        sp.generate_rotation_constraints(right, bottom, 2 * math.pi / 3, np.array([0.0, -1.0]))

        sp.generate_fixed_constraints(np.array([bottomright]), np.array([[0.0, -1.0]]))
        sp.generate_fixed_constraints(np.array([topleft]), np.array([[0.0, 1, 0]]))
        sp.generate_fixed_constraints(np.array([topright]), np.array([[math.sqrt(3), 0]]))
        sp.generate_fixed_constraints(np.array([bottomleft]), np.array([[-math.sqrt(3), 0]]))
        super(OrbifoldIIConstraints, self).__init__(sp)
        self.update_scaling()

    def update_scaling(self):
        self.ad_hoc_scaling = np.sqrt(2 / math.sqrt(3))

    def tiling_coloring_number(self):
        return 3

    def get_global_transformation_type(self):
        return MapType.IDENTITY

    def get_torus_directions(self):
        # This needs a drawing to understand
        return np.array([2 * math.sqrt(3), 0]), np.array([math.sqrt(3), -self.tile_width / 2 - 2])

    def get_torus_cover(self, vertices, sides):
        theta = 2 * math.pi / 3
        R120 = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        R120 = AffineTrans(R120, np.array([0, 0]))

        first = AffineTrans(np.eye(2), np.array([0, -self.tile_width / 2]), 0, (0, 0))

        A = [first]
        for i in range(2):
            A.append(R120.compose(A[-1], i + 1, (0, 0)))

        # fourth =  AffineTrans(np.eye(2), np.array([0, self.tile_width/2]), 0,(0,0))
        # fourth = R120.compose(fourth, 0,(0,0))
        # fourth = AffineTrans(np.eye(2), np.array([0, -self.tile_width]), 0,(0,0)).compose(fourth, 0,(0,0))
        # A.append(fourth)
        return A
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
