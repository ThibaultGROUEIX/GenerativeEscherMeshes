import math
import torch
import numpy as np
from escher.OTE.core.Constraints import Constraints
from escher.OTE.core.SparseSystem import SparseSystem
from escher.OTE.core.AffineTrans import AffineTrans
from escher.OTE.core.GlobalDeformation import MapType


class OrbifoldIHybridConstraints(Constraints):
    """
    Orbifold signature "4*2"
    """

    def __init__(self, vertices, sides):
        ###### constraint matrix
        self.tile_width = 1
        self.sides = sides
        topleft = sides["top"][0]  # this will be top corner
        topright = sides["top"][-1]  # this will be right corner
        bottomleft = sides["bottom"][-1]  # this will be left corner
        # bottomright = sides["bottom"][0]

        left = sides["left"][1:-1]
        top = np.flip(sides["top"])[1:-1]
        bottom_and_right = np.concatenate([sides["bottom"][0:-1], sides["right"][0:-1]])

        sp = SparseSystem(vertices.shape[0])

        sp.generate_rotation_constraints(left, top, 2 * math.pi / 4, np.array([0.0, 0]))
        sp.generate_straight_line_constraint(bottom_and_right, np.array([1, -1]))

        sp.generate_fixed_constraints(np.array([topright]), np.array([[1.0, 0.0]]))
        sp.generate_fixed_constraints(np.array([bottomleft]), np.array([[0.0, -1.0]]))
        sp.generate_fixed_constraints(np.array([topleft]), np.array([[0.0, 0]]))

        super(OrbifoldIHybridConstraints, self).__init__(sp)
        self.ad_hoc_scaling = 8.0

    def get_global_transformation_type(self):
        return MapType.IDENTITY

    def get_torus_tiling_width(self):
        return 2

    def get_torus_directions(self):
        # This needs a drawing to understand
        return np.array([0, 2]), np.array([2, 0])

    def get_torus_cover(self, vertices, sides):
        theta = 2 * math.pi / 4
        R90_mat = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        R90 = AffineTrans(R90_mat, np.array([[0, 0]]), 1, (0, 0))
        theta2 = 2 * math.pi / 8
        R45_mat = np.array([[math.cos(theta2), -math.sin(theta2)], [math.sin(theta2), math.cos(theta2)]])
        R45 = AffineTrans(R45_mat, np.array([0, 0]), 0, (0, 0))
        # p2 = np.array([0,1/math.sqrt(3)])
        A = list(range(8))
        A[0] = AffineTrans(R45_mat, np.array([[0, 0]]), 0, (0, 0))
        A[1] = A[0].compose(R90, 2, (0, 0))
        A[2] = A[1].compose(R90, 2, (0, 0))
        A[3] = A[2].compose(R90, 3, (0, 0))

        # flip_horz = AffineTrans(np.diag([-1,1]),np.array([0,0]),0,(0,0))
        flip_vert = AffineTrans(np.diag([1, -1]), np.array([0, math.sqrt(2) * self.tile_width / 2]), 0, (0, 0))
        flip_vert = R45.compose(flip_vert, 0, (0, 0))
        # flip_both = AffineTrans(np.diag([-1,-1]),np.array([0,0]),0,(0,0))
        for i in range(4):
            ind = i + 4
            A[ind] = flip_vert.compose(A[i], ind, (0, 0))

        for i in range(4):
            A[i] = R45.compose(A[i], i, (0, 0))
            # A[i] = AffineTrans(2*np.eye(2),np.array([0,0]),0,(0,0)).compose(A[i],i,(0,0))
        return A

        for i in range(4):
            ind = i + 8
            A[ind] = flip_both.compose(A[i], ind, (0, 0))

        for i in range(4):
            ind = i + 12
            A[ind] = flip_vert.compose(A[i], ind, (0, 0))

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
