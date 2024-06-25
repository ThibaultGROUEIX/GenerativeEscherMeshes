import math
import torch
import numpy as np
from escher.OTE.core.Constraints import Constraints
from escher.OTE.core.SparseSystem import SparseSystem
from escher.OTE.core.AffineTrans import AffineTrans
from escher.OTE.core.GlobalDeformation import MapType


class OrbifoldIVHybridConstraints(Constraints):
    """
    Orbifold signature "22*"
    """

    def __init__(self, vertices, sides):
        ###### constraint matrix
        self.sides = sides
        topleft = sides["top"][0]
        bottomright = sides["bottom"][0]
        bottomleft = sides["bottom"][-1]
        topright = sides["top"][-1]

        left = sides["left"]  # [1:-1]
        top = sides["top"]  # [:-1]
        bottom = sides["bottom"]  # [:-1]
        right = np.flip(sides["right"])  # [1:-1])

        sp = SparseSystem(vertices.shape[0])

        left1, left2 = np.split(left, 2)
        left2 = np.flip(left2)
        sp.generate_rotation_constraints(left1, left2, 2 * math.pi / 2, np.array([-1, 0]))

        right1, right2 = np.split(right, 2)
        right2 = np.flip(right2)
        sp.generate_rotation_constraints(right1, right2, 2 * math.pi / 2, np.array([1, 0]))

        # sp.generate_straight_line_constraint(bottom, np.array([0, 1]))
        # sp.generate_straight_line_constraint(top, np.array([0, 1]))

        sp.generate_fixed_constraints_y(bottom, -np.ones(bottom.shape))
        # sp.generate_fixed_constraints_y(np.array([bottomright]), np.array([-1]))

        # sp.generate_fixed_constraints_y(np.array([topleft]), np.array([1]))
        sp.generate_fixed_constraints_y(top[1:-1], np.ones(top[1:-1].shape))

        super(OrbifoldIVHybridConstraints, self).__init__(sp)

    def update_scaling(self):
        self.ad_hoc_scaling = 1

    def tiling_coloring_number(self):
        return 4

    def get_global_transformation_type(self):
        return MapType.NON_ISOTROPIC_SCALE

    def get_torus_directions(self):
        # This needs a drawing to understand
        return np.array([0, 4]), np.array([4, 0])

    def get_torus_cover(self, vertices, sides):
        theta = math.pi
        R180 = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

        first = AffineTrans(np.eye(2), np.array([0, 0]), 0, (0, 0))

        # p2 = np.array([0,1/math.sqrt(3)])
        second = AffineTrans(R180, np.array([2, 0]), 1, (0, 0))
        A = [first, second]
        Flip = np.diag([1, -1])
        for i in range(2):
            A.append(AffineTrans(Flip, np.array([[0, 2]]), i + 1, 0).compose(A[i], i + 2, (0, 0)))

        # third = AffineTrans(Flip_y, np.array([[0,2]]),0,1)
        # third = AffineTrans(R180, p2 - np.matmul(p2, R180.transpose()), 2)
        return A
        # A = [first]
        # second = AffineTrans(R90, np.array([0, 0]))
        # for i in range(3):
        #     A.append(second.compose(A[-1]))
        # return A

    def get_boundary(self):
        sides = self.sides
        return [(sides["top"], sides["left"]), (sides["bottom"], sides["right"])]

    def get_horizontal_symmetry_orientation(self):
        theta = np.pi / 4
        return torch.Tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
