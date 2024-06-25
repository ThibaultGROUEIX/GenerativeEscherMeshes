import math
import torch
import numpy as np
from escher.OTE.core.Constraints import Constraints
from escher.OTE.core.SparseSystem import SparseSystem
from escher.OTE.core.AffineTrans import AffineTrans
from escher.OTE.core.GlobalDeformation import MapType

### Orbifold signature "3*3"


class OrbifoldIIHybridConstraints(Constraints):
    """
    Orbifold signature "3*3"
    """

    def __init__(self, vertices, sides):
        ###### constraint matrix
        self.sides = sides
        topleft = sides["top"][0]  # this will be top corner
        topright = sides["top"][-1]  # this will be right corner
        bottomleft = sides["bottom"][-1]  # this will be left corner
        # bottomright = sides["bottom"][0]

        left = sides["left"][1:-1]
        top = np.flip(sides["top"])[1:-1]
        bottom_and_right = np.concatenate([sides["bottom"][0:-1], sides["right"][0:-1]])

        sp = SparseSystem(vertices.shape[0])

        sp.generate_rotation_constraints(left, top, 2 * math.pi / 3, np.array([0.0, 1 / math.sqrt(3)]))
        sp.generate_straight_line_constraint(bottom_and_right, np.array([0, 1]))

        sp.generate_fixed_constraints(np.array([topright]), np.array([[1.0, 0.0]]))
        sp.generate_fixed_constraints(np.array([bottomleft]), np.array([[-1.0, 0.0]]))
        sp.generate_fixed_constraints(np.array([topleft]), np.array([[0.0, 1 / math.sqrt(3)]]))
        super(OrbifoldIIHybridConstraints, self).__init__(sp)
        self.update_scaling()

    def update_scaling(self):
        self.ad_hoc_scaling = np.sqrt(4 * math.sqrt(3))

    def get_global_transformation_type(self):
        return MapType.IDENTITY

    def get_torus_directions(self):
        # This needs a drawing to understand
        return np.array([1, -3 / math.sqrt(3)]), np.array([1, +3 / math.sqrt(3)])

    def get_torus_cover(self, vertices, sides):
        theta = 2 * math.pi / 3
        R120 = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        R120 = AffineTrans(R120, np.array([0, 0]))

        first = AffineTrans(np.eye(2), np.array([0, -1 / math.sqrt(3)]), 0, (0, 0))

        A = [first]
        for i in range(2):
            A.append(R120.compose(A[-1], i + 1, (0, 0)))

        inverse_translation = AffineTrans(np.eye(2), np.array([0, 1 / math.sqrt(3)]), 0, (0, 0))
        for i in range(3):
            A[i] = inverse_translation.compose(A[i], i, (0, 0))

        Flip_y = np.diag([1, -1])
        second = AffineTrans(Flip_y, np.array([[0, 0]]), 0, 1)
        for i in range(3):
            A.append(second.compose(A[i], i + 3, (0, 0)))

        return A
        # A = [first]
        # second = AffineTrans(R90, np.array([0, 0]))
        # for i in range(3):
        #     A.append(second.compose(A[-1]))
        # fourth =  AffineTrans(np.eye(2), np.array([0, self.tile_width/2]), 0,(0,0))
        # fourth = R120.compose(fourth, 0,(0,0))
        # fourth = AffineTrans(np.eye(2), np.array([0, -self.tile_width]), 0,(0,0)).compose(fourth, 0,(0,0))
        # A.append(fourth)
        return A

    # def get_torus_cover(self, vertices, sides):
    #     theta = 2 * math.pi / 3
    #     R120 = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    #     p1 = np.array([0, 1 / math.sqrt(3)])
    #     # p2 = np.array([0,1/math.sqrt(3)])
    #     first = AffineTrans(R120, p1 - np.matmul(p1, R120.transpose()), 2)
    #     Flip_y = np.diag([1, -1])
    #     second = AffineTrans(Flip_y, np.array([[0, 0]]), 0, 1)
    #     return [first, second]
    #     # A = [first]
    #     # second = AffineTrans(R90, np.array([0, 0]))
    #     # for i in range(3):
    #     #     A.append(second.compose(A[-1]))
    #     # return A

    def get_boundary(self):
        sides = self.sides
        return [(sides["top"], sides["left"]), (sides["bottom"], sides["right"])]

    def tiling_coloring_number(self):
        return 6

    def get_horizontal_symmetry_orientation(self):
        theta = np.pi / 4
        return torch.Tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
