import math
import torch
import numpy as np
from escher.OTE.core.Constraints import Constraints
from escher.OTE.core.SparseSystem import SparseSystem
from escher.OTE.core.AffineTrans import AffineTrans
from escher.OTE.core.GlobalDeformation import MapType

### Orbifold signature "2*22"


class RightAngleHybridConstraints(Constraints):
    """
    Orbifold signature "2*22"
    """

    def __init__(self, vertices, sides):
        ###### constraint matrix
        self.sides = sides
        topleft = sides["top"][0]  # this will be top corner
        bottomright = sides["bottom"][0]  # this will be right corner
        bottomleft = sides["bottom"][-1]  # this will be left corner
        topright = sides["top"][-1]  # this will be rotation point

        left = sides["left"][0:-1]  # left reflection
        top = sides["top"][1:-1]
        bottom = sides["bottom"][0:-1]  # bottom reflection
        right = np.flip(sides["right"][1:-1])

        sp = SparseSystem(vertices.shape[0])

        sp.generate_rotation_constraints(top, right, 2 * math.pi / 2, np.array([1, 1]))
        sp.generate_straight_line_constraint(bottom, np.array([0, 1]))
        sp.generate_straight_line_constraint(left, np.array([1, 0]))

        sp.generate_fixed_constraints(np.array([topright]), np.array([[1.0, 1.0]]))
        sp.generate_fixed_constraints(np.array([bottomleft]), np.array([[0.0, 0.0]]))
        sp.generate_fixed_constraints(np.array([bottomright]), np.array([[2.0, 0.0]]))
        sp.generate_fixed_constraints(np.array([topleft]), np.array([[0.0, 2]]))

        super(RightAngleHybridConstraints, self).__init__(sp)
        self.update_scaling()

    def update_scaling(self):
        self.ad_hoc_scaling = np.sqrt(2)

    def tiling_coloring_number(self):
        return 8

    def get_global_transformation_type(self):
        return MapType.IDENTITY

    def get_torus_directions(self):
        # This needs a drawing to understand
        return np.array([0, 4]), np.array([4, 0])

    # def get_torus_cover(self, vertices, sides):
    #     I = np.identity(2)
    #     return [
    #         AffineTrans(I, np.array([0, 0]), 0),
    #         AffineTrans(I, np.array([0, 2]), 1),
    #         AffineTrans(I, np.array([2, 0]), 2),
    #         AffineTrans(I, np.array([2, 2]), 3),
    #     ]
    def get_torus_cover(self, vertices, sides):
        first = AffineTrans(np.eye(2), np.array([0, 0]), 0, (0, 0))
        R180 = AffineTrans(np.array([[-1, 0], [0, -1]]), np.array([0, 0]))
        ReflectX = AffineTrans(np.array([[-1, 0], [0, 1]]), np.array([0, 0]))
        ReflectY = AffineTrans(np.array([[1, 0], [0, -1]]), np.array([0, 0]))
        A = [first]
        A.append(R180.compose(A[0], 1, (0, 0)))
        A.append(ReflectX.compose(A[0], 2, (0, 0)))
        A.append(ReflectX.compose(A[1], 3, (0, 0)))
        Translate = AffineTrans(np.array([[1, 0], [0, 1]]), np.array([2, 2]))
        # A.append(Translate.compose(A[0], 4, (0, 0)))
        for i in range(4):
            A.append(Translate.compose(A[i], i + 4, (0, 0)))
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
