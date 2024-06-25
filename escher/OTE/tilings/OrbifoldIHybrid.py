import math
import torch
import numpy as np
from escher.OTE.core.Constraints import Constraints
from escher.OTE.core.SparseSystem import SparseSystem
from escher.OTE.core.AffineTrans import AffineTrans
from escher.OTE.core.GlobalDeformation import MapType
from escher.OTE.tilings.ReflectSquare import ReflectSquareConstraints


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
        bottomright = sides["bottom"][0]

        left = sides["left"]
        top = np.flip(sides["top"])
        bottom = sides["bottom"]
        right = np.flip(sides["right"])
        sp = SparseSystem(vertices.shape[0])

        sp.generate_rotation_constraints(left[1:-1], top[1:-1], 2 * math.pi / 4, np.array([-1, 1]))
        sp.generate_straight_line_constraint(bottom[:-1], np.array([0, -1]))
        sp.generate_straight_line_constraint(right[:-1], np.array([-1, 0]))

        sp.generate_fixed_constraints(np.array([topright]), np.array([[1.0, 1.0]]))
        sp.generate_fixed_constraints(np.array([bottomleft]), np.array([[-1.0, -1.0]]))
        sp.generate_fixed_constraints(np.array([topleft]), np.array([[-1.0, 1.0]]))
        sp.generate_fixed_constraints(np.array([bottomright]), np.array([[1.0, -1.0]]))

        self.reflect_square_constraints = ReflectSquareConstraints(vertices, sides)
        super(OrbifoldIHybridConstraints, self).__init__(sp)
        self.update_scaling()

    def update_scaling(self):
        self.ad_hoc_scaling = 2

    def get_global_transformation_type(self):
        return MapType.IDENTITY

    def get_torus_tiling_width(self):
        return 2

    def get_torus_directions(self):
        # This needs a drawing to understand
        return np.array([0, 4]), np.array([4, 0])

    def get_torus_cover(self, vertices, sides):
        theta = 2 * math.pi / 4
        R90_mat = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        R90 = AffineTrans(R90_mat, np.array([[0, 0]]), 1, (0, 0))

        theta2 = 2 * math.pi / 8
        R45_mat = np.array([[math.cos(theta2), -math.sin(theta2)], [math.sin(theta2), math.cos(theta2)]])
        R45 = AffineTrans(R45_mat, np.array([0, 0]), 0, (0, 0))
        # p2 = np.array([0,1/math.sqrt(3)])
        translation = AffineTrans(np.eye(2), np.array([1, -1]), 0, (0, 0))

        A = list()
        A.append(translation)
        A.append(R90.compose(translation, 1, (0, 0)))
        A.append(R90.compose(A[1], 2, (0, 0)))
        A.append(R90.compose(A[2], 3, (0, 0)))
        self.reflect_square_constraints = ReflectSquareConstraints(vertices, sides)

        A_reflect = self.reflect_square_constraints.get_torus_cover(vertices, sides)

        B = list()
        i = 0
        half_scale = AffineTrans(0.5 * np.eye(2), np.array([0, 0]), 0, (0, 0))
        for transfo_1 in A:
            for transfo_2 in A_reflect:
                B.append(transfo_2.compose(half_scale.compose(transfo_1, 0, (0, 0)), i, (0, 0)))
                i += 1
        return B
        # # flip_horz = AffineTrans(np.diag([-1,1]),np.array([0,0]),0,(0,0))
        # flip_vert = AffineTrans(np.diag([1, -1]), np.array([0, math.sqrt(2) * self.tile_width / 2]), 0, (0, 0))
        # flip_vert = R45.compose(flip_vert, 0, (0, 0))
        # # flip_both = AffineTrans(np.diag([-1,-1]),np.array([0,0]),0,(0,0))
        # for i in range(4):
        #     ind = i + 4
        #     A[ind] = flip_vert.compose(A[i], ind, (0, 0))

        # for i in range(4):
        #     A[i] = R45.compose(A[i], i, (0, 0))
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

    def get_num_orientation(self, vertices, sides):
        return len(self.get_torus_cover(vertices=vertices, sides=sides))
