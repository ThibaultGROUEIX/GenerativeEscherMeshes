import torch
import numpy as np
from escher.OTE.core.Constraints import Constraints
from escher.OTE.core.SparseSystem import SparseSystem
from escher.OTE.core.AffineTrans import AffineTrans
from escher.OTE.core.GlobalDeformation import MapType
from escher.OTE.tilings.ReflectSquare import ReflectSquareConstraints


class Reflect442Constraints(Constraints):
    """
    Orbifold signature "*442"
    """

    def __init__(self, vertices, sides):
        self.sides = sides
        ###### constraint matrix
        topright = sides["top"][-1]
        topleft = sides["top"][0]
        bottomleft = sides["bottom"][-1]
        bottomright = sides["bottom"][0]

        bottom = sides["bottom"]
        top = sides["top"]
        left = sides["left"]
        right = sides["right"]

        sp = SparseSystem(vertices.shape[0])
        n_up = np.array([0, 1])
        n_right = np.array([1, 0])
        sp.generate_straight_line_constraint(bottom[:-1], n_up)
        sp.generate_straight_line_constraint(right[:-1], n_right)
        sp.generate_straight_line_constraint(top[:-1], (n_up-n_right)/2)
        sp.generate_straight_line_constraint(left[:-1], (n_up-n_right)/2)
        sp.generate_fixed_constraints(np.array([bottomleft]), np.array([[-1, -1]]))
        sp.generate_fixed_constraints(np.array([bottomright]), np.array([[0, -1]]))
        sp.generate_fixed_constraints(np.array([topright]), np.array([[0, 0]]))
        sp.generate_fixed_constraints(np.array([topleft]), np.array([[-0.5, -0.5]]))
        # sp.generate_fixed_constraints(np.array([bottomleft]),np.array([[-1,-1]]))
        # sp.generate_fixed_constraints_y(np.array([topleft]),np.array([1]))
        # sp.generate_fixed_constraints_x(np.array([bottomright]),np.array([1]))

        # sp.generate_translation_constraint(bottom[1:-1],top[1:-1],np.array([0,2]))
        # sp.generate_translation_constraint(left[1:-1],right[1:-1],np.array([2,0]))
        # sp.generate_fixed_constraints(np.array([bottomleft]),np.array([[-1,-1]]))
        # sp.generate_fixed_constraints(np.array([bottomright]),np.array([[1,-1]]))
        # sp.generate_fixed_constraints(np.array([topright]),np.array([[1,1]]))
        # sp.generate_fixed_constraints(np.array([topleft]),np.array([[-1,1]]))
        self.reflect_square_constraints = ReflectSquareConstraints(vertices, sides)
        super(Reflect442Constraints, self).__init__(sp)
        self.update_scaling()

    def update_scaling(self):
        self.ad_hoc_scaling = np.sqrt(8)/2

    def get_global_transformation_type(self):
        return MapType.IDENTITY

    def get_torus_directions(self):
        # This needs a drawing to understand
        return np.array([0, 4]), np.array([4, 0])

    def get_torus_cover(self, vertices, sides):
        I = np.identity(2)
        Reflect_xy = np.array([[0, 1], [1, 0]])
        first = AffineTrans(I, np.array([0, 0]), 0)
        second = AffineTrans(Reflect_xy, np.array([0, 0]), 1)
        A = [first, second]
        
        # Now compose with reflect square
        self.reflect_square_constraints = ReflectSquareConstraints(vertices, sides)
        A_reflect = self.reflect_square_constraints.get_torus_cover(vertices, sides)

        B = list()
        i = 0
        scale_and_translate = AffineTrans(2 * np.eye(2), np.array([1, 1]), 0, (0, 0))
        for transfo_1 in A:
            for transfo_2 in A_reflect:
                B.append(transfo_2.compose(scale_and_translate.compose(transfo_1, 0, (0, 0)), i, (0, 0)))
                i += 1
        return B
    
    def get_boundary(self):
        sides = self.sides
        return [(sides["top"], sides["bottom"][-1::-1]), (sides["left"], sides["right"][-1::-1])]

    def tiling_coloring_number(self):
        return 8

    def get_horizontal_symmetry_orientation(self):
        # theta = pi / 4
        return torch.Tensor(np.eye(2))
