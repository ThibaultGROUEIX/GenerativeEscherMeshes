import torch
import numpy as np
from escher.OTE.core.Constraints import Constraints
from escher.OTE.core.SparseSystem import SparseSystem
from escher.OTE.core.AffineTrans import AffineTrans
from escher.OTE.core.GlobalDeformation import MapType


class Reflect333Constraints(Constraints):
    """
    Orbifold signature "*333"
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
        n_1 = np.array([1, 0])
        n_2 = np.array([1/2, -np.sqrt(3)/2])
        n_3 = np.array([-1/2, -np.sqrt(3)/2])
        #normalise
        n_1 = n_1 / np.linalg.norm(n_1)
        n_2 = n_2 / np.linalg.norm(n_2)
        n_3 = n_3 / np.linalg.norm(n_3)
        sp.generate_straight_line_constraint(bottom[:-1], n_3)
        sp.generate_straight_line_constraint(right[:-1], n_1)
        sp.generate_straight_line_constraint(top[:-1], n_2)
        sp.generate_straight_line_constraint(left[:-1], n_2)
        sp.generate_fixed_constraints(np.array([bottomleft]), np.array([[-np.sqrt(3)/2, -1/2]]))
        sp.generate_fixed_constraints(np.array([bottomright]), np.array([[0, -1]]))
        sp.generate_fixed_constraints(np.array([topright]), np.array([[0, 0]]))
        sp.generate_fixed_constraints(np.array([topleft]), np.array([[-np.sqrt(3)/4, -1/4]]))
        # sp.generate_fixed_constraints(np.array([bottomleft]),np.array([[-1,-1]]))
        # sp.generate_fixed_constraints_y(np.array([topleft]),np.array([1]))
        # sp.generate_fixed_constraints_x(np.array([bottomright]),np.array([1]))

        # sp.generate_translation_constraint(bottom[1:-1],top[1:-1],np.array([0,2]))
        # sp.generate_translation_constraint(left[1:-1],right[1:-1],np.array([2,0]))
        # sp.generate_fixed_constraints(np.array([bottomleft]),np.array([[-1,-1]]))
        # sp.generate_fixed_constraints(np.array([bottomright]),np.array([[1,-1]]))
        # sp.generate_fixed_constraints(np.array([topright]),np.array([[1,1]]))
        # sp.generate_fixed_constraints(np.array([topleft]),np.array([[-1,1]]))
        super(Reflect333Constraints, self).__init__(sp)

    def update_scaling(self):
        self.ad_hoc_scaling = np.sqrt(4/ (np.sqrt(3)/4))

    def get_global_transformation_type(self):
        return MapType.IDENTITY

    def get_torus_directions(self):
        # This needs a drawing to understand
        return np.array([np.sqrt(3), 0]), np.array([np.sqrt(3)/2,3/2])

    def get_torus_cover(self, vertices, sides):
        I = np.identity(2)
        first = AffineTrans(I, np.array([0, 0]), 0)
        A = [first]
        
        reflect1 = np.array([[1/2, np.sqrt(3)/2], [np.sqrt(3)/2, -1/2]])
        A.append(AffineTrans(reflect1, np.array([0, 0]), 1))
        
        reflect2 = np.array([[1/2, -np.sqrt(3)/2], [-np.sqrt(3)/2, -1/2]])
        A.append(AffineTrans(reflect2, np.array([0, 0]), 1))

        Reflect_y = np.array([[-1, 0], [0, 1]])
        Reflect_y = AffineTrans(Reflect_y, np.array([0, 0]), 2)
        B = list()
        i = 3
        for transfo_1 in A:
            B.append(Reflect_y.compose(transfo_1, i, (0, 0)))
            i += 1
        A = A + B
        return A

    def get_boundary(self):
        sides = self.sides
        return [(sides["top"], sides["bottom"][-1::-1]), (sides["left"], sides["right"][-1::-1])]

    def tiling_coloring_number(self):
        return 8

    def get_horizontal_symmetry_orientation(self):
        # theta = pi / 4
        return torch.Tensor(np.eye(2))
