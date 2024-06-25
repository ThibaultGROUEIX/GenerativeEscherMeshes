import torch
import numpy as np
import math
from escher.OTE.core.Constraints import Constraints
from escher.OTE.core.SparseSystem import SparseSystem
from escher.OTE.core.AffineTrans import AffineTrans
from escher.OTE.core.GlobalDeformation import MapType
from escher.OTE.tilings.ReflectSquare import ReflectSquareConstraints


class Reflect632Constraints(Constraints):
    """
    Orbifold signature "*632"
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
        n_other = np.array([np.sqrt(3) / 3, -1])
        #normalise
        n_other = n_other / np.linalg.norm(n_other)
        sp.generate_straight_line_constraint(bottom[:-1], n_up)
        sp.generate_straight_line_constraint(right[:-1], n_right)
        sp.generate_straight_line_constraint(top[:-1], n_other)
        sp.generate_straight_line_constraint(left[:-1], n_other)
        sp.generate_fixed_constraints(np.array([bottomleft]), np.array([[-1, -math.sqrt(3) / 3]]))
        sp.generate_fixed_constraints(np.array([bottomright]), np.array([[0, -math.sqrt(3) / 3]]))
        sp.generate_fixed_constraints(np.array([topright]), np.array([[0, 0]]))
        sp.generate_fixed_constraints(np.array([topleft]), np.array([[-0.5, -math.sqrt(3) / 6]]))
        # sp.generate_fixed_constraints(np.array([bottomleft]),np.array([[-1,-1]]))
        # sp.generate_fixed_constraints_y(np.array([topleft]),np.array([1]))
        # sp.generate_fixed_constraints_x(np.array([bottomright]),np.array([1]))

        # sp.generate_translation_constraint(bottom[1:-1],top[1:-1],np.array([0,2]))
        # sp.generate_translation_constraint(left[1:-1],right[1:-1],np.array([2,0]))
        # sp.generate_fixed_constraints(np.array([bottomleft]),np.array([[-1,-1]]))
        # sp.generate_fixed_constraints(np.array([bottomright]),np.array([[1,-1]]))
        # sp.generate_fixed_constraints(np.array([topright]),np.array([[1,1]]))
        # sp.generate_fixed_constraints(np.array([topleft]),np.array([[-1,1]]))
        super(Reflect632Constraints, self).__init__(sp)
        self.update_scaling()

    def update_scaling(self):
        self.ad_hoc_scaling = np.sqrt(4/ (np.sqrt(3)/6))

    def get_global_transformation_type(self):
        return MapType.IDENTITY

    def get_torus_directions(self):
        # This needs a drawing to understand
        return np.array([1, np.sqrt(3)]), np.array([2, 0])

    def get_torus_cover(self, vertices, sides):
        I = np.identity(2)
        first = AffineTrans(I, np.array([0, 0]), 0)
        
        Reflect_y = np.array([[-1, 0], [0, 1]])
        second = AffineTrans(Reflect_y, np.array([0, 0]), 1)
        A = [first, second]
        
        reflect3 = np.array([[1/2, np.sqrt(3)/2], [np.sqrt(3)/2, -1/2]])
        A.append(AffineTrans(reflect3, np.array([0, 0]), 2))
        
        reflect4 = AffineTrans( np.array([[1/2, -np.sqrt(3)/2], [-np.sqrt(3)/2, -1/2]]), np.array([0, 0]), 3)
        B = list()
        i = 3
        for transfo_1 in A:
            B.append(reflect4.compose(transfo_1, i, (0, 0)))
            i += 1
        A = A + B
        
        # a bit more involved, because the reflection is not through the origin
        reflect5 = AffineTrans( np.array([[-1/2, -np.sqrt(3)/2], [-np.sqrt(3)/2, 1/2]]), np.array([0.5, np.sqrt(3)/6]), 3)
        translate_to_origin = AffineTrans( np.eye(2), np.array([-0.5, -np.sqrt(3)/6]), 3)
        B = list()
        for transfo_1 in A:
            B.append(reflect5.compose(translate_to_origin.compose(transfo_1, i, (0, 0)), i, (0, 0)))
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
