import torch
import numpy as np
from escher.OTE.core.Constraints import Constraints
from escher.OTE.core.SparseSystem import SparseSystem
from escher.OTE.core.AffineTrans import AffineTrans
from escher.OTE.core.GlobalDeformation import MapType


class CylinderConstraints(Constraints):
    """
    orbifold signature "**"
    """

    def __init__(self, vertices, sides):
        self.sides = sides
        ###### constraint matrix
        topright = sides["top"][-1]
        topleft = sides["top"][0]
        bottomleft = sides["bottom"][-1]
        bottomright = sides["bottom"][0]

        bottom = sides["bottom"]
        top = np.flip(sides["top"])
        left = sides["left"]
        right = sides["right"]

        sp = SparseSystem(vertices.shape[0])
        sp.generate_relative_translation_constraint(top[1:], top[0], bottom[1:], bottom[0], True)
        sp.generate_relative_translation_constraint(top[1:], top[0], bottom[1:], bottom[0], False)
        n_right = np.array([-1, 0])
        sp.generate_straight_line_constraint(left[:-1], n_right)
        sp.generate_straight_line_constraint(right[:-1], n_right)

        sp.generate_fixed_constraints_x(np.array([bottomleft]), np.array([-1]))
        sp.generate_fixed_constraints(np.array([bottomright]), np.array([[1, -1]]))
        sp.generate_fixed_constraints(np.array([topright]), np.array([[1, 1]]))
        # sp.generate_fixed_constraints(np.array([topleft]), np.array([[-1, 1]]))

        # sp.generate_fixed_constraints(np.array([bottomleft]),np.array([[-1,-1]]))
        # sp.generate_fixed_constraints_y(np.array([topleft]),np.array([1]))
        # sp.generate_fixed_constraints_x(np.array([bottomright]),np.array([1]))

        # sp.generate_translation_constraint(bottom[1:-1],top[1:-1],np.array([0,2]))
        # sp.generate_translation_constraint(left[1:-1],right[1:-1],np.array([2,0]))
        # sp.generate_fixed_constraints(np.array([bottomleft]),np.array([[-1,-1]]))
        # sp.generate_fixed_constraints(np.array([bottomright]),np.array([[1,-1]]))
        # sp.generate_fixed_constraints(np.array([topright]),np.array([[1,1]]))
        # sp.generate_fixed_constraints(np.array([topleft]),np.array([[-1,1]]))

        super(CylinderConstraints, self).__init__(sp)
        self.symmetric_copies = 2

    def update_scaling(self):
        self.ad_hoc_scaling = 1

    def get_global_transformation_type(self):
        return MapType.NON_ISOTROPIC_SCALE

    def get_torus_directions(self):
        # This needs a drawing to understand
        return np.array([0, 4]), np.array([4, 0])

    def get_torus_cover(self, vertices, sides):
        reflect = AffineTrans(np.diag([-1, 1]), np.array([0, 0]), 0, (0, 0))
        shift_vert = AffineTrans(np.eye(2), np.array([0, self.tile_width]), 0, (0, 0))
        A = [1, 2, 3, 4]
        A[0] = AffineTrans(np.eye(2), np.array([0, 0]), 0, (0, 0))
        A[1] = reflect
        A[2] = shift_vert.compose(reflect, 0, (0, 0))
        A[3] = shift_vert

        shift_diag = AffineTrans(np.eye(2), np.array([self.tile_width / 2, self.tile_width / 2]), 0, (0, 0))
        for i in range(len(A)):
            A[i] = A[i].compose(shift_diag, i, (0, 0))
        return A

    # def get_torus_cover(self, vertices, sides):
    #     I = np.identity(2)
    #     Flip = np.diag([-1, 1])
    #     up_delta = vertices[sides["left"][-1]] - vertices[sides["left"][0]]
    #     right_delta = vertices[sides["top"][-1]] - vertices[sides["top"][0]]
    #     left_delta = -right_delta
    #     left_delta[1] = -left_delta[1]
    #     return [
    #         AffineTrans(I, np.array([0, 0]), 0),
    #         AffineTrans(I, up_delta, 2),
    #         AffineTrans(Flip, right_delta, 2),
    #         AffineTrans(I, -up_delta, 2),
    #         AffineTrans(Flip, left_delta, 2),
    #         # AffineTrans(I, up_delta+right_delta,1),
    #     ]

    def get_boundary(self):
        sides = self.sides
        return [(sides["top"], sides["bottom"][-1::-1]), (sides["left"], sides["right"][-1::-1])]

    def tiling_coloring_number(self):
        return 8

    def get_horizontal_symmetry_orientation(self):
        # theta = pi / 4
        return torch.Tensor(np.eye(2))

    def get_symmetry_map(self, vertices, sides):
        """
        If the tiling has a reflectional symmetry, return the symmetry map. Otherwise, return None.
        ** Cylinder has a reflectional symmetry
        become
        ** Cylinder
        could also become
        o Torus
        
        The new scale of the Cylinder is the original scale.
        """

        reflect_x = AffineTrans(np.diag([-1, 1]), np.array([0, 0]), 0, (0, 0))
        A = [1, 2]
        A[0] = AffineTrans(np.eye(2), np.array([0, 0]), 0, (0, 0))
        A[1] = reflect_x

        # Shift to positive values before reflecting
        shift_diag_x = AffineTrans(np.eye(2), np.array([self.tile_width / 2, 0]), 0, (0, 0))
        A = [A[i].compose(shift_diag_x, i, (0, 0)) for i in range(len(A))]

        # Scale down the x axis since we reflected
        scale_down_x = AffineTrans(np.diag([0.5, 1]), np.array([0, 0]), 0, (0, 0))
        A = [scale_down_x.compose(A[i], i, (0, 0)) for i in range(len(A))]

        # # Shift back to a centered position
        # shift_diag_inverse = AffineTrans(np.eye(2), np.array([-self.tile_width / 2, -self.tile_width / 2]), 0, (0, 0))
        # A = [shift_diag_inverse.compose(A[i], i, (0, 0)) for i in range(len(A))]

        return A
