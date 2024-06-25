import torch
import numpy as np
from escher.OTE.core.Constraints import Constraints
from escher.OTE.core.SparseSystem import SparseSystem
from escher.OTE.core.AffineTrans import AffineTrans
from escher.OTE.core.GlobalDeformation import MapType


class ProjectivePlaneConstraints(Constraints):
    """
    Orbifold signature "22x"
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
        sp.generate_relative_translation_constraint(top[1:-1], topleft, bottom[1:-1], bottomright, True, True)
        sp.generate_relative_translation_constraint(top[1:-1], top[0], bottom[1:-1], bottomright, False)
        sp.generate_relative_translation_constraint(left[1:-1], left[0], right[1:-1], right[0], True)
        sp.generate_relative_translation_constraint(left[1:-1], left[0], right[1:-1], right[0], False, True)

        sp.generate_fixed_constraints(np.array([bottomleft]), np.array([[-1, -1]]))
        sp.generate_fixed_constraints(np.array([bottomright]), np.array([[1, -1]]))
        sp.generate_fixed_constraints(np.array([topright]), np.array([[1, 1]]))
        sp.generate_fixed_constraints(np.array([topleft]), np.array([[-1, 1]]))
        # sp.generate_fixed_constraints(np.array([bottomleft]),np.array([[-1,-1]]))
        # sp.generate_fixed_constraints_y(np.array([topleft]),np.array([1]))
        # sp.generate_fixed_constraints_x(np.array([bottomright]),np.array([1]))

        # sp.generate_translation_constraint(bottom[1:-1],top[1:-1],np.array([0,2]))
        # sp.generate_translation_constraint(left[1:-1],right[1:-1],np.array([2,0]))
        # sp.generate_fixed_constraints(np.array([bottomleft]),np.array([[-1,-1]]))
        # sp.generate_fixed_constraints(np.array([bottomright]),np.array([[1,-1]]))
        # sp.generate_fixed_constraints(np.array([topright]),np.array([[1,1]]))
        # sp.generate_fixed_constraints(np.array([topleft]),np.array([[-1,1]]))

        super(ProjectivePlaneConstraints, self).__init__(sp)

    def update_scaling(self):
        self.ad_hoc_scaling = 1

    def get_global_transformation_type(self):
        return MapType.NON_ISOTROPIC_SCALE

    def get_torus_directions(self):
        # This needs a drawing to understand
        return np.array([0, 4]), np.array([4, 0])

    def get_torus_cover(self, vertices, sides):
        I = np.identity(2)
        first = AffineTrans(I, np.array([0, 0]), 0, (0, 0))

        FlipY = np.diag([1, -1])
        A = [first]
        A.append(AffineTrans(FlipY, np.array([[2, 0]]), 1, 0).compose(first, 1, (0, 0)))

        FlipX = np.diag([-1, 1])
        for i in range(2):
            A.append(AffineTrans(FlipX, np.array([[0, 2]]), i + 2, 0).compose(A[i], i + 2, (0, 0)))

        return A

    def get_boundary(self):
        sides = self.sides
        return [(sides["top"], sides["bottom"][-1::-1]), (sides["left"], sides["right"][-1::-1])]

    def tiling_coloring_number(self):
        return 8

    def get_horizontal_symmetry_orientation(self):
        # theta = pi / 4
        return torch.Tensor(np.eye(2))
