import math
import torch
import numpy as np
from escher.OTE.core.Constraints import Constraints
from escher.OTE.core.SparseSystem import SparseSystem
from escher.OTE.core.AffineTrans import AffineTrans
from escher.OTE.core.GlobalDeformation import MapType

### Orbifold signature "333"
# https://mcescher.com/gallery/symmetry/#iLightbox[gallery_image_1]/23

class OrbifoldIIHexagonConstraints(Constraints):
    """
    Orbifold signature "333"
    """

    def __init__(self, vertices, sides):
        ###### constraint matrix
        self.sides = sides
        P = [sides[i][0] for i in sides]
        edges_with_corners = [sides[i] for i in range(6)]
        sp = SparseSystem(vertices.shape[0])
        angle = 2 * math.pi / 3

        # reverse order of points for 5,1,3
        edges_with_corners[5]  = edges_with_corners[5][::-1]
        edges_with_corners[1]  = edges_with_corners[1][::-1]
        edges_with_corners[3]  = edges_with_corners[3][::-1]

        # loose the center of rotation
        edges_without_one_corner = [edges_with_corners[i][1:] for i in range(6)]

        P0 = vertices[P[0],:]#np.array([-math.sqrt(3) / 3, -1]) # Center of rotation
        # P1 = np.array([math.sqrt(3) / 3, -1])
        P2 = vertices[P[2],:] # np.array([2 * math.sqrt(3) / 3, 0])  # Center of rotation
        # P3 = np.array([math.sqrt(3) / 3, 1])
        P4 = vertices[P[4],:] #np.array([-math.sqrt(3) / 3, +1])  # Center of rotation
        # P5 = np.array([-2 * math.sqrt(3) / 3, 0])

        # First rotation
        sp.generate_rotation_constraints(edges_without_one_corner[0][:-1], edges_without_one_corner[5][:-1], angle, P0)
        # Second rotation
        sp.generate_rotation_constraints(edges_without_one_corner[2], edges_without_one_corner[1], angle, P2)
        # Second rotation
        sp.generate_rotation_constraints(edges_without_one_corner[4], edges_without_one_corner[3], angle, P4)
        # Other fixed points

        sp.generate_fixed_constraints(np.array([P[0]]), P0[np.newaxis, :])
        # sp.generate_fixed_constraints(np.array([P[1]]), P1[np.newaxis, :])
        sp.generate_fixed_constraints(np.array([P[2]]), P2[np.newaxis, :])
        # sp.generate_fixed_constraints(np.array([P[3]]), P3[np.newaxis, :])
        sp.generate_fixed_constraints(np.array([P[4]]), P4[np.newaxis, :])
        # sp.generate_fixed_constraints(np.array([P[5]]), P5[np.newaxis, :])

        super(OrbifoldIIHexagonConstraints, self).__init__(sp)
        self.update_scaling()

    def update_scaling(self):
        self.ad_hoc_scaling = np.sqrt(2.6 / math.sqrt(3)) # this is really just ad-hoc, there is no logic behind it

    def tiling_coloring_number(self):
        return 3

    def get_global_transformation_type(self):
        return MapType.IDENTITY

    def get_torus_directions(self):
        # This needs a drawing to understand
        # return np.array([2 * math.sqrt(3), 0]), np.array([math.sqrt(3), -self.tile_width / 2 - 2])
        c = np.sqrt(12)/3
        return np.array([2 * math.sqrt(3), 0])/c, np.array([math.sqrt(3), -self.tile_width / 2 - 2])/c
    def get_torus_cover(self, vertices, sides):
        # https://mcescher.com/gallery/symmetry/#iLightbox[gallery_image_1]/23
        theta = 2 * math.pi / 3
        R120 = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        R120 = AffineTrans(R120, np.array([0, 0]))

        first = AffineTrans(np.eye(2), np.array([-1, 0]), 0, (0, 0))

        A = [first]
        for i in range(2):
            A.append(R120.compose(A[-1], i + 1, (0, 0)))

        # fourth =  AffineTrans(np.eye(2), np.array([0, self.tile_width/2]), 0,(0,0))
        # fourth = R120.compose(fourth, 0,(0,0))
        # fourth = AffineTrans(np.eye(2), np.array([0, -self.tile_width]), 0,(0,0)).compose(fourth, 0,(0,0))
        # A.append(fourth)
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
