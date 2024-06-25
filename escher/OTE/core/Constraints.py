from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from escher.OTE.core.AffineTrans import AffineTrans
from escher.OTE.core.SparseSystem import SparseSystem
from escher.OTE.core.GlobalDeformation import MapType


class Constraints(ABC):
    def __init__(self, sp: SparseSystem):
        I, J, vals, b = sp.aggregate()
        self.cIJ = np.stack([I, J], axis=1)
        self.cV = vals
        self.b = b
        self.tile_width = 2
        self.ad_hoc_scaling = 1.0

    def get_torus_directions(self):
        return np.array([self.tile_width, 0]), np.array([0, self.tile_width])

    def get_torus_directions_with_scaling(self):
        try:
            # This can fail if the results.pkl is old
            self.ad_hoc_scaling = self.ad_hoc_scaling
        except:
            self.ad_hoc_scaling = 1.0
        vec_1, vec_2 = self.get_torus_directions()
        return vec_1 * self.ad_hoc_scaling, vec_2 * self.ad_hoc_scaling

    def get_torus_cover_with_scaling(self, vertices, sides):
        A = self.get_torus_cover(vertices=vertices, sides=sides)
        scaling = AffineTrans(self.ad_hoc_scaling * np.eye(2), [0, 0], 0, (0, 0))
        return [scaling.compose(A[i], A[i].orientation_index, (0, 0)) for i in range(len(A))]

    @abstractmethod
    def tiling_coloring_number(self):
        pass

    @abstractmethod
    def get_boundary(self):
        pass

    @abstractmethod
    def get_torus_cover(self, vertices, sides):
        pass

    @abstractmethod
    def get_horizontal_symmetry_orientation(self):
        pass

    @abstractmethod
    def get_global_transformation_type() -> MapType:
        pass

    def get_torus_tiling_width(self):
        return 2 * self.tile_width

    # @abstractmethod
    # def get_symmetry_map(self, vertices, sides):
    #     """If the tiling has a reflectional symmetry, return the symmetry map. Otherwise, return None."""
    #     # Commenting out because this function is not implemented for all orbifolds
    #     pass
    
    
    def get_tiling(self, half_grid_size, vertices, sides):
        I = np.identity(2)
        maps = []

        vec_1, vec_2 = self.get_torus_directions_with_scaling()
        torus_generator_maps = self.get_torus_cover_with_scaling(vertices=vertices, sides=sides)
        for i in range(0, half_grid_size + 1):
            for j in range(0, half_grid_size + 1):
                for sign_i in [-1, 1]:
                    for sign_j in [-1, 1]:
                        if i == 0 and sign_i == -1:
                            continue
                        if j == 0 and sign_j == -1:
                            continue
                        translation = AffineTrans(I, sign_i * i * vec_1 + sign_j * j * vec_2, None, None)
                        for generator in torus_generator_maps:
                            m = translation.compose(generator, generator.orientation_index, (i, j))
                            maps.append(m)
        return maps

    def get_more_tiling(self, half_grid_size, previous_half_grid_size, vertices, sides):
        I = np.identity(2)
        maps = []
        vec_1, vec_2 = self.get_torus_directions_with_scaling()

        torus_generator_maps = self.get_torus_cover_with_scaling(vertices=vertices, sides=sides)
        for i in range(-half_grid_size, half_grid_size + 1):
            for j in range(-half_grid_size, half_grid_size + 1):
                if abs(i) <= previous_half_grid_size and abs(j) <= previous_half_grid_size:
                    continue
                translation = AffineTrans(I, i * vec_1 + j * vec_2, None, None)
                for generator in torus_generator_maps:
                    m = translation.compose(generator, generator.orientation_index, (i, j))
                    maps.append(m)
        return maps
    

    def get_num_orientation(self, vertices, sides):
        return len(self.get_torus_cover(vertices=vertices, sides=sides))
