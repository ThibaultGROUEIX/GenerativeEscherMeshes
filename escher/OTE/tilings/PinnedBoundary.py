import numpy as np


# from escher.OTE.SparseSystem import SparseSystem
from escher.OTE.core.Constraints import Constraints
from escher.OTE.core.SparseSystem import SparseSystem
from escher.OTE.core.AffineTrans import AffineTrans


class PinnedBoundaryConstraints(Constraints):
    def __init__(self, V, bdry):
        ###### constraint matrix
        self.bdry = bdry
        cIJ = []
        cV = []
        for i in range(bdry.shape[0]):
            ind = bdry[i]
            # C[i,ind] = 1
            cIJ.append((i, ind))
            cV.append(1)
        cIJ = np.asarray(cIJ)
        # constraint for y is same as x
        cIJ2 = cIJ + [np.max(cIJ[:, 0]) + 1, np.max(cIJ[:, 1]) + 1]
        cIJ = np.concatenate((cIJ, cIJ2), axis=0)
        cV.extend(cV)
        cV = np.array(cV)
        bx = V[bdry, 0]
        by = V[bdry, 1]
        b = np.concatenate((bx, by))  # torch.cat((torch.from_numpy(bx),torch.from_numpy(by)),axis=0)
        super(PinnedBoundaryConstraints, self).__init__(cIJ, cV, b)

    def get_torus_cover(self, vertices, sides):
        raise Exception("not a torus cover")

    def get_tiling(self, torus_shifts=1):
        return [AffineTrans(np.identity(2), np.array([0, 0]))]

    def tiling_coloring_number(self):
        return 8

    def get_boundary(self):
        return [self.bdry]
