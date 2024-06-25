import numpy as np
import torch


# from KKTBuilderPackage.core.boundary_constraints_helpers import *
from escher.OTE.core.Constraints import Constraints


class OTESolver:
    def __init__(self, edge_pairs, V, constraints: Constraints):
        self.n_verts = V.shape[0]
        self.__verts = V
        IJ = []
        wInds = []
        wFacs = []

        ######## creating laplacian
        for i in range(edge_pairs.shape[0]):
            # M[IJ[i,0]][IJ[i,1]] = w[i]
            # M[IJ[i,1]][IJ[i,0]] = w[i]
            IJ.append((edge_pairs[i, 0], edge_pairs[i, 1]))
            IJ.append((edge_pairs[i, 1], edge_pairs[i, 0]))
            # newW.append(w[i])
            # newW.append(w[i])
            wInds.append(i)
            wInds.append(i)
            wFacs.append(1)
            wFacs.append(1)
            for j in range(2):
                # M[IJ[i,j]][IJ[i,j]] = M[IJ[i,j]][IJ[i,j]] - w[i]
                IJ.append((edge_pairs[i, j], edge_pairs[i, j]))
                # newW.append(-w[i])
                wInds.append(i)
                wFacs.append(-1)

        ####### duplicate laplacian to create block lap
        # L 0
        # 0 L
        IJ = np.asarray(IJ)
        IJ = np.concatenate((IJ, IJ + IJ.max() + 1), axis=0)
        wInds.extend(wInds)
        wFacs.extend(wFacs)
        self.wInds = wInds
        self.wFacs = wFacs

        # c_stuff = self.pinned_bdry_constraint_matrix(V,bdry)
        cIJ = constraints.cIJ.copy()  # constraint_data["cIJ"]
        cV = constraints.cV.copy()  # constraint_data["cV"]
        b = constraints.b.copy()  # constraint_data["b"]
        # bump the constraint block to be below the laplacian
        cIJ[:, 0] += np.max(IJ[:, 0]) + 1

        # the transpose of the constraint matrix
        cIJt = np.stack((cIJ[:, 1], cIJ[:, 0]), axis=1)
        cIJ = np.concatenate((cIJ, cIJt), axis=0)
        cV = np.tile(cV, 2)  # .extend(cV)

        self.cV = cV

        # add constraint matrix to the indices
        IJ = np.concatenate((IJ, cIJ), axis=0)

        ### making the 2D matrix into a 3D tensor with batch_size=1
        self.IJ = np.concatenate((np.zeros([IJ.shape[0], 1]), IJ), axis=1)

        self.b = np.concatenate((np.zeros((V.shape[0] * 2)), b), 0)

    def build_KKT_matrix(self, w):
        ####### generate w
        Wvals = w[self.wInds] * torch.tensor(self.wFacs).unsqueeze(1)
        # add the C vals
        vals = Wvals.squeeze()
        vals = torch.cat([vals, torch.from_numpy(self.cV)], axis=0)
        ##### create the matrix
        KKT = torch.sparse_coo_tensor(np.transpose(self.IJ), vals).coalesce()
        return KKT

    def solve(self, w):
        verify = True
        KKT = self.build_KKT_matrix(w)
        #### transform into torch_sparse_solve's specific format (float64, make b batched - KKT was generated batched) and solve
        KKT = KKT.to(torch.float64)
        b = torch.from_numpy(self.b).unsqueeze(0).unsqueeze(2).to(torch.float64)
        # X = torch_sparse_solve.solve(KKT, b)[0, :, 0]
        A = KKT.to_dense()
        X_dense = torch.linalg.solve(A, b.squeeze())[0]
        X = X_dense
        success = True
        if verify:
            nrm = torch.linalg.norm(torch.matmul(A, X_dense) - b.squeeze())
            if nrm > 1e-8:
                print(f"linear system not solved, error: {nrm}")
                success = False
                
            # A = np.squeeze(KKT.detach().to_dense().numpy())
            # rhs = b.detach().squeeze().numpy()
            # res = np.matmul(A, X.detach().numpy()) - rhs
            # assert np.abs(res).max() < 1e-5

        #### take only the vertex coordinates (discard lagrange multipliers) and split into n-by-2
        mapped = torch.zeros([self.n_verts, 2])
        mapped[:, 0] = X[0 : mapped.shape[0]]
        mapped[:, 1] = X[mapped.shape[0] : mapped.shape[0] * 2]
        return mapped, X, success


""" Timings

Dense
40x40 mesh, CPU torch.linalg.solve: 0.06s -> 17it/s
40x40 mesh, GPU torch.linalg.solve: 0.08s -> 12it/s
40x40 mesh, CPU torch.sparse.solve: 0.03s -> 33it/s  ## But the verify test does not pass, somehow we have much less

Conclusion : These routines are actually slower in their GPU implementations. 
Construction of the KKT matrice and moving it to CPU/GPU is negligible compared to the solve time.
"""
