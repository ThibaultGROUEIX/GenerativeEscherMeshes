import igl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.spatial import Delaunay


def split_tris(V, T):
    """
    Splits a mesh V,T and return a bool array A s.t. A[i] == true iff T's 3 vertices
    are to the left of the diagonal of the mesh
    """
    mid = (V[:, 0].max() + V[:, 0].min()) / 2
    left = V[:, 0] < mid
    leftT = left[T]
    fullLeftT = np.all(leftT, axis=1)
    return fullLeftT


if __name__ == "__main__":
    nx, ny = (20, 20)
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xv, yv = np.meshgrid(x, y)
    xv = xv.ravel()
    yv = yv.ravel()
    points = np.stack((xv, yv), axis=1)
    tri = Delaunay(points)
    tinds = split_tris(points, tri.simplices)
    plt.triplot(xv, yv, tri.simplices[tinds, :])
    plt.triplot(xv, yv, tri.simplices[np.logical_not(tinds), :])
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.show()
