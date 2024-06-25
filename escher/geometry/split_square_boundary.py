import numpy as np


def split_square_boundary(V, bdry):
    x = V[bdry, 0]
    y = V[bdry, 1]
    top = take_piece_and_sort(x, y, bdry)
    right = take_piece_and_sort(-y, x, bdry)
    bottom = take_piece_and_sort(-x, -y, bdry)
    left = take_piece_and_sort(y, -x, bdry)
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def take_piece_and_sort(sort_coord, choose_coord, bdry):
    sort_inds = np.argsort(sort_coord)
    inds = choose_coord[sort_inds] == max(choose_coord)
    return bdry[sort_inds[inds]]
