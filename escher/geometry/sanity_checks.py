import igl
import numpy as np
import torch


def check_triangle_orientation(V, T):
    # print(V)
    # print(T)
    if isinstance(V, torch.Tensor):
        V = V.cpu().detach().numpy()
    if isinstance(T, torch.Tensor):
        T = T.cpu().detach().numpy()
    pa = V[T[:, 0], :].astype(np.float64)
    pb = V[T[:, 1], :].astype(np.float64)
    pc = V[T[:, 2], :].astype(np.float64)
    det = np.cross(pb - pa, pc - pa)
    if det.max() < 0:
        det = -det
    if det.min() < -1e-6:
        print(det.min())
        import pickle

        print(f"min: {det.min()}")
        with open("triangle_orientation_error.pkl", "wb") as f:
            pickle.dump({"V": V, "T": T, "min": det.min()}, f)
            print("Triangle orientation is wrong! Saved debug info to triangle_orientation_error.pkl")
        # raise (Exception("Triangle orientation is wrong!"))
