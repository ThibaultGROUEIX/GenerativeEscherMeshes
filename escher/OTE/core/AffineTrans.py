import torch
import numpy as np


class AffineTrans:
    def __init__(self, A, b, orientation_index=0, translation_index=(0, 0)):
        if torch.is_tensor(A):
            A = A.cpu().detach().numpy()
        if torch.is_tensor(b):
            b = b.cpu().detach().numpy()
        self.A = A
        self.b = b
        self.orientation_index = orientation_index
        self.translation_index = translation_index

    def is_identity(self):
        return np.array_equal(self.A, np.eye(2)) and np.array_equal(self.b, np.zeros(2))

    def equal(self, other):
        # return True
        return np.allclose(self.A, other.A, 1e-1, 1e-1) and np.allclose(self.b, other.b, 1e-1, 1e-1)

    def map(self, vertices):
        if torch.is_tensor(vertices):
            A = torch.from_numpy(self.A.transpose()).to(vertices.device).float()
            if len(vertices.shape) == 3:
                A = A.unsqueeze(0)
            return torch.matmul(vertices, A) + torch.from_numpy(self.b).cuda().type(vertices.dtype)
        else:
            return np.matmul(vertices, self.A.transpose()) + self.b

    def compose(self, other, orientation_index, translation_index):
        # A1(A2*v+b2)+b1 = A1A2v + A1b2+b1
        return AffineTrans(
            np.matmul(self.A, other.A),
            np.matmul(other.b, self.A.transpose()) + self.b,
            orientation_index,
            translation_index,
        )
