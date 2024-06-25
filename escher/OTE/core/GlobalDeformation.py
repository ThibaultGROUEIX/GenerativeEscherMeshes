import torch
from enum import IntEnum


def _rotation_from_angle(theta):
    cs = torch.cos(theta)
    sn = torch.sin(theta)
    # global_R = torch.Tensor([[cs,-sn],[sn,cs]]).cuda()
    r1 = torch.concat([cs, -sn], axis=0)
    r2 = torch.concat([sn, cs], axis=0)
    return torch.stack((r1, r2), axis=1)


def map(vertices, A):
    """return the input vertices, mapped through A (which was returned from GlobalDeformation.get_matrix())"""
    return torch.matmul(vertices, torch.transpose(A, 0, 1).to(vertices.device).type(vertices.dtype))


class MapType:
    """
    different types of global linear transformations to choose from
    """

    IDENTITY = 0  # identity, i.e., no op
    NON_ISOTROPIC_SCALE = 1  # scale x and y axis
    SKEW = 2  # general skew transformation


class GlobalDeformation:
    def __init__(self, init_rotation=torch.eye(2), device="cpu", singular_value_bound=2, random_init=False):
        self.init_rotation = init_rotation.to(device)
        if not random_init:
            self.theta1 = torch.nn.Parameter(torch.zeros(1).to(device))
            self.theta2 = torch.nn.Parameter(torch.zeros(1).to(device))
            self.singular_value = torch.nn.Parameter(torch.ones(1).to(device))
        else:
            self.theta1 = torch.nn.Parameter(torch.rand(1).to(device) * 5)
            self.theta2 = torch.nn.Parameter(torch.rand(1).to(device) * 5)
            self.singular_value = torch.nn.Parameter(torch.rand(1).to(device) * 1000)
        self.singular_value_bound = singular_value_bound

    # def to(self,device):
    #     self.theta1 = self.theta1.to(device)
    #     self.theta2 = self.theta2.to(device)
    #     self.singular_value = self.singular_value.to(device)
    #     return self
    def get_matrix(self, global_rotation: bool, map_type: MapType):
        # A = R2*D*R1
        if global_rotation:
            R2 = _rotation_from_angle(self.theta1)
        else:
            R2 = torch.eye(2).to(self.theta1.device)
        if map_type is not MapType.IDENTITY:
            sv = torch.sigmoid(self.singular_value)  # sv in [0,1]
            sv = sv * (self.singular_value_bound - 1 / self.singular_value_bound)  # sv in [0, K - 1/K]
            sv = sv + 1 / self.singular_value_bound  # sv in [1/K, K]
        else:
            sv = torch.tensor([1]).to(self.theta1.device)
        D = torch.diag(torch.concat([sv, 1 / sv]))
        if map_type is MapType.SKEW:
            R1 = _rotation_from_angle(self.theta1)
        else:
            R1 = torch.eye(2).to(R2.device)

        return torch.matmul(self.init_rotation, torch.matmul(R2, torch.matmul(D, R1)))
