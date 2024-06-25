"""
This function is specific to the case where we have multiple shapes within a single tile 
i.e. if the user provides multiple prompts for a single tile.
We observe that without regularization, the model tends to shrink some sub-part of the tile a little bit to much.
Having each sub-part of the tile have the same area is a good way to prevent this, and produces the most visually pleasing results.
"""
import torch
from torch.nn import functional as F


class EqualAreaLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vals = []

    @staticmethod
    def compute_area(V, faces):
        """Compute the area of a mesh by summing the areas of its faces.

        Args:
            V torch.Tensor: #vertex, 2
            faces (_type_): #faces, 3
        """
        a = V[faces[:, 0]]
        b = V[faces[:, 1]]
        c = V[faces[:, 2]]
        # make 3D vectors
        a = F.pad(a, (0, 1))
        b = F.pad(b, (0, 1))
        c = F.pad(c, (0, 1))
        # Cross product to get area of triangle
        out = torch.cross((b - a), (c - a))
        # Sum over all faces, and divide by 2
        area = torch.sum(torch.norm(out, dim=1) / 2)
        return area

    def equal_area_loss(self, V, faces_split):
        num_parts = len(faces_split)
        areas = [EqualAreaLoss.compute_area(V, faces_split[i]) for i in range(num_parts)]
        areas = torch.hstack(areas)
        mean_area = areas.sum() / num_parts
        loss = torch.sum((areas - mean_area.detach()) ** 2)
        self.vals.append(areas.detach().cpu().numpy())
        return loss

    def save_curves(self, path):
        # save area curves to disk
        import matplotlib.pyplot as plt

        if len(self.vals) == 0:
            return
        plt.plot(self.vals)
        plt.savefig(path)


if __name__ == "__main__":
    pass
