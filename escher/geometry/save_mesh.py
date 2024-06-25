from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
import trimesh


def save_mesh(fname, vertices, triangles, uvs, texture_image=None):
    # texture_image should be None or a tensor of shape (H, W, 3), between 1 and 1

    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    # Create a texture for the UV coordinates
    texture = trimesh.visual.texture.TextureVisuals(uv=uvs)

    # Assign the texture to the mesh
    mesh.visual = texture

    # Save the mesh to an OBJ file
    mesh.export(fname, file_type="obj")

    # Save the texture image
    # permute texture image using torch.einsum
    if texture_image is not None:
        # save to material_0.png, replace name of obj by material_0.png
        save_path = Path(fname).parent / "material_0.png"
        texture_img = texture_image.detach().cpu().squeeze()
        if texture_img.shape[2] < 5 and texture_img.shape[0] > 5:
            texture_img = torch.einsum("hwc->chw", texture_img)

        # flip y axis
        texture_img = texture_img.flip([1])
        torchvision.utils.save_image(texture_img, save_path.as_posix())
