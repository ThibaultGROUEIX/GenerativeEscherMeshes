import igl
import numpy as np
import PIL
import torch
import escher.rendering.renderer_nvdiffrast as renderer_nvdiffrast
import nvdiffrast.torch as dr


OUTPUT_DIR = ""
glctx = dr.RasterizeCudaContext()


def save_img(img, path):
    # receives a tensor of shape (1, H, W, C) and saves it to path
    # save image without alpha channel using PIL
    img = img.cpu().detach().squeeze().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    PIL.Image.fromarray(img).save(path)


def render_from_path(obj_path, material_path, center_and_normalize=True):
    mv = torch.eye(4)[None, ...].cuda()
    proj = torch.eye(4)[None, ...].cuda()

    V, UV, _, T, _, _ = igl.read_obj(obj_path)
    texture = np.asarray(PIL.Image.open(material_path)) / 255.0

    V = torch.Tensor(V).cuda().unsqueeze(0).to(torch.float32)
    if center_and_normalize:
        V = V - V.mean(1)
        V = V / V.abs().max()
        V = V / 1.1
        # set depth to 1
        V[:, :, 2] = 1

    UV = torch.Tensor(UV).cuda().to(torch.float32)
    T = torch.Tensor(T).cuda().to(torch.float32)
    texture = torch.Tensor(texture).cuda().unsqueeze(0).to(torch.float32)
    img, _ = renderer_nvdiffrast.render_mesh_nvdiffrast(
        vertices=V,
        faces=T,
        uv=UV,
        mv=mv,
        proj=proj,
        image_size=(512, 512),
        glctx=glctx,
        texture=texture,
        vertices_color=None,
    )  #
    # save image with alpha channel using PIL
    mask = img[:, :, :, -1].unsqueeze(-1)
    img = img * mask  # Set black background
    save_img(img[:, :, :, :3], obj_path.replace(".obj", ".png"))
    save_img(mask, obj_path.replace(".obj", "_mask.png"))
    return img
    # return np.asarray(PIL.Image.open("temp_tiling.png")) / 255


if __name__ == "__main__":
    img = render_from_path("output/final_mesh.obj", "output/material_0.png")
    # look at temp_warp.png - the warped image, and then temp_unwarped.png is the unwarpped image
