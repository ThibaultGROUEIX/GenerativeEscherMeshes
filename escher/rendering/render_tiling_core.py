import os
from pathlib import Path
import igl
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision
from os.path import join
from escher.rendering.renderer_nvdiffrast import render_mesh_nvdiffrast
from escher.rendering.get_colors import ColorIndex
import escher.geometry.split_square_boundary as split_square_boundary
from escher.geometry.sanity_checks import check_triangle_orientation

import nvdiffrast.torch as dr
from escher.OTE.core.OTESolver import Constraints
import imageio
from typing import List
OUTPUT_DIR = ""
glctx = dr.RasterizeCudaContext()
from time import time


def _render_tiling_same_color(V, UV, T, texture, constraints: Constraints, R, grid_size=4):
    maps = constraints.get_tiling(half_grid_size=grid_size)
    rV = []
    rT = []
    rUV = []
    for i, map in enumerate(maps):
        m = map.map(V[:, 0:2])
        m = torch.matmul(m.float(), torch.transpose(R, 0, 1))
        newT = T + i * V.shape[0]

        rV.append(m)
        rT.append(newT)
        rUV.append(UV)
    rV = torch.concat(rV, axis=0)
    rT = torch.concat(rT, axis=0)
    rUV = torch.concat(rUV, axis=0)
    img, _ = render_mesh_nvdiffrast(
        vertices=rV, uv=rUV, faces=rT, texture=texture, image_size=1024 if grid_size <= 4 else 2048, glctx=glctx
    )
    return img


def infinite_video(image: torch.Tensor, direction: torch.Tensor, fnames: List[str]) -> None:
    # image_imageio = (image.cpu().squeeze()[:,:,:3].contiguous()*255).int().contiguous().numpy().astype(np.uint8)
    """ The fonction creates an infinite video loop from a single image and the direction of the tiling"""
    
    image_tv = image.cpu().squeeze()[:, :, :3].contiguous().permute(2, 0, 1).contiguous()

    torchvision.utils.save_image(image_tv, "infinite_image.png")
    torch.save(direction, "infinite_direction.pt")

    temp_dir = Path(fnames[0]).parent / "intermediate"
    temp_dir = temp_dir.as_posix()
    os.makedirs(temp_dir, exist_ok=True)

    direction = direction
    # We can further divide the direction by 2 because the view frustrum of nvdiffrast is [-1,1]
    n_step = 180 # number of frames in the infinite loop
    if direction[0] < 0 and direction[1] < 0:
        direction = -direction

    elif direction[0] < 0 and direction[1] > 0:
        direction = -direction

    infinite_loop_size = (1 - direction.abs()).min()
    bottom_left = torch.tensor([0, 0]).cuda()
    if direction[1] < 0:
        bottom_left = torch.tensor([0, direction[1].abs()]).cuda()

    # flip image on the Y axis
    # image = image.flip(2)
    for i in range(n_step):
        w = image.shape[1]  # assume image is square
        i_start = bottom_left + i * direction / n_step
        i_end = i_start + infinite_loop_size

        i_start = torch.round(i_start * w).int().clip(0, w - 1)
        i_end = torch.round(i_end * w).int().clip(0, w - 1)
        # start = time()
        torchvision.utils.save_image(image_tv[:, i_start[0] : i_end[0], i_start[1] : i_end[1]], f"infinite_{str(i).zfill(4)}.png")
        # print(f"[Time to save image TORCHVISION {i}]: {time() - start}")
        # start = time()
        # # save with imageio
        # image_to_save = image_imageio[i_start[0]:i_end[0], i_start[1]:i_end[1]]
        # imageio.imwrite(join(temp_dir, f"infinite_{str(i).zfill(4)}.png"), image_to_save)
        # print(f"[Time to save image IMAGEIO {i}]: {time() - start}")

    # os.system(f"ffmpeg -framerate 30  -i {temp_dir}/infinite_%04d.png -y {fnames[0]}")
    os.system(f"ffmpeg -framerate 60 -i infinite_%04d.png -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -y {fnames[0]}")

    for fname in fnames:
        os.system(f"cp {fnames[0]} {fname}")
    # os.system(f"rm {temp_dir}/infinite*.png")


def debug_code_for_infinite_video():
    vec_1, vec_2 = constraints.get_torus_directions_with_scaling()
    vec_1 = torch.from_numpy(vec_1).cuda().float()
    vec_2 = torch.from_numpy(vec_2).cuda().float()

    vec_1 = torch.matmul(vec_1, torch.transpose(R, 0, 1))
    vec_2 = torch.matmul(vec_1, torch.transpose(R, 0, 1))

    all_directions = torch.vstack((vec_1, vec_2, vec_1 + vec_2, vec_1 - vec_2))
    all_directions_norm = torch.vstack((vec_1.norm(), vec_2.norm(), (vec_1 + vec_2).norm(), (vec_1 - vec_2).norm()))
    direction = all_directions[all_directions_norm.argmin()]

    direction = direction / grid_size
    direction_compare = torch.hstack([-direction[1], direction[0]])
    direction_compare = torch.round(direction_compare * (img.size(1)))

    epsilon = 0.005
    vertices_direction = torch.vstack(
        (torch.Tensor([0, 0]).cuda(), torch.Tensor([0, 0]).cuda() + epsilon, direction, direction + epsilon)
    )
    img, _ = render_mesh_nvdiffrast(
        vertices=vertices_direction,
        faces=torch.LongTensor([[0, 1, 2], [1, 2, 3]]).cuda(),
        uv=None,
        texture=None,
        vertices_color=torch.Tensor([[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]]).cuda(),
        image_size=1024 if grid_size <= 4 else 2048,
        glctx=glctx,
    )
    mask = img[:, :, :, 3:]
    img_color = img[:, :, :, :3]
    nz = (img_color == torch.max(img_color)).nonzero()
    max_point_idx = (nz[:, 1] == torch.max(nz[:, 1])).nonzero()[0]
    min_point_idx = (nz[:, 1] == torch.min(nz[:, 1])).nonzero()[0]
    min_point = nz[min_point_idx]
    max_point = nz[max_point_idx]
    direction_after = max_point - min_point

    print(img_color[nz[0][0], nz[0][1], nz[0][2], :])
    total_img = total_img * (1 - mask) + mask * img
    total_mask = total_img + mask


def _render_tiling_different_colors(
    V,
    UV,
    T,
    texture,
    constraints: Constraints,
    R,
    color_strategy="RANDOM",
    num_labels=1,
    texture_already_has_color=False,
    video_names=None,
    faces_split=None,
    grid_size=4,
    verbose=False,
):
    check_triangle_orientation(V, T)
    color_getter = ColorIndex(
        n_orientation_index=constraints.get_num_orientation(V, constraints.sides),
        n_sub_tile_index=len(faces_split),
        color_strategy=color_strategy,
    )

    maximum = 3000
    limit_black_pixels = 100

    # Convert T to torch if it's a numpy array, same for faces_split
    if isinstance(T, np.ndarray):
        T = torch.from_numpy(T)

    if faces_split is None:
        faces_split = [T]

    if isinstance(faces_split[0], np.ndarray):
        faces_split = [torch.from_numpy(faces) for faces in faces_split]

    with torch.no_grad():
        start = time()
        
        grid_size_base = 2
        maps = constraints.get_tiling(half_grid_size=grid_size_base, vertices=V, sides=constraints.sides)
        step = 0  # this will store a counter for every map that's visible
        set_first_to_zero = True  # this will be used to set the first frame of the video to black
        if verbose:
            print(f"[Time to get tiling maps]: {time() - start}")
        i_map = -1
        # for i_map, map in enumerate(maps):
        while True and i_map < maximum:
            i_map += 1
            try:
                map = maps[i_map]
            except:
                start2 = time()
                new_maps = constraints.get_more_tiling(
                    half_grid_size=2 * grid_size_base, previous_half_grid_size=grid_size_base, vertices=V, sides=constraints.sides
                )
                grid_size_base *= 2
                maps.extend(new_maps)
                if verbose:
                    print(f"[Time to get{len(new_maps)} more tiling maps]: {time() - start2}")
                map = maps[i_map]

            m = map.map(V[:, 0:2])
            m = torch.matmul(m.float(), torch.transpose(R, 0, 1))
            m = m / grid_size
            start = time()
            for index_face_split, face_split in enumerate(faces_split):
                c_texture = texture
                if not texture_already_has_color:
                    inv_texture = 1 - texture
                    color = color_getter.get_color(map.orientation_index, sub_tile_index=index_face_split)
                    color = color.cuda()
                    # c_texture = inv_texture * color
                    inv_color = 1 - color
                    c_texture = inv_texture * inv_color
                    c_texture = 1 - c_texture

                img, _ = render_mesh_nvdiffrast(
                    vertices=m,
                    faces=face_split,
                    uv=UV,
                    texture=c_texture,
                    image_size=1024 if grid_size <= 4 else 2048,
                    glctx=glctx,
                )
                mask = img[:, :, :, 3:]
                if set_first_to_zero:
                    total_img = img * 0
                    total_mask = mask * 0
                    set_first_to_zero = False

                if not mask.max() > 0:
                    # not need to save this map, it's not visible
                    continue

                # Raise an error if the intersection of the mask and the total mask is not empty
                # if ((mask * total_mask)>0).sum() / ((mask)>0).sum() > 0.05:
                if ((mask >= 1) * (total_mask >= 1)).sum():
                    from kornia.morphology import erosion

                    intersection = ((mask >= 1) * (total_mask >= 1)).float()
                    kernel = torch.ones(2, 2).cuda()
                    output = erosion(intersection, kernel)
                    if output.sum():
                        print(f"step {step} imap {i_map} face {index_face_split}")
                        print(
                            "You are trying to overwrite pixels that were already written, this is very conspicuous and should not happen for a perfect tiling up to aliasing effect at the boundary. Please check your code."
                        )
                        print("the overlap percentage is ", ((mask * total_mask) > 0).sum() / ((mask) > 0).sum())
                        torchvision.utils.save_image(
                            ((mask >= 1) * (total_mask >= 1)).float().permute(0, 3, 1, 2), "debug_mask_intersection.png"
                        )
                        torchvision.utils.save_image((mask).permute(0, 3, 1, 2), "debug_mask.png")
                        torchvision.utils.save_image((total_mask).permute(0, 3, 1, 2), "debug_total_mask.png")
                        torchvision.utils.save_image((output).permute(0, 3, 1, 2), "debug_total_mask_eroded.png")
                        print("please check the mask_intersection.png, mask.png, and mask.png images")
                        # raise ValueError("You are trying to overwrite pixels that were already written, this is very conspicuous and should not happen for a perfect tiling up to aliasing effect at the boundary. Please check your code.")

                total_img = total_img * (1 - mask) + mask * img
                total_mask = total_mask + mask
                # save the image
                if video_names is not None and step < 100:
                    # save the image
                    parent_dir = Path(video_names[0]).parent / "intermediate"
                    os.makedirs(parent_dir.as_posix(), exist_ok=True)
                    path = parent_dir / f"./{str(step).zfill(4)}_temp_tile.png"
                    if step == 0:
                        # save first frame as black
                        torchvision.utils.save_image((img * 0).permute(0, 3, 1, 2), path.as_posix())
                        step += 1
                        path = parent_dir / f"./{str(step).zfill(4)}_temp_tile.png"
                    torchvision.utils.save_image(total_img.permute(0, 3, 1, 2), path.as_posix())
                    step += 1

            if (total_mask < 0.1).float().sum() < 100:
                print("Reached the limit of black pixels, this is good enough")
                break
            if verbose:
                print(f"[Time to render map] {i_map}: {time() - start}")

        if i_map == maximum:
            print("Reached the limit of maps")
        # debug_code_for_infinite_video()

        # Make Gifs
        if video_names is not None:
            parent_dir = Path(video_names[0]).parent / "intermediate"
            parent_dir = parent_dir.as_posix()
            for i in range(step, 100):
                # copy last frame to make sure each gif has the same number of frames
                os.system(
                    f"cp {parent_dir}/{str(i-1).zfill(4)}_temp_tile.png {parent_dir}/{str(i).zfill(4)}_temp_tile.png"
                )
            os.system(
                f"ffmpeg -framerate 5  -i {parent_dir}/%04d_temp_tile.png -y {video_names[0].replace('.gif', '.mp4')}"
            )
            # os.system(f"ffmpeg -framerate 5  -i {parent_dir}/%04d_temp_tile.png -y {video_names[0]}")
            # os.system(f"rm {parent_dir}/*temp_tile.png")

            for video_name in video_names:
                # copy the first gif to the other names
                os.system(f"cp {video_names[0].replace('.gif', '.mp4')} {video_name.replace('.gif', '.mp4')}")

        return total_img


def render_tiling(
    V,
    UV,
    T,
    texture,
    bdry,
    constraints: Constraints,
    R,
    fnames,
    grid_sizes=4,
    color_strategy="RANDOM",  # Accepts RANDOM, BW, ALREADY_HAS_COLOR, or a function
    make_video=False,  # Creates a gif of the tiling
    num_labels=1,  # Number of sub-tiles in the tile
    faces_split=None,  # If you have a multitile, you need to provide a list of faces for each sub-tile
    highlight_single_tile=False,  # If you want to highlight a single central tile in the tiling
    make_infinite_video=False,  # If you want to make a gif that loops infinitely
):
    with torch.no_grad():
        R = R.cuda()
        V = V.cuda()
        T = T.cuda()
        UV = UV.cuda()
        texture = texture.cuda()
        if not isinstance(fnames, list):
            fnames = [fnames]

        for grid_size in grid_sizes:
            fnames_grid = [fname.replace(".png", f"_{grid_size}.png") for fname in fnames]

            video_names = None
            if make_video:
                video_names = [fname.replace(".png", ".gif") for fname in fnames_grid]

            if not color_strategy == "BW":
                img = _render_tiling_different_colors(
                    V,
                    UV,
                    T,
                    texture,
                    constraints,
                    R,
                    color_strategy=color_strategy,
                    texture_already_has_color=color_strategy == "ALREADY_HAS_COLOR",
                    num_labels=num_labels,
                    video_names=video_names,
                    faces_split=faces_split,
                    grid_size=grid_size,
                )
            elif color_strategy == "BW":
                img = _render_tiling_same_color(V, UV, T, texture, constraints, R, grid_size)
            else:
                raise NotImplementedError(f"color strategy {color_strategy} not implemented")
            [torchvision.utils.save_image(img.permute(0, 3, 1, 2), fname) for fname in fnames_grid]

            if make_infinite_video:
                vec_1, vec_2 = constraints.get_torus_directions_with_scaling()
                vec_1 = torch.from_numpy(vec_1).cuda().float()
                vec_2 = torch.from_numpy(vec_2).cuda().float()

                vec_1 = torch.matmul(vec_1, torch.transpose(R, 0, 1))
                vec_2 = torch.matmul(vec_2, torch.transpose(R, 0, 1))

                all_directions = torch.vstack((vec_1, vec_2, vec_1 + vec_2, vec_1 - vec_2))
                all_directions_norm = torch.vstack(
                    (vec_1.norm(), vec_2.norm(), (vec_1 + vec_2).norm(), (vec_1 - vec_2).norm())
                )
                direction = all_directions[all_directions_norm.argmin()]
                direction = direction / grid_size / 2

                direction = torch.hstack([-direction[1], direction[0]])

                infinite_video(
                    img, direction=direction, fnames=[fname.replace(".png", "infinite.mp4") for fname in fnames]
                )

            if highlight_single_tile:
                start = time()
                img = img[:, :, :, :3]
                img = img.detach().cpu()
                img_coords = V.detach().cpu().numpy()
                tile_map = constraints.get_tiling(half_grid_size=2, vertices=V, sides=constraints.sides)
                tile_map = tile_map[0]  # select first map
                img_coords[:, :2] = np.matmul(
                    tile_map.map(img_coords[:, :2]), torch.transpose(R, 0, 1).cpu().detach().numpy()
                )
                img_coords[:, 1] = -img_coords[:, 1]
                img_coords = (img_coords / grid_size + 1) * img.shape[1] / 2
                closed_bdry = np.concatenate((bdry, bdry[:1]), axis=0)
                plt.figure(frameon=False)

                if str(constraints.__class__) == "<class 'escher.OTE.tilings.OrbifoldII_hexagon.OrbifoldIIHexagonConstraints'>":
                    colors = ["green", "blue", "blue", "magenta", "magenta", "green"]
                    for i in range(6):
                        plt.plot(
                            img_coords[constraints.sides[i], 0],
                            img_coords[constraints.sides[i], 1],
                            linewidth=2,
                            color=colors[i],
                        )

                    for i in range(6):
                        plt.plot(
                            img_coords[constraints.sides[i][:1], 0],
                            img_coords[constraints.sides[i][:1], 1],
                            "ko",
                            markersize=5,
                        )
                elif str(constraints.__class__) == "<class 'escher.OTE.tilings.OrbifoldII.OrbifoldIIConstraints'>":
                    plt.plot(
                        img_coords[constraints.sides["left"], 0],
                        img_coords[constraints.sides["left"], 1],
                        linewidth=2,
                        color="green",
                    )
                    plt.plot(
                        img_coords[constraints.sides["right"], 0],
                        img_coords[constraints.sides["right"], 1],
                        linewidth=2,
                        color="blue",
                    )
                    plt.plot(
                        img_coords[constraints.sides["top"], 0],
                        img_coords[constraints.sides["top"], 1],
                        linewidth=2,
                        color="green",
                    )
                    plt.plot(
                        img_coords[constraints.sides["bottom"], 0],
                        img_coords[constraints.sides["bottom"], 1],
                        linewidth=2,
                        color="blue",
                    )
                    plt.plot(
                        img_coords[constraints.sides["bottom"][:1], 0],
                        img_coords[constraints.sides["bottom"][:1], 1],
                        "ko",
                        markersize=5,
                    )
                    plt.plot(
                        img_coords[constraints.sides["top"][:1], 0],
                        img_coords[constraints.sides["top"][:1], 1],
                        "ko",
                        markersize=5,
                    )
                    plt.plot(
                        img_coords[constraints.sides["left"][:1], 0],
                        img_coords[constraints.sides["left"][:1], 1],
                        "ko",
                        markersize=5,
                    )
                    plt.plot(
                        img_coords[constraints.sides["right"][:1], 0],
                        img_coords[constraints.sides["right"][:1], 1],
                        "ko",
                        markersize=5,
                    )
                else:
                    plt.plot(
                        img_coords[closed_bdry, 0],
                        img_coords[closed_bdry, 1],
                        dashes=[5, 5],
                        linewidth=1,
                        color=np.array([0, 162, 255]) / 255,
                        gapcolor=np.array([255, 56, 238]) / 255,
                    )
                plt.imshow(img[0, ...])
                plt.show()
                plt.axis("off")
                plt.margins(x=0, y=0)
                # plt.savefig("dummy_name.eps", bbox_inches='tight')
                dpi = 120

                [
                    plt.savefig(fname.replace(".png", "highlight.png"), dpi=dpi, bbox_inches="tight", pad_inches=0)
                    for fname in fnames_grid
                ]
                [
                    plt.savefig(fname.replace(".png", "highlight.pdf"), dpi=dpi, bbox_inches="tight", pad_inches=0)
                    for fname in fnames_grid
                ]
                plt.close("all")
                print(f"[Time to highlight tile]: {time() - start}")


if __name__ == "__main__":
    from escher.OTE.core.OTESolver import OrbifoldI, TorusConstraints

    V, UV, _, T, _, _ = igl.read_obj("for_noam_test/mesh.obj")
    texture = np.asarray(PIL.Image.open("for_noam_test/cat.png")) / 255
    bdry = igl.boundary_loop(T)

    # split the bdry into 4 sides (left,right,top,down)
    square_sides = split_square_boundary.split_square_boundary(UV, bdry)
    constraints = TorusConstraints(V[:, 0:2], square_sides)
    _render_tiling_different_colors(
        torch.Tensor(V).cuda(),
        torch.Tensor(UV).cuda(),
        torch.Tensor(T).cuda(),
        torch.Tensor(texture),
        bdry,
        constraints,
        constraints.get_horizontal_symmetry_orientation().cuda().float(),
        "testing_render_tiling.jpg",
        True,
    )
