import os
import pickle
import torch
import numpy as np
from escher.rendering.render_tiling_core import render_tiling
from escher.geometry.get_base_mesh import get_2d_square_mesh


# Recall strucutre of pkl of results.pkl.
#  {
#                             "V": mapped_org.detach().cpu().numpy(),
#                             "T": self.faces.cpu().numpy(),
#                             "UV": self.uv[0, ...].detach().cpu().numpy(),
#                             "texture": texture_img.cpu().detach().numpy(),
#                             "bdry": self.bdry,
#                             "constraint_data": self.constraint_data,
#                             "R": R.numpy(),
#                         },:


def render_from_pkl(
    fname,
    color_strategy="RANDOM",
    grid_sizes=[4],
    num_labels=1,
    make_video=False,
    highlight_single_tile=False,
    make_infinite_video=False,
):
    # colors = False : black and white, with highlights of the basic tile
    # colors = True : colors, random colors,
    with open(fname, "rb") as f:
        data = pickle.load(f)
    V = torch.from_numpy(data["V"])
    T = torch.from_numpy(data["T"])
    UV = torch.from_numpy(data["UV"])
    faces_split = None
    try:
        faces_split = torch.from_numpy(data["faces_split"])
    except:
        try:
            faces_split = [torch.from_numpy(data["faces_split"][i]) for i in range(len(data["faces_split"]))]
        except:
            _, _, faces_split, _ = get_2d_square_mesh(int(np.sqrt(V.shape[0])), num_labels=num_labels)

    texture = torch.from_numpy(data["texture"])
    if texture.shape[0] == 3:
        texture = texture.permute(1, 2, 0).contiguous()
    bdry = data["bdry"]
    constraint_data = data["constraint_data"]
    constraint_data.update_scaling()
    # if isinstance(colors, list) or isinstance(colors, torch.Tensor):
    #     colors = colors[: constraint_data.tiling_coloring_number(), :]
    #     colors = get_color_function(colors / 255.0)

    A = torch.from_numpy(data["R"])
    output = fname.replace(".pkl", "tile.png")

    render_tiling(
        V,
        UV,
        T,
        texture,
        bdry,
        constraint_data,
        A,
        output,
        grid_sizes=grid_sizes,
        color_strategy=color_strategy,
        make_video=make_video,
        num_labels=num_labels,
        faces_split=faces_split,
        highlight_single_tile=highlight_single_tile,
        make_infinite_video=make_infinite_video,
    )


def convert_all_subfolders(path, **kwargs):
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            convert_all_subfolders(os.path.join(path, file), **kwargs)
        elif file.endswith(".pkl") and not file.endswith("old.pkl"):
            print(os.path.join(path, file))
            try:
                render_from_pkl(os.path.join(path, file), **kwargs)
            except:
                print(f"failed to convert{os.path.join(path, file)}")


def sanity_checks():
    render_from_pkl(
        "output/results.pkl",
        color_strategy="PERIODIC",
        num_labels=1,
        make_video=False,
        grid_sizes=[5],
        highlight_single_tile=True,
        make_infinite_video=False,
    )


if __name__ == "__main__":
    # sanity_checks()
    # os.exit(0)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        help="path to pkl to render",
        default="/sensei-fs/users/groueix/projects/escher/tog_xp/./grid_mesh6/_prompt=acartoonillustrationofanaligatoramasterpieceacartoonillustrationofacatamasterpieceacartoonillustrationofaelephantamasterpieceacartoonillustrationofamonkeyamasterpiece_area=100000_tile=OrbifoldII",
    )
    parser.add_argument("--num_labels", type=int, default=1, help="number of labels")
    parser.add_argument("--make_video", action="store_true", help="save make_video where tiles appear one by one")
    parser.add_argument("--grid_sizes", type=int, nargs='+', default=[7], help="grid_size of the render")
    parser.add_argument("--make_infinite_video", action="store_true", help="make_infinite_video")
    args = parser.parse_args()
    print(args)
    # colors = torch.Tensor(
    #     [[25, 88, 181], [176, 39, 34], [44, 218, 157], [245, 26, 164], [255, 209, 102], [198, 212, 255]]
    # )

    if os.path.isfile(args.path):
        render_from_pkl(
            args.path,
            color_strategy="RANDOM",
            num_labels=args.num_labels,
            make_video=False,
            highlight_single_tile=True,
            grid_sizes=args.grid_sizes,
            make_infinite_video=args.make_infinite_video,
        )
    # check if path is a folder
    else:
        convert_all_subfolders(
            args.path,
            color_strategy="RANDOM",
            num_labels=args.num_labels,
            make_video=False,
            highlight_single_tile=True,
            grid_sizes=args.grid_sizes,
            make_infinite_video=args.make_infinite_video,
        )
