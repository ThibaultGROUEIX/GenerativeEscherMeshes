import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from rich import print
from rich.traceback import install
from tqdm import tqdm
import torchvision

# sys.path.append(os.path.join(os.path.dirname(__file__), "stable-dreamfusion", "nerf"))
import escher.guidance.sd_2 as sd
import escher.guidance.deepfloyd as dp

install(show_locals=False)
torch.autograd.set_detect_anomaly(True)
import math
from pathlib import Path

from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import RandomCrop


def save_loss_curve(loss_curve: np.ndarray, filename: str):
    plt.clf()
    plt.title("Loss")
    plt.plot(loss_curve)
    plt.savefig(filename)


def save_timestep_vs_loss(timestep_curve: np.ndarray, loss_curve: np.ndarray, filename: str):
    plt.clf()
    plt.title("Timestep vs. Loss")
    plt.scatter(timestep_curve, loss_curve)
    plt.xlabel("timestep")
    plt.ylabel("loss")
    plt.savefig(filename)


def create_video_with_images(glob_pattern: str, output_filename: str):
    """
    Create a video from a set of images using ffmpeg.
    Images are gathered using a glob pattern (e.g. "img_????.png").
    """
    os.system(f"ffmpeg -framerate 10 -pattern_type glob -i '{glob_pattern}' {output_filename}")
    # os.system(f"rm {glob_pattern}")


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, -1)


def optimize_img(
    prompt: str,
    sd_model: sd.StableDiffusion,
    num_steps: int,
    save_interval: int = 10,
    number_of_repeated_tiles: int = 1,
    image_paths: list = None,
) -> torch.Tensor:
    prompt = [prompt]
    prompt_embedding = sd_model.get_text_embeds(prompt)
    negative_embedding = sd_model.get_text_embeds("")
    text_embeds = torch.cat([prompt_embedding, negative_embedding], dim=0)

    init_image = torch.rand(1, 512, 512, 3, device=device)
    # init_image = init_image*0
    # init_image[:,:,50:100, 50:100] = 1

    img_parameters = torch.nn.Parameter(init_image)
    optimizer = torch.optim.Adam([img_parameters], lr=2 * 1e-1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(num_steps * 1.5))

    loss_curve = []
    timestep_curve = []
    # create output folder .baseline
    os.makedirs(".baseline", exist_ok=True)
    folder_name = Path(".baseline")
    folder_name = folder_name / str(number_of_repeated_tiles) / sd_model.__class__.__name__
    os.makedirs(folder_name, exist_ok=True)
    folder_name = folder_name / prompt[0].replace(" ", "")
    os.makedirs(folder_name, exist_ok=True)

    transform_crop = RandomCrop((512, 512))

    for i in tqdm(range(num_steps)):
        if i % save_interval == 0:
            visu = True
        optimizer.zero_grad()
        tiled_image = img_parameters.repeat(1, number_of_repeated_tiles, number_of_repeated_tiles, 1).contiguous()

        loss, timestep = sd_model.train_step(
            transform_crop(tiled_image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1), text_embeds
        )

        loss_curve.append(loss.item())
        timestep_curve.append(timestep.item())

        img_id = str(i).zfill(4)

        if i % save_interval == 0:
            print(f"step {i}: {loss.item()}")
            # convert image from torch tensor to numpy array
            img = tiled_image.detach().cpu().numpy()[0, :, :, :]
            PIL.Image.fromarray((img * 255).astype(np.uint8)).save(f"{folder_name.as_posix()}/tile_{img_id}.png")
            PIL.Image.fromarray((img_parameters.detach().cpu().numpy()[0, :, :, :] * 255).astype(np.uint8)).save(
                f"{folder_name.as_posix()}/img_{img_id}.png"
            )
        loss.backward()
        optimizer.step()
        scheduler.step()
        img_parameters.data = img_parameters.data.clip(-1, 1)
    if image_paths is not None:
        image_paths.append(f"{folder_name.as_posix()}/tile_4900.png")

    # save_loss_curve(loss_curve, f"{folder_name.as_posix()}/loss_curve.png")
    # save_timestep_vs_loss(timestep_curve, loss_curve, f"{folder_name.as_posix()}/timestep_vs_loss.png")
    # create_video_with_images(f"{folder_name.as_posix()}/img_????.png", f"image_{prompt[0].replace(' ', '_')}.mp4")
    return img_parameters.detach()


def run_comparison(opt):
    image_paths = []
    for ntiles in [3, 5, 7]:
        optimize_img("a children's book illustration of a nerdy bear", model, opt.num_iter, 100, ntiles, image_paths)
        optimize_img("a penguin", model, opt.num_iter, 100, ntiles, image_paths)
        optimize_img(
            "a comic book illustration of Saturn with shooting stars", model, opt.num_iter, 100, ntiles, image_paths
        )
        optimize_img("a lion, best quality, extremely detailed", model, opt.num_iter, 100, ntiles, image_paths)
        optimize_img("a children's book illustration of a bat", model, opt.num_iter, 100, ntiles, image_paths)
        optimize_img("a nerdy bear", model, opt.num_iter, 100, ntiles, image_paths)
        optimize_img("saturn with shooting stars", model, opt.num_iter, 100, ntiles, image_paths)
        optimize_img("a lion", model, opt.num_iter, 100, ntiles, image_paths)
        optimize_img("a bat", model, opt.num_iter, 100, ntiles, image_paths)

    import netvision

    os.makedirs("html_tiled_image", exist_ok=True)
    webpage = netvision.HtmlGenerator(path="html_tiled_image/index.html", local_copy=True)
    # Make a 1st table
    table1 = webpage.add_table("My awesome table")
    for col in range(10):
        table1.add_column(str(col))
    to_add_img = []
    to_add_path = []
    for i, img in enumerate(image_paths):
        to_add_path = to_add_path + [img]
        to_add_img = to_add_img + [webpage.image(img)]
        if len(to_add_img) == 5:
            table1.add_row(to_add_img + to_add_path)
            to_add_img = []
            to_add_path = []
    webpage.return_html()


if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt

    os.makedirs(".results", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-prompt", type=str, default="an old smoking pipe.")
    parser.add_argument("--num_iter", type=int, default=0)
    parser.add_argument("--number_of_repeated_tiles", type=int, default=3)
    parser.add_argument("--model", type=str, default="sd", choices=["sd", "dp"])
    parser.add_argument("--run_comparison", action="store_true")
    opt = parser.parse_args()

    device = torch.device("cuda")
    model = sd.StableDiffusion(sd.Config()) if opt.model == "sd" else dp.DeepFloydGuidance()

    if opt.run_comparison:
        run_comparison(opt)
    else:
        final_image = optimize_img(
            opt.prompt, model, opt.num_iter, number_of_repeated_tiles=opt.number_of_repeated_tiles
        )
