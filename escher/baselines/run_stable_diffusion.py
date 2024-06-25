import sys
import os
from os.path import join
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


def run_prompt(prompt, directory, list_of_prompts=None):
    if list_of_prompts is not None:
        list_of_prompts.append(prompt)
    # image = pipe(prompt).images[0]
    os.makedirs(directory, exist_ok=True)
    # image.save(join(directory, prompt.replace(" ", "_") + ".png"))


def run_prompt_with_augmentations(prompt, image_paths=None):
    list_of_prompts = []

    directory = "./.direct/" + prompt.replace(" ", "_")
    run_prompt(prompt, directory, list_of_prompts)
    prefixes = [
        "An Escher drawing of ",
        "An Escher tiling of ",
        "An Escher pattern of ",
        "An Escherization of ",
        "An Escher-like drawing of ",
        "An Escher-like tiling of ",
        "An Escher-like pattern of ",
        "An Escher-like image of ",
        "An Escher-like picture of ",
    ]
    for prefix in prefixes:
        run_prompt(prefix + prompt, directory, list_of_prompts)

    suffixes = [
        " in the style of Escher",
    ]
    for suffix in suffixes:
        run_prompt(prompt + suffix, directory, list_of_prompts)

    run_prompt("An image composed of the repetition of " + prompt, directory, list_of_prompts)
    run_prompt(
        "An image composed of the repetition of " + prompt + "just like Escher drawings in the metamorphosis series.",
        directory,
        list_of_prompts,
    )
    run_prompt(
        "An image composed of the repetition of " + prompt + "just like Escher drawings.", directory, list_of_prompts
    )
    run_prompt(
        "An image composed of the repetition of " + prompt + " in the style of Escher", directory, list_of_prompts
    )
    run_prompt(
        "An image composed of the repetition of "
        + prompt
        + " in the style of Escher. There should be no gaps nor overlaps.",
        directory,
        list_of_prompts,
    )
    run_prompt(
        "An image composed of the repetition of " + prompt + ". There should be no gaps nor overlaps.",
        directory,
        list_of_prompts,
    )
    run_prompt(
        "An image composed of the repetition of "
        + prompt
        + ". There should be no gaps nor overlaps. The object is always the same in all repetitions",
        directory,
        list_of_prompts,
    )
    run_prompt(
        "An image made by repeating "
        + prompt
        + " with translations in such a way that there is no gaps nor overlaps. It is always the same object.",
        directory,
        list_of_prompts,
    )
    run_prompt(
        "An image made by repeating "
        + prompt
        + " with translations in such a way that there is no gaps nor overlaps. It is always the same object. The style is Escher-like.",
        directory,
        list_of_prompts,
    )
    with open(join(directory, "prompts.txt"), "w") as f:
        f.write("\n".join(list_of_prompts))

    if image_paths is not None:
        for prompt in list_of_prompts:
            image_paths[0].append(join(directory, prompt.replace(" ", "_") + ".png"))
            image_paths[1].append(prompt)


def run_direct_baseline():
    image_paths = [[], []]
    run_prompt_with_augmentations("a children's book illustration of a nerdy bear", image_paths)
    run_prompt_with_augmentations("a penguin", image_paths)
    run_prompt_with_augmentations("a comic book illustration of Saturn with shooting stars", image_paths)
    run_prompt_with_augmentations("a lion, best quality, extremely detailed", image_paths)
    run_prompt_with_augmentations("a children's book illustration of a bat", image_paths)
    run_prompt_with_augmentations("a nerdy bear", image_paths)
    run_prompt_with_augmentations("saturn with shooting stars", image_paths)
    run_prompt_with_augmentations("a lion", image_paths)
    run_prompt_with_augmentations("a bat", image_paths)

    import netvision

    os.makedirs("html_run_", exist_ok=True)
    webpage = netvision.HtmlGenerator(path="html_run_/index.html", local_copy=True)
    # Make a 1st table
    table1 = webpage.add_table("My awesome table")
    for col in range(10):
        table1.add_column(str(col))
    to_add_img = []
    to_add_path = []
    for i, img in enumerate(image_paths[0]):
        to_add_path = to_add_path + [image_paths[1][i]]
        to_add_img = to_add_img + [webpage.image(img)]
        if len(to_add_img) == 5:
            table1.add_row(to_add_img + to_add_path)
            to_add_img = []
            to_add_path = []
    webpage.return_html()


if __name__ == "__main__":
    # prompt = sys.argv[1]
    run_direct_baseline()
