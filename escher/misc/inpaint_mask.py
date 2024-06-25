import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from torchvision import transforms

from escher.guidance.sd import StableDiffusion, seed_everything

seed_everything(2346789)
import torch.nn.functional as F

MODES = {
    0: "inpaint with stable diffusion inpainting",
    1: "inpaint with latent 2 latent enhancement base stable diffusion model",
    2: "inpaint with latent 2 latent enhancement inpainting stable diffusion ",
}


def inpaint_mode_0(image, mask_image, prompt):
    with torch.no_grad():
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
        pipe.to("cuda")
    """PIL image as input
    Use stable diffusion inpainting model
    """
    return pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]


def inpaint_mode_1(image, prompt, recursive=1):
    """Use stable diffusion img2img base model."""
    with torch.no_grad():
        sd = StableDiffusion(torch.device("cuda:0"), True, False)

        prompt_embedding = sd.get_text_embeds(prompt)
        negative_embedding = sd.get_text_embeds("")
        text_embeds_single_batch = torch.cat([negative_embedding] + [prompt_embedding])
        del sd.text_encoder

        latents = sd.encode_imgs(image)

        for _ in range(recursive):
            latents = sd.latent_to_better_latent(
                text_embeddings=text_embeds_single_batch,
                latents=latents,
                start_step=28,
            )
        image = sd.decode_latents(latents)  # [1, 3, 512, 512]

        return transforms.ToPILImage()(image[0].cpu())


def inpaint_mode_2(image, image_mask, prompt, recursive=1):
    """Use stable diffusion img2img base model."""
    with torch.no_grad():
        from escher.guidance import sd_inpainting

        sd_inpaint = sd_inpainting.StableDiffusion(
            sd_version="2.1.inpaint", device=torch.device("cuda:0"), fp16=True, vram_O=False
        )

        # Convert to tensor
        input_tensor = transforms.ToTensor()(image).unsqueeze(0).cuda()
        input_tensor_mask = transforms.ToTensor()(image_mask).unsqueeze(0).cuda()

        # Resize to 512x512
        input_tensor = F.interpolate(input_tensor, size=(512, 512), mode="bilinear", align_corners=False)
        input_tensor_mask = F.interpolate(input_tensor_mask, size=(512, 512), mode="bilinear", align_corners=False)

        for _ in range(recursive):
            input_tensor = sd_inpaint.prompt_to_img(
                prompt,
                negative_prompts="",
                height=512,
                width=512,
                num_inference_steps=50,
                guidance_scale=7.5,
                latents=None,
                input_image=input_tensor,
                mask_image=input_tensor_mask,
                start_step=33,
            )
        imgs = input_tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return transforms.ToPILImage()(imgs[0])


def inpaint(image_path, mask_image_path, prompt, mode=0):
    # image and mask_image should be PIL images.
    # The mask structure is white for inpainting and black for keeping as is
    image = Image.open(image_path)
    mask_image = Image.open(mask_image_path)
    if mode == 0:
        image_gen = inpaint_mode_0(image, mask_image, prompt)
    elif mode == 1:
        image_gen = inpaint_mode_1(image, prompt, recursive=10)
    elif mode == 2:
        image_gen = inpaint_mode_2(image, mask_image, prompt, recursive=6)

    image_gen = transforms.PILToTensor()(image_gen)
    mask_image = transforms.PILToTensor()(mask_image) / 255
    image_gen_new_background = image_gen * mask_image + (1 - mask_image) * 128
    image_gen_new_background = torch.einsum("cwh->whc", image_gen_new_background).numpy().astype("uint8")

    image_gen_new_background = transforms.ToPILImage()(image_gen_new_background)
    image_gen_new_background.save(image_path.replace(".png", f"_inpaint_{mode}.png"))
    return image_gen_new_background


if __name__ == "__main__":
    prompt = "a cat."
    inpaint(image_path="./output/final_mesh.png", mask_image_path="./output/final_mesh_mask.png", prompt=prompt, mode=2)
    inpaint(image_path="./output/final_mesh.png", mask_image_path="./output/final_mesh_mask.png", prompt=prompt, mode=1)
    inpaint(image_path="./output/final_mesh.png", mask_image_path="./output/final_mesh_mask.png", prompt=prompt, mode=0)
