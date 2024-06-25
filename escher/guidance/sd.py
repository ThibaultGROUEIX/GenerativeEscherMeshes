from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from packaging import version

"""
Updated from ThreeStudio : Apache-2.0 license
"""
# Basic types
from typing import Any, Optional

# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import Float, Int

# PyTorch Tensor type
from torch import Tensor


def parse_version(ver: str):
    return version.parse(ver)


def C(value: Any, epoch: int, global_step: int) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
        elif isinstance(end_step, float):
            current_step = epoch
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
    return value


@dataclass
class Config:
    textual_inversion: str = ""
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
    enable_memory_efficient_attention: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_attention_slicing: bool = False
    enable_channels_last_format: bool = False
    guidance_scale: float = 100.0
    grad_clip: Optional[Any] = field(default_factory=lambda: [0, 2.0, 8.0, 1000])
    half_precision_weights: bool = True

    min_step_percent: float = 0.02
    max_step_percent: float = 0.98

    use_sjc: bool = False
    var_red: bool = True
    weighting_strategy: str = "sds"

    token_merging: bool = False
    token_merging_params: Optional[dict] = field(default_factory=dict)


class StableDiffusion(nn.Module):
    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configure()

    def configure(self) -> None:
        print(f"Loading Stable Diffusion ... from {self.cfg.pretrained_model_name_or_path}!")

        self.weights_dtype = torch.float16 if self.cfg.half_precision_weights else torch.float32

        pipe_kwargs = {
            "safety_checker": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)
        
        if self.cfg.textual_inversion != "":
            print(f"Loading textual inversion from {self.cfg.textual_inversion}")
            self.pipe.load_textual_inversion(self.cfg.textual_inversion)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                print("PyTorch2.0 uses memory efficient attention by default.")
            elif not is_xformers_available():
                print("xformers is not available, memory efficient attention is not enabled.")
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        # for p in self.tokenizer.parameters():
        #     p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        if self.cfg.use_sjc:
            # score jacobian chaining use DDPM
            self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)
        if self.cfg.use_sjc:
            # score jacobian chaining need mu
            self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)

        self.grad_clip_val: Optional[float] = None

        print(f"Loaded Stable Diffusion!")

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt, negative_prompt: [str]

        # positive
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(self, imgs: Float[Tensor, "B 3 512 512"]) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(latents, (latent_height, latent_width), mode="bilinear", align_corners=False)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        t: Int[Tensor, "B"],
    ):
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
            )

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.cfg.weighting_strategy}")

        grad = w * (noise_pred - noise)
        return grad

    def compute_grad_sjc(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        t: Int[Tensor, "B"],
    ):
        sigma = self.us[t]
        sigma = sigma.view(-1, 1, 1, 1)
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            y = latents

            zs = y + sigma * noise
            scaled_zs = zs / torch.sqrt(1 + sigma**2)

            # pred noise
            latent_model_input = torch.cat([scaled_zs] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
            )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

            Ds = zs - sigma * noise_pred

            if self.cfg.var_red:
                grad = -(Ds - y) / sigma
            else:
                grad = -(Ds - zs) / sigma

        return grad

    def train_step(
        self,
        rgb: Float[Tensor, "B H W C"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        rgb_as_latents=False,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = F.interpolate(rgb_BCHW, (64, 64), mode="bilinear", align_corners=False)
        else:
            rgb_BCHW_512 = F.interpolate(rgb_BCHW, (512, 512), mode="bilinear", align_corners=False)
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        if self.cfg.use_sjc:
            grad = self.compute_grad_sjc(latents, text_embeddings, t)
        else:
            grad = self.compute_grad_sds(latents, text_embeddings, t)

        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return loss, t
        # return {
        #     "sds": loss,
        #     "grad_norm": grad.norm(),
        #     "timestep": t,
        # }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)
