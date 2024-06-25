"""
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="/home/groueix/GenerativeEscherPatterns/data_dreambooth/4_escher"
export OUTPUT_DIR="/home/groueix/GenerativeEscherPatterns/data_dreambooth/4_escher_dreambooth"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 
  
"""

from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("/home/groueix/GenerativeEscherPatterns/data_dreambooth/4_escher_dreambooth", torch_dtype=torch.float16, use_safetensors=True).to("cuda")

image = pipeline("A photo of sks smoking a pipe", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("test/4_escher_dreambooth_pipe.png")

image = pipeline("A photo of sks", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("test/4_escher_dreambooth.png")

image = pipeline("A photo of sks at the beach", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("test/4_escher_dreambooth_beach.png")