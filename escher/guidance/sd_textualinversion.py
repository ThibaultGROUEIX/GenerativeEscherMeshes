"""
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export DATA_DIR="/home/groueix/GenerativeEscherPatterns/Textual/2_dragon"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<2dragon-toy>" \
  --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="textual_inversion_2_dragon" \
  """
  
  
from diffusers import StableDiffusionPipeline
import torch

pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16).to("cuda")
pipeline.load_textual_inversion("/home/groueix/GenerativeEscherPatterns/diffusers/examples/textual_inversion/textual_inversion_cat/learned_embeds.safetensors")

def generate_image():
    image = pipeline("A <cat-toy> mug", num_inference_steps=50).images[0]
    image.save("cat-train.png")

if __name__ == "__main__":
    generate_image()