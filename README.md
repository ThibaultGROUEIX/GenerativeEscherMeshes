# **Generative Escher Meshes**
Noam Aigerman, Thibault Groueix, _SIGGRAPH 2024_

[[paper]](https://arxiv.org/abs/2309.14564) [[website]](https://imagine.enpc.fr/~groueixt/escher) 

<img width="1016" alt="340161035-b44013bb-fb3c-408e-9516-fe9ae6e5cad1" src="https://github.com/ThibaultGROUEIX/GenerativeEscherMeshes/assets/11445067/8e5f57c9-1d45-4273-b895-e2b75ea091fb">




## Install

Torch need to be > 2.0 for the sparse solver.
We use CUDA118 and python3.8 but other might work.

```
git clone https://github.com/ThibaultGROUEIX/GenerativeEscherPatterns.git
cd GenerativeEscherPatterns
```

Run `./install.sh` or follow these steps :

```
conda create -y -n  escher python=3.8
conda activate escher
```

then

```
conda install -y suitesparse
conda install -y -c conda-forge igl ffmpeg

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

### Nvdiffrast renderer
Install the CUDA TOOLKIT 11.8.
```sh
export CUDA_HOME=/usr/local/cuda
sudo chmod -R 777 /usr/local/cuda 
git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    ninja-build \
    curl
export PYTHONDONTWRITEBYTECODE=1
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export PYOPENGL_PLATFORM=egl
pip install --upgrade pip
# python setup.py install : NotADirectoryError
pip install .
```

### DeeeFloyd

Follow these additional steps to install DeepFloyd
```sh
pip install bitsandbytes
pip install sentencepiece
``` 

## Quick Run

Remember to always activate the conda environment : `conda activate escher`
We use OmegaConf to load arguments. Base arguments are defined in `configs/base.yaml`. They can be overwritten by the command line :
```
python -m escher.main TILING_TYPE="OrbifoldIII" PROMPT="A beautiful illustration of a flower, a masterpiece" OUTPUT_DIR="./output"
```

* TILING_TYPE can be either of : Cylinder, KleinBottle, MobiusStrip, OrbifoldI, OrbifoldIHybrid, OrbifoldII,OrbifoldIIHybrid, OrbifoldIII, OrbifoldIV,OrbifoldIVHybrid, PinnedBoundary, ProjectivePlane, ReflectSquare, RightAngleHybrid, Torus


If you want to reuse a specific config file from a prior experiment:
```python -m escher.main CONF_FILE=/path/to/config/config.yaml```

### Other arguments :
Check out all the arguments in `configs/base.yaml`. 


### Visualization 
By default, the code will generate tilings at different resolution, as well as a video of the camera moving over the liting. You can achieve the same result from a checkpoint (all logs from an experiment are stored in a `.pkl` )

```
python -m escher.rendering.render_tiling_from_pkl --path path/to/pkl --make_infinite_video --grid_sizes 10 --make_video --num_labels 1
```
This will produce a video of the camera moving over the tiling, as well as a video where tiles appear one by one, as well as static images of the tiling at different resolutions.

## Misc
* Deepfloyd is twice faster than SD (6it/sec versus 3it/sec)
* SD with latent opt is even faster (10it / seconds)
* High guidance is critical
* Larger batch-sizes help

## Areas of Improvements
* Distorsion is probably bad for optimization. It would be great to continuously remesh and update the constraints accordingly
![319353292-877d74b9-8d0f-472d-835e-bab4884eedb7](https://github.com/ThibaultGROUEIX/GenerativeEscherMeshes/assets/11445067/36fcb5ce-7bc2-4633-869a-f4190703de8f)
* Post-process the texture with generative fill to create texture variations with the same shape
* Autoconvert the output to Adobe Illustrator file format

