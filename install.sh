conda create -y -n  escher python=3.8
conda activate escher

conda install -y suitesparse
conda install -y -c conda-forge igl, ffmpeg

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .

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