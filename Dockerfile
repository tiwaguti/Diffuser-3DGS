FROM tiwaguti/colmap-dev:latest as colmap
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# common ubuntu settings
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt-get update && \
    apt-get install -y openssh-server sudo wget vim git libgl1-mesa-dev ffmpeg

# install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
    rm -r Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/opt/miniconda3/bin:${PATH}
RUN conda init

# enter conda env
RUN conda create -y -n pointrix python=3.9
SHELL ["conda", "run", "-n", "pointrix", "/bin/bash", "-c"]

# install pytorch
RUN conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit
RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

# install requirements
RUN pip install \
    pytorch_msssim \
    jaxtyping \
    omegaconf \
    tabulate \
    lpips \
    tensorboard \
    packaging \
    rich \
    imageio \
    plyfile \
    pyqt5 \
    opencv-python \
    opencv-contrib-python \
    ninja \
    gradio \
    scikit-image \
    configargparse \
    tensorboardX>=2.0 \
    einops


# build pointrix dependencies
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.6+PTX"
# RUN git clone --depth 1 https://github.com/pointrix-project/dptr.git --recursive /usr/local/dptr && \
#     cd /usr/local/dptr && \
#     pip install -e .
RUN pip install "git+https://gitlab.inria.fr/bkerbl/simple-knn.git"

# colmap
# Minimal dependencies to run COLMAP binary compiled in the builder stage.
# Note: this reduces the size of the final image considerably, since all the
# build dependencies are not needed.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        libboost-filesystem1.74.0 \
        libboost-program-options1.74.0 \
        libc6 \
        libceres2 \
        libfreeimage3 \
        libgcc-s1 \
        libgl1 \
        libglew2.2 \
        libgoogle-glog0v5 \
        libqt5core5a \
        libqt5gui5 \
        libqt5widgets5
COPY --from=colmap /colmap_installed/ /usr/local/

# setup .bashrc
RUN echo "conda activate pointrix" >> /root/.bashrc
WORKDIR /workspaces