FROM condaforge/miniforge3:latest
# FROM mambaorg/micromamba:jammy-cuda-12.1.0
USER root

RUN apt-get update -y && apt install -y \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    g++

RUN apt-get update -y && apt install -y \
    build-essential

# COPY . .
WORKDIR /diffusion_policy
ADD conda_environment.yaml .
RUN mamba env create -y -f conda_environment.yaml

