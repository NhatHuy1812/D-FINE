FROM registry.cn-hangzhou.aliyuncs.com/peterande/dfine:v1


RUN apt install -y tensorrt

RUN conda init

# create env
RUN conda create -n dfine python=3.11.9 pip -y

# install everything into dfine without activating
RUN conda run -n dfine python -m pip install --upgrade pip setuptools && \
    conda run -n dfine conda install -y -c conda-forge numpy && \
    conda run -n dfine pip install --no-cache-dir \
      jupyterlab pycocotools PyYAML tensorboard scipy \
      faster-coco-eval calflops transformers loguru opencv-python nvitop onnx onnxsim matplotlib && \
    conda run -n dfine pip install --default-timeout=10000 torch torchvision

# add TensorRT bin to PATH at build-time (no need to source later)
ENV PATH=/usr/src/tensorrt/bin:$PATH

# make sure bash is used so conda init hooks in
SHELL ["/bin/bash", "-lc"]
