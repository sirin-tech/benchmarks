FROM debian:jessie

# RUN deb http://httpredir.debian.org/debian jessie-backports main contrib non-free

# General APT-GET instalations
RUN apt-get update && apt-get install -y --no-install-recommends \
  aptitude \
  build-essential \
  cmake \
  git \
  libgoogle-glog-dev \
  libprotobuf-dev \
  protobuf-compiler \
  python-dev \
  python-tk \
  python-pip \
  wget

RUN pip install \
  numpy \
  protobuf

# Install CUDA TOOLKIT
RUN mkdir /distrs
WORKDIR /distrs
RUN wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"
RUN dpkg -i /distrs/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
# RUN apt-get update && apt-get install -y cuda
RUN aptitude install cuda
ENV PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\ ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Tensorflow installation
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp27-none-linux_x86_64.whl

# Caffe2 installation
# && cd caffe2
RUN git clone --recursive https://github.com/caffe2/caffe2.git
WORKDIR /distrs/caffe2
RUN make && cd build && make install
RUN python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
