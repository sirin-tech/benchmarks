FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

COPY ./keyboard /etc/default/keyboard

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  apt-utils \
  cuda-drivers

RUN mv /usr/lib/nvidia-375/libEGL.so.1 /usr/lib/nvidia-375/libEGL.so.1.org
RUN mv /usr/lib32/nvidia-375/libEGL.so.1 /usr/lib32/nvidia-375/libEGL.so.1.org
RUN ln -s /usr/lib/nvidia-375/libEGL.so.375.39 /usr/lib/nvidia-375/libEGL.so.1
RUN ln -s /usr/lib32/nvidia-375/libEGL.so.375.39 /usr/lib32/nvidia-375/libEGL.so.1
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.6 /usr/local/cuda/lib64/libcudnn.so.5

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  git \
  libgflags-dev \
  libgoogle-glog-dev \
  libgtest-dev \
  libiomp-dev \
  libleveldb-dev \
  liblmdb-dev \
  libopencv-dev \
  libopenmpi-dev \
  libprotobuf-dev \
  libsnappy-dev \
  openmpi-bin \
  openmpi-doc \
  protobuf-compiler \
  python-dev \
  python-numpy \
  python-pydot \
  python-pip \
  python-setuptools \
  python-scipy \
  python-tk \
  wget

RUN pip install --upgrade pip && pip install setuptools

RUN pip install \
  flask \
  future \
  graphviz \
  hypothesis \
  jupyter \
  matplotlib \
  numpy \
  protobuf \
  pydot \
  python-nvd3 \
  pyyaml \
  requests \
  scikit-image \
  scipy \
  six \
  tornado

ENV PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64/:/usr/local/cuda-8.0/lib64\ ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV PYTHONPATH=/usr/local
ENV LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

# Tensorflow installation
RUN pip install --upgrade tensorflow_gpu

# Caffe2 installation
RUN git clone --recursive https://github.com/caffe2/caffe2.git
WORKDIR /caffe2
RUN make
WORKDIR /caffe2/build
RUN make install

# Copying benchmarks
COPY ./benchmark/caffe /caffe
COPY ./benchmark/tensorflow /tensorflow

# Copying bash script
COPY ./benchmark.sh /benchmark.sh

WORKDIR /

# Elixir installation
RUN wget https://packages.erlang-solutions.com/erlang-solutions_1.0_all.deb && dpkg -i erlang-solutions_1.0_all.deb
RUN apt-get update && apt-get install -y esl-erlang
RUN apt-get install -y elixir

WORKDIR /

COPY ./benchmark/sirin /sirin
RUN cd /sirin/neuro && mix local.hex --force && cd /sirin/cuda && mix local.hex --force

WORKDIR /sirin/cuda

RUN mix clean
RUN GPU_SPEED_DEBUG=1 mix compile

WORKDIR /sirin/neuro

RUN mix compile

WORKDIR /

CMD ["/benchmark.sh"]
