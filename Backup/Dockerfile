#FROM debian:jessie
FROM nvidia/cuda

COPY ./keyboard /etc/default/keyboard

# General APT-GET instalations
RUN apt-get update && apt-get install -y --no-install-recommends \
  apt-utils \
  build-essential \
  cmake \
  git \
  libgoogle-glog-dev \
  libprotobuf-dev \
  protobuf-compiler \
  python-dev \
  python-tk \
  python-pip

RUN pip install --upgrade pip

RUN pip install \
  numpy \
  protobuf

ENV PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda-8.0/lib64\ ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# jessie-backports
# RUN echo 'deb http://httpredir.debian.org/debian jessie-backports main contrib non-free' >> /etc/apt/sources.list
# RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --allow-unauthenticated -y -t jessie-backports nvidia-driver

RUN mkdir /distrs
COPY libcudnn6_6.0.21-1+cuda8.0_amd64.deb /distrs/
RUN DEBIAN_FRONTEND=noninteractive dpkg -i /distrs/libcudnn6_6.0.21-1+cuda8.0_amd64.deb

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-drivers

RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.6 /usr/local/cuda/lib64/libcudnn.so.5

# Tensorflow installation
RUN pip install --upgrade tensorflow_gpu


# Caffe2 installation
#RUN git clone --recursive https://github.com/caffe2/caffe2.git
#WORKDIR /distrs/caffe2
#RUN make && cd build && make install
#RUN python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

# Cleanup
#WORKDIR /
#RUN rm -rf /distrs
