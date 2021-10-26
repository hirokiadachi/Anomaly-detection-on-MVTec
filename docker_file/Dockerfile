FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV WORK_DIR=/root
WORKDIR $WORK_DIR

ADD ./set_jupyter_lab.sh $WORK_DIR

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV CPATH=/usr/local/include:$CPATH
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=$CUDA_PATH/bin:$PATH
ENV CPATH=$CUDA_PATH/include:$CPATH
ENV LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/lib:$LD_LIBRARY_PATH
ENV DEBIAN_FRONTEND=noninteractive

RUN rm -rf /var/lib/apt/lists/*\
            /etc/apt/source.list.d/cuda.list\
            /etc/apt/source.list.d/nvidia-ml.list

RUN apt-get update -y \
 && apt-get upgrade -y
 
RUN curl -sL https://deb.nodesource.com/setup_13.x |bash - \
 && apt-get install -y --no-install-recommends build-essential\
                                               apt-utils\
                                               ca-certificates\
                                               make\
                                               cmake\
                                               wget\
                                               git\
                                               curl\
                                               vim\
                                               nodejs
RUN apt-get autoremove -y
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* \
           /var/cache/apt/* \
           /usr/local/src/* \
           /tmp/*
           

######################################
# Python
######################################
RUN apt-get update\
 && apt-get install unzip\
 && apt-get install -y software-properties-common\
 && add-apt-repository ppa:deadsnakes/ppa\
 && apt-get update\
 && apt-get install -y python3.6 python3.6-dev python3-distutils-extra\
 && wget -O ~/get-pip.py https://bootstrap.pypa.io/get-pip.py\
 && python3.6 ~/get-pip.py\
 && ln -s /usr/bin/python3.6 /usr/local/bin/python3\
 && ln -s /usr/bin/python3.6 /usr/local/bin/python\
 && pip install setuptools \
 && pip install numpy scipy pandas cloudpickle scikit-learn matplotlib Cython seaborn\
 && pip install lmdb

######################################
# OpenCV
######################################
RUN apt-get install -y --no-install-recommends libatlas-base-dev\
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler
RUN git clone --branch 4.0.1 https://github.com/opencv/opencv ~/opencv && \
mkdir -p ~/opencv/build && cd ~/opencv/build && \
cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_IPP=OFF \
          -D WITH_CUDA=OFF \
          -D WITH_OPENCL=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          .. &&\
    make -j"$(nproc)" install && \
    ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2

######################################
# pytorch
######################################
RUN pip install future \
 && pip install numpy \
 && pip install protobuf\
 && pip install enum34\
 && pip install pyyaml \
 && pip install typing\
 && pip install pillow\
 && pip install matplotlib\
 && pip install scikit-learn\
 && pip install tqdm\
 && pip install scipy\
 && pip install pandas\
 && pip install h5py\
 && pip install bhtsne umap-learn\
 && pip install advertorch \ 
 && pip install imageio \
 && pip install timm

RUN pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0

######################################
# jupyter
######################################
RUN pip install jupyter \
 && pip install --upgrade --no-cache-dir 'jupyterlab~=3.0'
 
RUN apt install -y nodejs npm
RUN npm install n -g
RUN n stable
RUN apt purge -y nodejs npm

RUN pip install jupyterlab-git \
 && pip install plotly \
 && pip install jedi==0.17.2

RUN pip install --upgrade --no-cache-dir \
    'jupyterlab-kite>=2.0.2' \
    jupyterlab_code_formatter \
    jupyterlab-vimrc \
    yapf \
 && rm -rf ~/.cache/pip

######################################
# tensorflow
######################################
RUN pip install --no-cache-dir tensorflow-gpu==1.14.0

EXPOSE 8888 6006
