FROM ubuntu:18.04
LABEL maintainer="geaxgx@gmail.com"

ARG proxy

ENV http_proxy $proxy
ENV https_proxy $proxy
ENV PATH="/root/miniconda3/bin:${PATH}"

# For dbind-WARNING **: 11:17:47.806: Couldn't connect to accessibility bus: Failed to connect to socket /tmp/dbus-Rg7Us89bAW: Connection refused
ENV NO_AT_BRIDGE=1

ARG PATH="/root/miniconda/envs/tello:/root/miniconda3/bin:${PATH}"
ARG DEBIAN_FRONTEND=noninteractive

# Exit when RUN get non-zero value
RUN set -e 

# Get basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
	apt-utils 
RUN apt-get install -y --no-install-recommends \
	locales ca-certificates
RUN apt-get install -y --no-install-recommends \
    pciutils \
    build-essential \
	sudo \
	nano \
    wget \
    git \
	gcc \
	cmake \
	libcanberra-gtk3-module \
    alsa-base \
    alsa-utils \
    pulseaudio

# Clone github reposirotry https://github.com/geaxgx/tello-openpose.git
RUN cd / && git clone https://github.com/geaxgx/tello-humanpose-openvino.git

# Miniconda
RUN wget --no-check-certificate \
   https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
   && mkdir /root/.conda \
   && bash Miniconda3-latest-Linux-x86_64.sh -b \
   && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda env create -f /tello-humanpose-openvino/environment.yml 
RUN conda init bash
RUN echo "conda activate tello" >> ~/.bashrc

# Openvino
ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/16345/l_openvino_toolkit_p_2020.1.023_online.tgz
ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/openvino_installer
RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
    wget -c $DOWNLOAD_LINK && \
    tar xf l_openvino_toolkit_p_2020.1.023_online.tgz && \
    cd l_openvino_toolkit_p_2020.1.023_online && \
    ./install_openvino_dependencies.sh

# Packages Openvino: Inference Engine CPU, OpenCV
RUN cd $TEMP_DIR/l_openvino_toolkit_p_2020.1.023_online && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg --components "intel-openvino-ie-sdk-ubuntu-bionic__x86_64;intel-openvino-ie-rt-cpu-ubuntu-bionic__x86_64;intel-openvino-opencv-lib-ubuntu-bionic__x86_64" 

RUN echo "source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6" >> ~/.bashrc


# Install some python packages
RUN conda run -n tello pip install pynput simple_pid pygame

# Install modified version of TelloPy
RUN cd /tello-humanpose-openvino/TelloPy && conda run -n tello python setup.py bdist_wheel && conda run -n tello pip install dist/tellopy-*.dev*.whl --upgrade

WORKDIR /tello-humanpose-openvino
CMD bash
