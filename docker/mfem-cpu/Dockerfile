FROM ubuntu:24.04

ARG num_jobs
ARG hypre_version
ARG uid
ARG gid

USER root

RUN apt-get update && apt-get install -y \
   curl git git-lfs clang clang-tools clangd cmake autoconf \
   automake gdb libffi-dev zlib1g-dev python3-pip \
   libssl-dev xz-utils pkgconf gfortran openmpi-bin libopenmpi-dev \
   wget libarchive-tools htop libmetis-dev libparmetis-dev libopenblas-dev \
   zsh vim nano

RUN update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100 && \
   update-alternatives --install /usr/bin/cc cc /usr/bin/clang 100

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

ENV OMPI_CC=clang
ENV OMPI_CXX=clang++
ENV LD_LIBRARY_PATH=/usr/local/lib

# hypre
WORKDIR /opt/archives
RUN curl -L https://github.com/hypre-space/hypre/archive/refs/tags/v${hypre_version}.tar.gz > /opt/archives/hypre-v${hypre_version}.tar.gz
RUN tar xzf hypre-v${hypre_version}.tar.gz && cd hypre-${hypre_version}/src && \
    ./configure --prefix /usr/local --enable-shared --disable-static && \
    make -j ${num_jobs} && make install
WORKDIR /opt
RUN rm -rf /opt/archives

# mfem
WORKDIR /opt/archives
RUN git clone --depth=1 https://github.com/mfem/mfem.git mfem
COPY user.mk mfem/config/user.mk
RUN cd mfem && make config && make -j ${num_jobs} && cd examples && make ex1 && make ex1p \
    && mkdir /usr/local/bin/mfem-examples && cp ex1 ex1p /usr/local/bin/mfem-examples
RUN cd mfem && make install PREFIX=/usr/local

RUN chmod 644 /usr/local/include/mfem*.hpp
RUN find /usr/local/include/mfem -type f -exec chmod 644 {} \;

WORKDIR /opt
RUN rm -rf /opt/archives

ARG USER=euler
RUN useradd --system --uid=${uid} --gid=${gid} --create-home --shell /bin/zsh $USER

USER $USER
ENV HOME /home/$USER
ENV USER $USER
RUN wget -O $HOME/.zshrc https://git.grml.org/f/grml-etc-core/etc/zsh/zshrc
WORKDIR $HOME
