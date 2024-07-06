FROM spack/ubuntu-noble:develop as builder

RUN mkdir -p /opt/spack-environment
COPY spack.yaml /opt/spack-environment/spack.yaml
RUN cd /opt/spack-environment && spack env activate . && spack install --fail-fast && spack gc -y

# Strip all the binaries
RUN find -L /opt/views/view/* -type f -exec readlink -f '{}' \; | \
    xargs file -i | \
    grep 'charset=binary' | \
    grep 'x-executable\|x-sharedlib' | \
    awk -F: '{print $1}' | xargs strip

# Modifications to the environment that are necessary to run
RUN cd /opt/spack-environment && \
    spack env activate --sh -d . > activate.sh


# Bare OS image to run the installed executables
FROM ubuntu:24.04

ARG uid
ARG gid

COPY --from=builder /opt/spack-environment /opt/spack-environment
COPY --from=builder /opt/software /opt/software

# paths.view is a symlink, so copy the parent to avoid dereferencing and duplicating it
COPY --from=builder /opt/views /opt/views

RUN { \
      echo '#!/bin/sh' \
      && echo '.' /opt/spack-environment/activate.sh \
      && echo 'exec "$@"'; \
    } > /entrypoint.sh \
&& chmod a+x /entrypoint.sh \
&& ln -s /opt/views/view /opt/view


RUN apt-get -yqq update && apt-get -yqq upgrade \
    && apt-get -yqq install build-essential clangd clang-format curl gfortran \
    git vim wget zsh nano \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN . /opt/spack-environment/activate.sh && python -m ensurepip && python -m pip install black scipy numpy matplotlib pandas -i https://pypi.org/simple && rm -rf /root/.cache

ENTRYPOINT [ "/entrypoint.sh" ]

ARG USER=euler
RUN useradd --system --uid=${uid} --gid=${gid} --create-home --shell /bin/zsh $USER

USER $USER
ENV HOME /home/$USER
ENV USER $USER
RUN wget -O $HOME/.zshrc https://git.grml.org/f/grml-etc-core/etc/zsh/zshrc
WORKDIR $HOME
