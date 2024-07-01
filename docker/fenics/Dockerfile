FROM dolfinx:nightly

ARG uid
ARG gid

USER root

RUN apt-get update && apt-get install -y zsh vim nano
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

ARG USER=euler
RUN useradd --system --uid=${uid} --gid=${gid} --create-home --shell /bin/zsh $USER

USER $USER
ENV HOME /home/$USER
ENV USER $USER
RUN wget -O $HOME/.zshrc https://git.grml.org/f/grml-etc-core/etc/zsh/zshrc
WORKDIR $HOME
