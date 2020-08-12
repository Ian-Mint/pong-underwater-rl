FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx vim git wget sudo psmisc locales cmake htop pylint
RUN locale-gen en_US.UTF-8
ADD environment.yml /tmp/environment.yml
RUN conda env create --file /tmp/environment.yml
RUN conda init bash && echo "source activate torch" >> ~/.bashrc
ENV PATH /opt/conda/envs/torch/bin:$PATH
ADD vimrc /root/.vimrc
RUN git clone https://github.com/VundleVim/Vundle.vim.git /root/.vim/bundle/Vundle.vim
