FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev vim git wget sudo psmisc locales cmake htop
RUN locale-gen en_US.UTF-8
ADD https://raw.githubusercontent.com/Ian-Mint/pong-underwater-rl/master/gpu-environment.yml /tmp/environment.yml
RUN conda env create --file /tmp/environment.yml
RUN conda init bash && echo "source activate torch" >> ~/.bashrc
ENV PATH /opt/conda/envs/torch/bin:$PATH