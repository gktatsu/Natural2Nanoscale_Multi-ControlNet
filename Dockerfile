FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime


# Update and upgrade system packages
RUN apt-get update && apt-get upgrade -y
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6 git libgmp-dev libmpfr-dev libmpc-dev rsync gcc

# Set the working directory in the container
WORKDIR /Natural2Nanoscale

COPY . /Natural2Nanoscale

RUN pip install -r req.txt
