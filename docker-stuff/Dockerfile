# FROM pytorch/pytorch:latest
#FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
WORKDIR /train-docker

# Get wget and unzip
#RUN apt-get update && apt-get install -y wget unzip

# Download the data (http://vision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-dataset.zip) and unzip it
# RUN wget http://vision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-dataset.zip
# RUN unzip suction-based-grasping-dataset.zip

ENV DEBIAN_FRONTEND=noninteractive

# Copy the data folder to the docker container
COPY data/ data/
COPY data_synthetic/ data_synthetic/

# Run update
RUN apt-get update


# Install libgl1-mesa-glx for opencv
RUN apt-get install -y libopencv-dev

# Download python3.10
RUN apt-get install -y python3.10
RUN apt-get install -y python3-pip


# RUN mkdir pip-cache
# RUN python3 -mpip install segmentation_models_pytorch.Unet --cache-dir pip-cache

# Copy python requirements and install them
COPY requirements.txt .
RUN python3 -mpip install -r requirements.txt
RUN python3 -mpip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117

# Copy the python files
COPY train.py .

RUN mkdir results

#CMD ["python3", "train.py"]

