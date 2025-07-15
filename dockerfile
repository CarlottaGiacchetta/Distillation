FROM nvcr.io/nvidia/pytorch:24.05-py3
ARG USER=standard
ARG USER_ID=1006 # uid from the previus step
ARG USER_GROUP=standard
ARG USER_GROUP_ID=1006 # gid from the previus step
ARG USER_HOME=/home/${USER}


RUN groupadd --gid $USER_GROUP_ID $USER \
    && useradd --uid $USER_ID --gid $USER_GROUP_ID -m $USER


RUN apt-get update && apt-get install -y curl


USER $USER


RUN  pip install torchgeo 



#CONCATMEAN 

CMD ["python", "main_unic.py", "--batch_size", "128", "--data_dir", "dati", "--arch", "vit_large", "--saveckpt_freq", "5", "--in_chans", "9", "--output_dir", "ScalemaeDistill9/Vit_Large/ConcatMean", "--teachers", "scalemae_rgb,scalemae_veg,scalemae_geo", "--Teacher_strategy", "[\"mean\"]", "--transform", "True", "--num_frames", "1", "--imagenet_pretrained", "False", "--patch_size", "16"]
