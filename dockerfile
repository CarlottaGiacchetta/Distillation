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

#CMD ["python", "main_unic.py", "--batch_size", "128", "--data_dir", "dati", "--arch", "vit_large", "--saveckpt_freq", "5", "--in_chans", "9", "--output_dir", "ScalemaeDistill9/Vit_Large/ConcatMeanAUG", "--teachers", "scalemae_rgb,scalemae_veg,scalemae_geo", "--Teacher_strategy", "[\"mean\"]", "--transform", "True", "--num_frames", "1", "--imagenet_pretrained", "False", "--patch_size", "16"]



#SINGLE TEACHER
#CMD ["python", "main_unic.py", "--batch_size", "128", "--data_dir", "dati", "--arch", "vit_large", "--saveckpt_freq", "5", "--in_chans", "9", "--output_dir", "ScalemaeDistill9/Vit_Large/AAAA", "--teachers", "scalemae_rgb", "--Teacher_strategy", "", "--transform", "True", "--num_frames", "1", "--imagenet_pretrained", "False", "--patch_size", "16"]


#DINOBACKBONE
CMD ["python", "main_unic.py", "--batch_size", "64", "--data_dir", "dati", "--arch", "vit_large", "--saveckpt_freq", "5", "--in_chans", "9", "--output_dir", "ScalemaeDistill9/Vit_Large/DinoTeacher/DinoV2Large_baseline", "--teachers", "DinoV2Large_baseline", "--Teacher_strategy", "", "--transform", "True", "--num_frames", "1", "--imagenet_pretrained", "False", "--patch_size", "14", "--loss", "cosine"]



#FINETUNING teacher 
#CMD ["python", "run_finetuning.py", "--batch_size", "128", "--data_dir", "dati", "--fintuning_bands", "rgb", "--output_dir", "models/new/scalemae_RGB/", "--checkpoint_dir", "models/new/scalemae_RGB/", "--checkpoint_path", "models/new/scalemae_RGB/"] 

#CMD ["python", "run_finetuning.py", "--batch_size", "128", "--data_dir", "dati", "--fintuning_bands", "veg", "--output_dir", "models/new/scalemae_VEG/", "--checkpoint_dir", "models/new/scalemae_VEG/", "--checkpoint_path", "models/new/scalemae_VEG/"] 

#CMD ["python", "run_finetuning.py", "--batch_size", "128", "--data_dir", "dati", "--fintuning_bands", "geo", "--output_dir", "models/new/scalemae_GEO/", "--checkpoint_dir", "models/new/scalemae_GEO/", "--checkpoint_path", "models/new/scalemae_GEO/"] 


#FINETUNING student
#CMD ["python", "run_finetuning.py", "--batch_size", "64", "--data_dir", "dati", "--checkpoint_path", "ScalemaeDistill9/Vit_Large/ConcatMean/checkpoint_0020.pth", "--checkpoint_dir", "ScalemaeDistill9/Vit_Large/ConcatMean/", "--output_dir", "ScalemaeDistill9/Vit_Large/ConcatMean/", "--model", "vit", "--arch", "vit_large", "--finetune_backbone", "False", "--in_chans", "9", "--lr", "1e-4", "--patience", "4", "--epochs", "100", "--fintuning_bands", "nove", "--patch_size", "16"]


#TEST
#CMD ["python", "test_finetuning.py", "--batch_size", "128", "--data_dir", "dati", "--checkpoint_path", "/raid/home/rsde/cgiacchetta_unic/Distillation/ScalemaeDistill9/Vit_Large/PROVA/best-checkpoint020.ckpt", "--output_dir", "/raid/home/rsde/cgiacchetta_unic/Distillation/ScalemaeDistill9/Vit_Large/PROVA/", "--model", "vit", "--arch", "vit_large", "--in_chans", "9", "--finetuning_bands", "nove"]

