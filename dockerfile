# -------------------------------------------
#  PyTorch 2.5.1 | CUDA 12.4 | Python 3.11
# -------------------------------------------
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG USER=standard
ARG UID=1006
ARG GID=1006
ARG HOME=/home/$USER

# --------------------------------------------
#  1. Utente non-root (opzionale ma consigliato)
# --------------------------------------------
RUN groupadd -g $GID $USER && \
    useradd  -m -u $UID -g $GID -s /bin/bash $USER
USER $USER
WORKDIR $HOME

# --------------------------------------------
#  2. Aggiorna pip e installa librerie
# --------------------------------------------
RUN python -m pip install --upgrade pip && \
    python -m pip install \
        torchgeo==0.7.1 \
        prettytable \
        pycocotools \
        wandb \
        h5py \
        litdata\
        tensorboard\
        scikit-learn



# --------------------------------------------
#  4. Comando di default (modifica a piacere)
# --------------------------------------------
#    Per test rapidi:
#CMD ["python", "extract.py"]

#CONCATMEAN 

#CMD ["python", "main_unic.py", "--batch_size", "32", "--data_dir", "dati", "--arch", "grouping", "--saveckpt_freq", "5", "--in_chans", "9", "--output_dir", "ScalemaeDistill9/Vit_Large/ConcatMeanGROUP_Large", "--teachers", "scalemae_rgb,scalemae_veg,scalemae_geo", "--Teacher_strategy", "[\"mean\"]", "--transform", "True", "--num_frames", "1", "--imagenet_pretrained", "False", "--patch_size", "16"]

#SINGLE TEACHER
#CMD ["python", "main_unic.py", "--batch_size", "128", "--data_dir", "dati", "--arch", "vit_large", "--saveckpt_freq", "5", "--in_chans", "9", "--output_dir", "ScalemaeDistill9/Vit_Large/AAAA", "--teachers", "scalemae_rgb", "--Teacher_strategy", "", "--transform", "True", "--num_frames", "1", "--imagenet_pretrained", "False", "--patch_size", "16"]


#DINOBACKBONE
#CMD ["python", "main_unic.py", "--batch_size", "64", "--data_dir", "dati", "--arch", "vit_large", "--saveckpt_freq", "5", "--in_chans", "9", "--output_dir", "ScalemaeDistill9/Vit_Large/DinoTeacher/DinoV2LargePROVA", "--teachers", "DinoV2Large", "--Teacher_strategy", "", "--transform", "True", "--num_frames", "1", "--imagenet_pretrained", "False", "--patch_size", "14", "--loss", "cosine"]



#FINETUNING teacher 
#CMD ["python", "run_finetuning.py", "--batch_size", "128", "--data_dir", "dati", "--fintuning_bands", "rgb", "--output_dir", "modell/new/scalemae_RGB/", "--checkpoint_dir", "modell/new/scalemae_RGB/", "--checkpoint_path", "modell/new/scalemae_RGB/"] 

#CMD ["python", "run_finetuning.py", "--batch_size", "128", "--data_dir", "dati", "--fintuning_bands", "veg", "--output_dir", "modell/new/scalemae_VEG/", "--checkpoint_dir", "models/new/scalemae_VEG/", "--checkpoint_path", "models/new/scalemae_VEG/"] 

#CMD ["python", "run_finetuning.py", "--batch_size", "128", "--data_dir", "dati", "--fintuning_bands", "geo", "--output_dir", "modell/new/scalemae_GEO/", "--checkpoint_dir", "modell/new/scalemae_GEO/", "--checkpoint_path", "modell/new/scalemae_GEO/"] 

#CMD ["python", "run_finetuning.py", "--batch_size", "128", "--data_dir", "dati", "--fintuning_bands", "mix", "--output_dir", "modell/new/scalemae_MIX1/", "--checkpoint_dir", "modell/new/scalemae_MIX1/", "--checkpoint_path", "modell/new/scalemae_MIX1/", "--finetune_backbone", "True"] 


#FINETUNING student
#CMD ["python", "run_finetuning.py", "--batch_size", "128", "--data_dir", "dati", "--checkpoint_path", "ScalemaeDistill9/Vit_Large/ConcatMeanGROUP_Large/checkpoint_0020.pth", "--checkpoint_dir", "ScalemaeDistill9/Vit_Large/ConcatMeanGROUP_Large/", "--output_dir", "ScalemaeDistill9/Vit_Large/ConcatMeanGROUP_Large/", "--model", "vit", "--arch", "grouping", "--finetune_backbone", "False", "--in_chans", "9", "--lr", "1e-4", "--patience", "4", "--epochs", "100", "--fintuning_bands", "nove", "--patch_size", "16", "--transform", "False"]



#TEST
CMD ["python", "test_finetuning.py", "--batch_size", "128", "--data_dir", "dati", "--checkpoint_path", "/raid/home/rsde/cgiacchetta_unic/Distillation/ScalemaeDistill9/Vit_Large/ConcatMeanGROUP_Large/best-checkpointFreezed20.ckpt", "--output_dir", "/raid/home/rsde/cgiacchetta_unic/Distillation/ScalemaeDistill9/Vit_Large/ConcatMeanGROUP_Large/", "--model", "vit", "--arch", "grouping", "--in_chans", "9", "--finetuning_bands", "nove"]


