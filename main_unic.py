# -*- coding: utf-8 -*-
import argparse
import datetime
import math
import os
import shutil
import sys
import time
import logging
import ast

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torchvision.transforms import v2 as T
from torchvision.datasets import ImageFolder

import utils
from modeling.unic import build_student_from_args
from teachers import build_teachers, get_teacher_output
from teachers.concat import TeacherAggregator
from modeling.losses import unic_loss
from dinov2.logging import setup_logging, ExternalLogger, MetricLogger
from dinov2.distributed import get_global_rank
from Dataset import carica_dati


logger = logging.getLogger()




def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arch",
        type=str,
        default="vit_base", #or time_tiny
        help="Architecture of the student model. "
        "See dinov2/models/vision_transformer.py for options. See dinov2/models/timesformer.py for options.",
    )
    
    parser.add_argument("--imagenet_pretrained", 
        type=utils.bool_flag,
        default=False,
        help="Se True, carica i pesi ImageNet di default per larch scelto",
    )

    parser.add_argument("--finetune_backbone", 
        type=utils.bool_flag,
        default=False,
        help="Se True, finetuning anche del backbone",
    )
    
    parser.add_argument(
        "--num_frames",
        type=int,
        default=3,
        help="Architecture of the student model. "
        "See dinov2/models/vision_transformer.py for options.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Patch size for the student model.",
    )
    parser.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.1,
        help="Drop path rate for the student model.",
    )
    parser.add_argument(
        "--lp_args",
        type=str,
        default="{'std': 0.02}",
        help="Dictionary of keyword arguments for the ladder of projector.",
    )
    parser.add_argument(
        "--use_lp",
        type=utils.bool_flag,  # oppure `type=bool` se non usi la tua `bool_flag`
        default=True,
        help="Whether to use Ladder of Projection modules.",
    )


    parser.add_argument(
        "--teachers",
        type=str,
        default="scalemae_rgb,scalemae_veg,scalemae_geo",#, scalemae_veg",
        help="Comma-separated list of teacher names.",
    )
    parser.add_argument(
        "--Teacher_strategy",
        type=lambda s: eval(s) if s else [],  # converte stringa in lista
        default=[],
        help='Fusion strategy, e.g., \'["rab", "abf"]\''
    )
    parser.add_argument(
        "--Student_strategy",
        type=lambda s: eval(s) if s else [],  # converte stringa in lista
        default=[],
        help='Fusion strategy delle features dello student, e.g., \'["rab", "abf"]\''
    )
    parser.add_argument(
        "--aggregation_parameter",
        type=str,
        default="{'alpha': 0.9, 'beta': 0.1}",
        help="Dictionary of keyword arguments (alpha - mean, beta - abf) for the aggregation (sum must be 1).",
    )
    parser.add_argument(
        "--aggregation_scheduler",
        type=utils.bool_flag,
        default=False,
        help="cambia i parametri alpha e beta.",
    )
    parser.add_argument(
        "--tnorm_ema_momentum_start",
        type=float,
        default=1.0,
        help="Starting value for the EMA momentum for teacher feature statistics.",
    )
    parser.add_argument(
        "--tnorm_ema_momentum_end",
        type=float,
        default=0.001,
        help="Final value for the EMA momentum for teacher feature statistics.",
    )
    parser.add_argument(
        "--t_drop_prob",
        type=float,
        default=0.5,
        help="Probability of dropping a teacher in loss.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="cosine", #cross-entropy
        help="Loss type.",
    )
    parser.add_argument(
        "--lam_lcos",
        default=0.5,
        type=float,
        help="Coefficient for the cosine loss.",
    )
    parser.add_argument(
        "--lam_lsl1",
        default=0.5,
        type=float,
        help="Coefficient for the smooth L1 loss.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path for the BigearthNet data directory",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size (for both training and validation). "
        "We assume the input images are square.",
    )

    parser.add_argument(
        "--use_fp16",
        type=utils.bool_flag,
        default=True,
        help="Whether or not to use mixed precision for training.",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=0.0,
        help="Gradient clipping value.",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=128,
        type=int,
        help="Batch size per GPU. Total batch size is proportional to the number of GPUs.",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size.Total batch size is proportional to the number of GPUs.",
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--optim_args",
        type=str,
        default="{'betas':(0.9, 0.99), 'eps':1e-8}",
        help="Dictionary of keyword arguments for the optimizer.",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=3e-2,
        help="Weight decay for the SGD optimizer.",
    )
    parser.add_argument(
        "--lr",
        default=3e-4,
        type=float,
        help="Maximum learning rate at the end of linear warmup.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate at the end of training.",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=10,
        type=int,
        help="Number of training epochs for the learning-rate-warm-up phase.",
    )

    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Path to the output folder to save logs and checkpoints.",
    )
    parser.add_argument(
        "--saveckpt_freq",
        default=20,
        type=int,
        help="Frequency of intermediate checkpointing.",
    )
    parser.add_argument(
        "--seed",
        default=22,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        default=12,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore this argument; No need to set it manually.",
    )
    parser.add_argument(
        "--transform",
        default=True,
        type=utils.bool_flag,
        help="Do augmentations",
    )
    parser.add_argument(
        "--fintuning_bands",
        default="",
        type=str,
        help="Weight the label of the loss",
    )
    parser.add_argument(
        "--bands",
        type=str,
        default="s2",
        help="Multispectral bands of BigEarthNet.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=19,
        help="Num of classes for classification of BigEarthNet.",
    )
    parser.add_argument(
        "--in_chans",
        type=int,
        default=12,
        help="Num of classes for classification of BigEarthNet.",
    )
    
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="Url used to set up distributed training.",
    )

    args = parser.parse_args()

    args.teachers = args.teachers.split(",")
    args.num_cpus = len(os.sched_getaffinity(0))
    try:
        args.aggregation_parameter = ast.literal_eval(args.aggregation_parameter)
    except Exception as e:
        raise ValueError(f"Invalid --aggregation_parameter: {args.aggregation_parameter}. Error: {e}")
    
    if not isinstance(args.aggregation_parameter, dict):
        raise ValueError("--aggregation_parameter must be a dictionary.")
    os.makedirs(args.output_dir, exist_ok=True)
    

    return args

logger = logging.getLogger()

def main(args):

    ema_momentum_start = 0.996      # tipico 0.99�0.999
    ema_momentum_end   = 0.996
    
    
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed + get_global_rank())
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True

    setup_logging(os.path.join(args.output_dir, "log.txt"), level=logging.INFO)
    utils.print_program_info(args)
    ext_logger = ExternalLogger(args.output_dir)

    logger.info("Creating data loaders ...")
    train_loader, val_loader = carica_dati(args)
    
    sample = next(iter(train_loader))
    logger.info(f"Shape batch immagini: {sample['image'].shape}")



    logger.info("Loading teachers ...")
    teachers, teacher_ft_stats, teacher_dims = build_teachers(args.teachers, args.loss)

    if not args.Teacher_strategy:
        aggregator = None
        has_trainable_params = None

    else:
        logger.info("Build aggregator ...", args.Teacher_strategy)
        aggregator = TeacherAggregator(teacher_dims, args.Teacher_strategy).cuda()
    
        # Controlla se ha parametri che richiedono gradiente
        has_trainable_params = any(p.requires_grad for p in aggregator.parameters())
        
        if has_trainable_params:
            print('has trainable params')
            aggregator = nn.parallel.DistributedDataParallel(aggregator, device_ids=[args.gpu], find_unused_parameters=True)
            logger.info("Wrapped aggregator in DDP.")
        else:
            logger.info("Using strategy - identity fallback")


    logger.info("Creating student model")
    baselines = [t for t in args.teachers if 'baseline' in t]

    if baselines:
        from teachers.teachers_config import TEACHER_CFG
        cfg        = TEACHER_CFG[baselines[0]]
        ckpt_path  = cfg.get("ckpt_path", "")
        ckpt_key   = cfg.get("ckpt_key", None)
        loader_fn  = cfg["loader"]    
        model = loader_fn(ckpt_path, args.loss)
    else:
        model = build_student_from_args(args)
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu], find_unused_parameters=True
    )
    utils.save_model_defn(model.module, os.path.join(args.output_dir, "model_defn.txt"))
    
    param_groups = utils.get_params_groups(
          model,
          save_file_path=os.path.join(args.output_dir, "params_groups.txt"),
      )
      
    if has_trainable_params and args.Teacher_strategy:          # True se strategy ha "rab" o "abf"
        param_groups.append(
            {
                "params": [p for p in aggregator.parameters()
                          if p.requires_grad],
                # volendo puoi assegnare un LR diverso, es:
                # "lr": args.lr * 0.5
            }
        )
    
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=0,
        **eval(args.optim_args),
    )
    logger.info("Optimizer: {}".format(optimizer))

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    args.lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu) / 256.0,
        args.min_lr,
        args.epochs,
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )

    args.tnorm_ema_schedule = utils.cosine_scheduler(
        args.tnorm_ema_momentum_start,
        args.tnorm_ema_momentum_end,
        args.epochs,
        len(train_loader),
        warmup_epochs=0,
    )

    to_restore = {"epoch": 0, "teacher_ft_stats": teacher_ft_stats}
    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if fp16_scaler is not None and "fp16_scaler" in checkpoint:
            fp16_scaler.load_state_dict(checkpoint["fp16_scaler"])
        to_restore["epoch"] = checkpoint["epoch"]
        to_restore["teacher_ft_stats"] = checkpoint["teacher_ft_stats"]
        logger.info(f"Resuming training from epoch {to_restore['epoch']}")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
    
    start_epoch = to_restore["epoch"]
    teacher_ft_stats = to_restore["teacher_ft_stats"]

    logger.info("Training starts ...")
    start_time = time.time()
    
    aggregation_parameter = args.aggregation_parameter.copy()


    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
          train_loader.sampler.set_epoch(epoch)
        
        # aggiorna solo se richiesto
        if args.aggregation_scheduler:
            aggregation_parameter = utils.scheduler_aggregation_weights(epoch, aggregation_parameter)
            logger.info(f"aggregation_parameter = {aggregation_parameter}")
            
        else:
            aggregation_parameter = args.aggregation_parameter


        train_one_epoch(
            model,
            teachers,
            teacher_ft_stats,
            aggregator,
            train_loader,
            optimizer,
            epoch,
            fp16_scaler,
            ext_logger,
            args,
            aggregation_parameter  # passa quello aggiornato
        )

        evaluate(
            model,
            teachers,
            teacher_ft_stats,
            aggregator,
            val_loader,
            epoch,
            ext_logger,
            args,
        )

        save_dict = {
            "model": model.state_dict(),
            "teacher_ft_stats": teacher_ft_stats,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()

        
        torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckpt_freq and epoch % args.saveckpt_freq == 0:
            shutil.copy(
                os.path.join(args.output_dir, "checkpoint.pth"),
                os.path.join(args.output_dir, f"checkpoint_{epoch:04}.pth"),
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


def train_one_epoch(
    model,
    teachers,
    teacher_ft_stats,
    aggregator,
    data_loader,
    optimizer,
    epoch,
    fp16_scaler,
    ext_logger,
    args,
    aggregation_parameter
):
    logger.info("-" * 50)
    logger.info("Starting training epoch {}".format(epoch))
    logger.info(torch.cuda.memory_summary())

    metrics_file = os.path.join(args.output_dir, "metrics_training.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training - Epoch: [{}/{}]".format(epoch, args.epochs)

    model.train()
    if aggregator:
        aggregator.train()

    for it, sample in enumerate(
        metric_logger.log_every(
            data_loader,
            40,
            header,
            start_iteration=epoch * len(data_loader),
            ext_logger=ext_logger,
            ext_logger_prefix="train/batch/",
        )
    ):

        image = sample["image"]
        target = sample["label"]

        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        it = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = args.lr_schedule[it]
            if i == 0:
                param_group["weight_decay"] = args.wd

        metric_dict = {
            "lr": optimizer.param_groups[0]["lr"],
            "wd": optimizer.param_groups[0]["weight_decay"],
            "tnorm_ema_momentum": args.tnorm_ema_schedule[it],
        }

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            if args.in_chans == 9 and "DinoV2Large_baseline" in args.teachers:
                student_output = model(image[:, [1,2,3,4,5,6,7,10,11], :, :])
                student_output = {
                    f"DinoV2Large_baseline_{k}": v for k, v in student_output.items()
                }     
            else:
                student_output = model(image) 
                  

            
            teacher_output = get_teacher_output(
                image, teachers, teacher_ft_stats, args.tnorm_ema_schedule[it], args.Teacher_strategy, aggregation_parameter, aggregator=aggregator
            )
            
            
            loss, _ = unic_loss(
                student_output,
                teacher_output,
                args.lam_lcos,
                args.lam_lsl1,
                args.t_drop_prob,
                metric_dict=metric_dict,
            )
            

        if not math.isfinite(loss.item()):
            logger.info("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        grad_norms = None
        if fp16_scaler is None:
            loss.backward()

            if args.clip_grad > 0:
                grad_norms = utils.clip_gradients(model, args.clip_grad)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad > 0:
                fp16_scaler.unscale_(optimizer)
                grad_norms = utils.clip_gradients(model, args.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

            #logger.info("Updating teacher with EMA: 'DinoV2Large'")

            # ------------- EMA UPDATE SOLO DEL BACKBONE (escludendo patch_embed) ----------------
            with torch.no_grad():
                m = args.tnorm_ema_schedule[it]
                if "DinoV2Large" in args.teachers:
                    name = "DinoV2Large"
                    utils.ema_update_model(
                        teachers[name].backbone,
                        model.module.encoder,
                        decay=m,
                        skip_prefixes=("patch_embed",),
                    )
                    
                    utils.ema_update_model(
                        teachers[name].agg_lp,
                        model.module.lp,
                        decay=m,
                    )


                elif "DinoV2Large_baseline" in args.teachers:
                    name = "DinoV2Large_baseline"
                    utils.ema_update_model(
                        teachers[name].encoder,
                        model.module.encoder,
                        decay=m,
                        skip_prefixes=("patch_embed",),
                    )
                    utils.ema_update_model(
                        teachers[name].agg_lp,
                        model.module.agg_lp,
                        decay=m,
                    )
                
            for n, p in teachers[name].named_parameters():
                if p.requires_grad and p.grad is not None:
                    logger.warning("ERRORE:", n, "ha gradiente!")



        if grad_norms is not None:
            metric_dict.update(
                {
                    "grad_norm/mean": grad_norms.mean().item(),
                    "grad_norm/std": grad_norms.std().item(),
                    "grad_norm/max": grad_norms.max().item(),
                    "grad_norm/min": grad_norms.min().item(),
                }
            )

    metric_logger.synchronize_between_processes()
    metric_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    logger.info("Averaged train stats:")
    for k, v in metric_dict.items():
        logger.info("{}: {}".format(k, v))

    ext_logger.log(
        metric_dict,
        epoch + 1,
        prefix="train/epoch/",
        save_path=os.path.join(args.output_dir, "log_train.txt"),
    )

    return metric_dict


@torch.no_grad()
def evaluate(
    model,
    teachers,
    teacher_ft_stats,
    aggregator,
    data_loader,
    epoch,
    ext_logger,
    args,
):
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test - Epoch: [{}/{}]".format(epoch, args.epochs)

    model.eval()
    if aggregator:
        aggregator.eval()

    for it, sample in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):
        image = sample["image"]
        target = sample["label"]

        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        student_output = model(image)
        teacher_output = get_teacher_output(image, teachers, teacher_ft_stats, 0.0, args.Teacher_strategy, args.aggregation_parameter, aggregator)

        metric_dict = {}
        unic_loss(
            student_output,
            teacher_output,
            args.lam_lcos,
            args.lam_lsl1,
            args.t_drop_prob,
            metric_dict=metric_dict,
        )
        metric_logger.update(**metric_dict)

    metric_logger.synchronize_between_processes()
    metric_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    logger.info("Averaged test stats:")
    for k, v in metric_dict.items():
        logger.info("{}: {}".format(k, v))

    ext_logger.log(
        metric_dict,
        epoch + 1,
        prefix="test/epoch/",
        save_path=os.path.join(args.output_dir, "log_test.txt"),
    )
    return metric_dict




if __name__ == "__main__":
    args = get_args()
    main(args)
