##################################################
# Utility routines (functions and classes) used for training models.
# Some of these routines are re-used from
# - DINO (https://github.com/facebookresearch/dino)
# - MoCo (https://github.com/facebookresearch/moco)
# - PyTorch examples (https://github.com/pytorch/examples)
##################################################

import argparse
import os
import random
import sys
import logging
import json
import pickle
from enum import Enum

import numpy as np
import torch
import torch.distributed as dist

from dinov2 import distributed


logger = logging.getLogger()


def save_pickle(obj, save_path):
    with open(save_path, "wb") as fid:
        pickle.dump(obj, fid)


def load_pickle(save_path):
    with open(save_path, "rb") as fid:
        obj = pickle.load(fid)
    return obj


def bool_flag(s):
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag")


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])

        if "RANK" in os.environ:
            args.rank = int(os.environ["RANK"])
        elif "SLURM_PROCID" in os.environ:
            args.rank = int(os.environ["SLURM_PROCID"])
        else:
            print("Cannot find rank in environment variables")
            sys.exit(-1)

        n_gpus_per_node = torch.cuda.device_count()
        assert n_gpus_per_node > 0, "No GPU device detected"

        args.gpu = args.rank - n_gpus_per_node * (args.rank // n_gpus_per_node)

    # launched naively with "python main.py"
    elif torch.cuda.is_available():
        print("==> Will run the code on one GPU.")
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "12345"

    else:
        print("==> Does not support training without GPU.")
        sys.exit(1)

    print(
        "=> WORLD_SIZE={}, RANK={}, GPU={}, MASTER_ADDR={}, MASTER_PORT={}, INIT_METHOD={}".format(
            args.world_size,
            args.rank,
            args.gpu,
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
            args.dist_url,
        ),
        flush=True,
    )

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    torch.cuda.set_device(args.gpu)
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def fix_random_seeds(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_program_info(args):
    logger.info("Args:")
    for k, v in sorted(dict(vars(args)).items()):
        logger.info("\t{}: {}".format(k, str(v)))

    with open(os.path.join(args.output_dir, "args.json"), "w") as fp:
        json.dump(
            dict(vars(args)),
            fp,
            indent=4,
            sort_keys=True,
        )

    logger.info("Env vars:")
    for env_var in [
        "ONEDAL_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "KMP_AFFINITY",
        "KMP_BLOCKTIME",
        "MYDEBUG",
    ]:
        logger.info("\t{}={}".format(env_var, os.environ.get(env_var, "(unset)")))

    logger.info("Script caller: {}".format(sys.argv[0]))
    for parg in sys.argv[1:]:
        logger.info("\t{}".format(parg))


def save_model_defn(model, save_path):
    fp = open(os.path.join(save_path), "w")
    fp.write("{}".format(model))
    fp.write("\n")
    
    
    modules = {
        k: v for k, v in {
            "model": model,
            "encoder": getattr(model, "encoder", None),
            "lp": getattr(model, "lp", None),
        }.items() if v is not None
    }
    
    for mname, module in modules.items():
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
        fp.write(
            "Number of trainable parameters in {} : {:,}\n".format(mname, trainable)
        )
        fp.write("Number of frozen parameters in {} : {:,}\n".format(mname, frozen))

    fp.flush()
    fp.close()




def ema_update(target_sd, source_sd, momentum, skip_prefix=()):
    for k, v_t in target_sd.items():
        if any(k.startswith(pref) for pref in skip_prefix):
            continue
        v_s = source_sd.get(k, None)
        if v_s is not None:
            v_t.mul_(momentum).add_(v_s, alpha=1.0 - momentum)
            
def ema_update_model(ema_model, model, decay, skip_prefixes=()):
    """
    Esegue EMA update tra `model` -> `ema_model` in-place.
    Puoi saltare parametri con certi prefissi.

    Args:
        ema_model (nn.Module)
        model (nn.Module)
        decay (float): valore EMA (es. 0.99)
        skip_prefixes (tuple[str]): prefissi dei parametri da saltare
    """
    for (name_ema, param_ema), (name_model, param_model) in zip(
        ema_model.named_parameters(), model.named_parameters()
    ):
        if any(name_ema.startswith(p) for p in skip_prefixes):
            continue
        if param_ema.data.shape != param_model.data.shape:
            continue  # skip parametri non allineati
        param_ema.data.mul_(decay).add_(param_model.data, alpha=1 - decay)

    for (name_ema, buffer_ema), (name_model, buffer_model) in zip(
        ema_model.named_buffers(), model.named_buffers()
    ):
        if any(name_ema.startswith(p) for p in skip_prefixes):
            continue
        if buffer_ema.data.shape != buffer_model.data.shape:
            continue
        buffer_ema.data.copy_(buffer_model.data)


def get_params_groups(model, save_file_path=None):
    """
    Returns two parameters group, one for regularized parameters with weight decay,
    and another for unregularized parameters.
    """
    regularized = []
    not_regularized = []

    fp = None
    if save_file_path is not None:
        fp = open(save_file_path, "w")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.endswith(".bias") or len(param.shape) == 1:
            regstat = "Not Regularized"
            not_regularized.append(param)
        else:
            regstat = "Regularized"
            regularized.append(param)

        if fp is not None:
            fp.write("{} - {} - {}\n".format(name, list(param.shape), regstat))

    if fp is not None:
        fp.flush()
        fp.close()

    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    """
    Creates a cosine scheduler with linear warm-up.
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def clip_gradients(model, clip):
    norms = []
    for _, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(p=2)
            norms.append(param_norm)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return torch.stack(norms)


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    if not os.path.isfile(ckp_path):
        return
    logger.info("Found checkpoint at {}".format(ckp_path))
    checkpoint = torch.load(ckp_path, map_location="cpu")

    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                logger.info(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    logger.info(
                        "=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path)
                    )
                except ValueError:
                    logger.info(
                        "=> failed to load '{}' from checkpoint: '{}'".format(
                            key, ckp_path
                        )
                    )
        else:
            logger.info(
                "=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path)
            )

    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                var = checkpoint[var_name]
                var = move_tensors_to_cuda(var)
                run_variables[var_name] = var


def move_tensors_to_cuda(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = move_tensors_to_cuda(value)
    elif isinstance(obj, (list, tuple)):
        obj = [move_tensors_to_cuda(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        obj = obj.cuda()
    return obj


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]


def standard_normalize(data, mean_ema=None, std_ema=None, ema_momentum=0.1, eps=1e-6):
    """
    Applies standard normalization to the input tensor.
    Data can be either a 2D or 3D tensor.
    """
    
    ndims = len(data.shape)
    assert ndims in (2, 3), "Data must be either 2D or 3D, received: {}".format(ndims)
    all_data = concat_all_gather(data.contiguous())

    # Compute mean and std over the first dimension.
    # If data is 3D, then compute the mean and std
    # over the first two dimensions.
    dims = [0]
    if ndims == 3:
        dims.append(1)

    mean = all_data.mean(dim=dims, keepdim=True) #prima True
    std = all_data.std(dim=dims, keepdim=True) + eps #prima True
    
    if mean_ema is not None and mean_ema.shape != mean.shape:
        mean_ema = mean_ema.view(mean.shape)
        std_ema  = std_ema.view(std.shape)

    if mean_ema is None:
        data = (data - mean) / std
    else:
        assert mean_ema.shape == mean.shape
        assert std_ema.shape == std.shape
        data = (data - mean_ema) / (std_ema + eps)
        mean_ema.copy_(mean_ema * (1 - ema_momentum) + mean * ema_momentum)
        std_ema.copy_(std_ema * (1 - ema_momentum) + std * ema_momentum)

    return data


def scheduler_aggregation_weights(epoch, aggregation_parameter, warmup=30, total_epochs=200):
    """
    Brutal step scheduler:
    - Starts with beta = 0.1, alpha = 0.9
    - Every 2 epochs, alpha increases by 0.1 and beta decreases by 0.1
    - Capped to alpha=0.9, beta=0.1
    """
    #step = min(epoch // 5, 8)  # step ? [0, 8] ? alpha ? [0.1, 0.9]
    alpha = aggregation_parameter.get("alpha", 0.5)
    if epoch < warmup:
        return {"alpha": alpha, "beta": 1-alpha}
    prog = (epoch - warmup) / (total_epochs - warmup)
    beta = (1-alpha) + prog * (alpha - (1-alpha))  # da 0.1 ? 0.9
    beta = min(beta, 0.9)
    return {"alpha": 1.0 - beta, "beta": beta}


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not distributed.is_enabled():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        logging.info(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
        
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)

    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float64, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.double()

