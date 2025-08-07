import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse
import numpy as np
import os
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    label_ranking_loss,
    label_ranking_average_precision_score,
    classification_report,
    precision_recall_fscore_support
)

from Dataset import carica_dati
from utils import print_program_info


from teachers.ScaleMae import ScaleMAE
from teachers.ViT import ViT  # Assicurati che questo modulo sia disponibile
from teachers.Grouping import GroupIng



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_classes",
        type=int,
        default=19,
        help="Num of classes for classification of BigEarthNet.",
    ) 
    parser.add_argument(
        "--bands",
        type=str,
        default="s2",
        help="Multispectral bands of BigEarthNet.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for the finetuning.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="vit_base",
        help="ViT architecture to use when selecting the ViT baseline.",
    )
    parser.add_argument(
        "--finetuning_bands",  # <-- aggiunta la 'e'
        default="rgb",
        type=str,
        help="Subset of bands to use for fine-tuning",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="D:/tesi_carlotta/data",
        help="Path for the dataset directory --default: local dir",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--use_weight",
        default=False,
        type=bool,
        help="Weight the label of the loss",
    )
    parser.add_argument(
        "--transform",
        default=False,
        type=bool,
        help="Do augmentations",
    )
    parser.add_argument(
        "--finetuning",
        default=False,
        type=bool,
        help="Do augmentations",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="D:/tesi_carlotta/checkpoints/scalemae_RGB/best-checkpoint-giusto.ckpt",
        type=str,
        help="Path of the checkpoint to load",
    )
    parser.add_argument(
        "--output_dir",
        default="output/risultati",
        type=str,
        help="Output path",
    )
    parser.add_argument(
        "--image_size",
        default=224,
        type=int,
        help="Output path",
    )
    
    parser.add_argument(
        "--in_chans",
        type=int,
        default=12,  # adjust accordingly: 3 for RGB or 12 for multispectral data
        help="Number of input channels for the model (default: 12).",
    )

    # Nuovo argomento per scegliere il modello
    parser.add_argument(
        "--model",
        type=str,
        default="scalemae",
        choices=["scalemae", "vit"],
        help="Select model type for infer: 'scalemae' or 'vit'",
    )

    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Select model type for infer: 'scalemae' or 'vit'",
    )

    args = parser.parse_args()
    return args

def get_metrics(labels, predictions, probabilities):
    """
    Calcola un set completo di metriche multilabel: Recall, F2-score (micro, macro, sample),
    Hamming loss, One-error, Coverage, Ranking loss, LRAP.
    """
    metrics = {}

    # Basic counts
    true_positive = np.logical_and(predictions == 1, labels == 1).sum(axis=0)
    false_positive = np.logical_and(predictions == 1, labels == 0).sum(axis=0)
    false_negative = np.logical_and(predictions == 0, labels == 1).sum(axis=0)

    # Recall
    macro_recall = np.mean([
        tp / (tp + fn) if (tp + fn) > 0 else 0
        for tp, fn in zip(true_positive, false_negative)
    ])
    metrics["R_macr"] = macro_recall * 100

    sample_recall = np.mean([
        np.sum(np.logical_and(p == 1, l == 1)) / (np.sum(l) + 1e-10)
        for p, l in zip(predictions, labels)
    ])
    metrics["R_smpl"] = sample_recall * 100

    micro_tp = true_positive.sum()
    micro_fn = false_negative.sum()
    micro_recall = micro_tp / (micro_tp + micro_fn + 1e-10)
    metrics["R_micr"] = micro_recall * 100

    # F2-score (beta=2)
    def fbeta(precision, recall, beta=2):
        beta2 = beta ** 2
        return (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-10)

    # Precision
    sample_precision = np.mean([
        np.sum(np.logical_and(p == 1, l == 1)) / (np.sum(p) + 1e-10)
        for p, l in zip(predictions, labels)
    ])
    sample_f2 = fbeta(sample_precision, sample_recall)
    metrics["F2_smpl"] = sample_f2 * 100

    micro_fp = false_positive.sum()
    micro_precision = micro_tp / (micro_tp + micro_fp + 1e-10)
    micro_f2 = fbeta(micro_precision, micro_recall)
    metrics["F2_micr"] = micro_f2 * 100

    macro_f2 = np.mean([
        fbeta(tp / (tp + fp + 1e-10), tp / (tp + fn + 1e-10))
        if (tp + fp > 0 and tp + fn > 0) else 0
        for tp, fp, fn in zip(true_positive, false_positive, false_negative)
    ])
    metrics["F2_macr"] = macro_f2 * 100

    # Hamming loss
    metrics["HL"] = hamming_loss(labels, predictions) * 100

    # One-error
    top_pred = np.argmax(probabilities, axis=1)
    one_error = 1 - np.mean([labels[i, top_pred[i]] for i in range(len(labels))])
    metrics["OE"] = one_error * 100

    # Coverage
    sorted_indices = np.argsort(-probabilities, axis=1)
    ranks = np.zeros_like(labels)
    for i in range(labels.shape[0]):
        ranks[i, sorted_indices[i]] = np.arange(labels.shape[1])
    coverage = np.max(ranks * labels, axis=1).mean()
    metrics["COV"] = coverage

    # Ranking Loss
    rl = label_ranking_loss(labels, probabilities)
    metrics["RL"] = rl * 100

    # LRAP
    lrap = label_ranking_average_precision_score(labels, probabilities)
    metrics["LRAP"] = lrap * 100

    return metrics


import re, torch

def main(args):
    # Carica il checkpoint del modello
    checkpoint_path = args.checkpoint_path
    args.checkpoint_path = None
    if args.model.lower() == "scalemae":
        model = ScaleMAE.load_from_checkpoint(checkpoint_path, args=args, strict=True)
    elif args.model.lower() == "vit":
        if args.arch.lower() == "grouping":
            model = GroupIng(args)
            ckpt = torch.load(checkpoint_path)
            sd = ckpt.get("state_dict", ckpt)          # lightning o plain
            # ----- 1. Detect prefix -------------------------------------------------
            encoder_prefix = None
            for p in ("module.encoder.", "encoder."):
                if any(k.startswith(p) for k in sd):
                    encoder_prefix = p
                    break
            if encoder_prefix is None:
                raise RuntimeError("Nessun prefisso encoder.* trovato nel checkpoint")
        
            # ----- 2. Encoder state_dict senza prefisso -----------------------------
            enc_sd = {re.sub(f"^{encoder_prefix}", "", k): v
                      for k, v in sd.items() if k.startswith(encoder_prefix)}
            miss, unexp = model.encoder.load_state_dict(enc_sd, strict=True)
            print(f"[ENCODER] loaded from {checkpoint_path}")
            print("   missing :", len(miss), " | unexpected :", len(unexp))
        
            # ----- 3. Linear head ----------------------------------------------------
            if "classifier.weight" in sd:
                w, b = sd["classifier.weight"], sd["classifier.bias"]
                if w.shape == model.classifier.weight.shape:
                    model.classifier.load_state_dict(
                        {"weight": w, "bias": b}, strict=True
                    )
                    print("[HEAD]   classifier caricato con shape", tuple(w.shape))
                else:
                    print(f"[HEAD]   shape mismatch ({tuple(w.shape)} ? "
                          f"{tuple(model.classifier.weight.shape)}). "
                          "Head lasciato random.")
            else:
                print("[HEAD]   pesi head assenti")

        else: 
            model = ViT(args)
            ckpt = torch.load(checkpoint_path)
            sd = ckpt.get("state_dict", ckpt)          # lightning o plain
            # ----- 1. Detect prefix -------------------------------------------------
            encoder_prefix = None
            for p in ("module.encoder.", "encoder."):
                if any(k.startswith(p) for k in sd):
                    encoder_prefix = p
                    break
            if encoder_prefix is None:
                raise RuntimeError("Nessun prefisso encoder.* trovato nel checkpoint")
        
            # ----- 2. Encoder state_dict senza prefisso -----------------------------
            enc_sd = {re.sub(f"^{encoder_prefix}", "", k): v
                      for k, v in sd.items() if k.startswith(encoder_prefix)}
            miss, unexp = model.encoder.load_state_dict(enc_sd, strict=True)
            print(f"[ENCODER] loaded from {checkpoint_path}")
            print("   missing :", len(miss), " | unexpected :", len(unexp))
        
            # ----- 3. Linear head ----------------------------------------------------
            if "classifier.weight" in sd:
                w, b = sd["classifier.weight"], sd["classifier.bias"]
                if w.shape == model.classifier.weight.shape:
                    model.classifier.load_state_dict(
                        {"weight": w, "bias": b}, strict=True
                    )
                    print("[HEAD]   classifier caricato con shape", tuple(w.shape))
                else:
                    print(f"[HEAD]   shape mismatch ({tuple(w.shape)} ? "
                          f"{tuple(model.classifier.weight.shape)}). "
                          "Head lasciato random.")
            else:
                print("[HEAD]   pesi head assenti")
            
    else:
        raise ValueError("Invalid model type specified.")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print("-- Caricato il modello:", args.model)

    # Carica il dataset di test
    test_loader = carica_dati(args, setup="test")
    print("-- Caricati i dati di test")

    # Inferenza
    all_preds = []
    all_probs = []
    all_labels = []

    for batch in test_loader:
        for k in batch:
            batch[k] = batch[k].to(device)

        with torch.no_grad():
            logits = model(batch["image"])
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calcolo metriche
    metrics = get_metrics(all_labels, all_preds, all_probs)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    class_report = classification_report(
        all_labels,
        all_preds,
        target_names=[f"Class_{i}" for i in range(all_labels.shape[1])],
        zero_division=0
    )

    # Costruzione stringa di report
    report_lines = []
    report_lines.append(f"== Risultati per modello: {args.model.upper()} ==\n")

    report_lines.append("== Metriche Globali ==")
    for k, v in metrics.items():
        report_lines.append(f"{k}: {v:.3f}")

    report_lines.append("\n== Report per Classe ==")
    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
        report_lines.append(f"Classe {i:02d} | Precision: {p:.3f} | Recall: {r:.3f} | F1: {f:.3f} | Support: {s}")

    report_lines.append("\n== Classification Report Esteso ==")
    report_lines.append(class_report)
    
    print(report_lines)

    # Salvataggio su file
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, f"report_{args.model.lower()}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        for line in report_lines:
            f.write(line + "\n")

    print(f"Report salvato in: {report_path}")


if __name__ == '__main__':
    args = get_args()
    print_program_info(args)
    main(args)
