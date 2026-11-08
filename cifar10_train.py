
# cifar10_train.py
# Training script for CIFAR-10 using ResNet-20 and a flexible optimizer/scheduler setup.
# Example:
#   python cifar10_train.py --epochs 200 --opt sgd --lr 0.1 --sched cosine --amp
#
from __future__ import annotations
import argparse
import os
import numpy as np
import time
import tyro
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from cifar10_model import build_resnet20

from anytime_infw import AnytimeINFWConfig, AnytimeINFWResult, anytime_infw
from ingd import INGDConfig, INGDResult, ingd

from baselines import SGDConfig, SGDResult, sgd
from baselines import StochasticINGDConfig, StochasticINGDResult, stochastic_ingd
from baselines import LipschitzMinimizationConfig, LipschitzMinimizationResult, lipschitz_minimization

from oracle import Oracle, NNOracle

from dataclasses import dataclass

from experiment_runner import run_algo_on_task

from registry import TaskSpec, AlgoSpec, register_task, register_algo
import algorithms_builtin





def seed_everything(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # faster on conv nets

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += float(loss.item()) * images.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == targets).sum().item())
        total += images.size(0)
    avg_loss = total_loss / max(1, total)
    top1 = 100.0 * correct / max(1, total)
    return top1
# -------------------------
# Datasets
# -------------------------
def build_dataloaders(data_dir: str, batch_size: int, num_workers: int):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tfms)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tfms)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def build_cifar10_model_and_oracle(
    *,
    stochastic: bool = True,
    device: torch.device,
    train_loader: DataLoader,
    **kwargs: Any,
) -> Tuple[nn.Module, Oracle]:
    model = build_resnet20(num_classes=10).to(device)
    oracle = NNOracle(model, train_loader, device=device, stochastic=stochastic)
    return model, oracle

def run_infw_cifar10(*, batch_size: int = 256, stochastic: bool = False, seed: int = 0,
                     device_str: Optional[str] = None, data_dir: str = "./data",
                     T: int = 50, epsilon_1: float = 1e-1, eval_freq=1):
    return run_algo_on_task(
        "INFW", "CIFAR-10",
        batch_size=batch_size, stochastic=stochastic, seed=seed, device_str=device_str,
        task_kwargs={"data_dir": data_dir},
        config_kwargs={"T": T, "epsilon_1": epsilon_1, "seed": seed, "eval_freq": eval_freq},
    )

def run_ingd_cifar10(*, batch_size: int = 256, stochastic: bool = False, seed: int = 0,
                     device_str: Optional[str] = None, data_dir: str = "./data",
                     T: int = 100, epsilon_t: float = 1e-1, delta: float = 1e-3,
                    lip: float = 1.0, eval_freq=1):
    return run_algo_on_task(
        "INGD", "CIFAR-10",
        batch_size=batch_size, stochastic=stochastic, seed=seed, device_str=device_str,
        task_kwargs={"data_dir": data_dir},
        config_kwargs={"T": T, "epsilon_t": epsilon_t, "delta": delta, "Lip": lip, "seed": seed, "eval_freq": eval_freq},
    )

def run_sgd_cifar10(*, batch_size: int = 256, stochastic: bool = False, seed: int = 0,
                    device_str: Optional[str] = None, data_dir: str = "./data",
                    T: int = 100, lr: float = 1e-2, eval_freq=1):
    return run_algo_on_task(
        "SGD", "CIFAR-10",
        batch_size=batch_size, stochastic=stochastic, seed=seed, device_str=device_str,
        task_kwargs={"data_dir": data_dir},
        config_kwargs={"T": T, "lr": lr, "eval_freq": eval_freq},
    )

def run_stochastic_ingd_cifar10(*, batch_size: int = 256, stochastic: bool = False, seed: int = 0,
                                device_str: Optional[str] = None, data_dir: str = "./data",
                                T: int = 100, p: float = 1e-2, q: float = 1e-2, 
                                beta: float = 1e-2, eval_freq=1):
    return run_algo_on_task(
        "StochasticINGD", "CIFAR-10",
        batch_size=batch_size, stochastic=stochastic, seed=seed, device_str=device_str,
        task_kwargs={"data_dir": data_dir},
        config_kwargs={"T": T, "p": p, "q": q, "beta": beta, "seed": seed, "eval_freq": eval_freq},
    )

def run_lipschitz_minimization_cifar10(*, batch_size: int = 256, stochastic: bool = False, seed: int = 0,
                                       device_str: Optional[str] = None, data_dir: str = "./data",
                                       T: int = 100, tolerance: float = 1e-6, L: float = 1.0,
                                         beta: float = 1.0, eval_freq=1):
    return run_algo_on_task(
        "LipschitzMinimization", "CIFAR-10",
        batch_size=batch_size, stochastic=stochastic, seed=seed, device_str=device_str,
        task_kwargs={"data_dir": data_dir},
        config_kwargs={"T": T, "tolerance": tolerance, "L": L, "beta": beta, "seed": seed, "eval_freq": eval_freq},
    )


@dataclass
## TODO: merge the config of mnisttrain and cifar10_train into one config
class RunConfig:
    hidden: int = 256
    batch_size: int = 1024
    stochastic: bool = True
    eval_freq : int = 20
    
    ## INFW  ############################################
    infw_T: int = 250
    infw_epsilon_1: float = 0.4
    #####################################################

    ## INGD  ############################################
    ingd_T: int = 2000
    ingd_epsilon_t: float = 1e-2
    ingd_delta: float = 1e-2
    ingd_lip: float = 1.0
    #####################################################

    ## SGD  ############################################
    sgd_T: int = 2000
    sgd_lr: float = 0.1
    #####################################################

    ## Jinzhao's Stochastic INGD  ######################
    ingd_jz_T: int = 2000
    ## eta = 1 / (p ||m_t|| + q)
    ingd_jz_p: float = 1e+1
    ingd_jz_q: float = 4e+1
    ingd_jz_beta: float = 0.99
    #####################################################

    ## Lipschitz Minimization KS25's Algorithm 8.4 ######
    lip_min_T: int = 350
    lip_min_tolerance: float = 1e-6
    lip_min_L: float = 1.0
    lip_min_beta: float = 1.0
    #####################################################


    ingd_jz: bool = False
    infw: bool = False
    ingd: bool = False
    sgd: bool = False
    lip_min: bool = False
    seed: int = 0

    data_dir: str = './data'
# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    import os
    import pickle
    save_dir = "results/cifar10"
    os.makedirs(save_dir, exist_ok=True)
    cfg = tyro.cli(RunConfig)

    ### register tasks
    register_task(TaskSpec(
        name="CIFAR-10",
        dataloaders_builder=build_dataloaders,
        model_oracle_builder=build_cifar10_model_and_oracle,
        eval_fn=lambda model, loader, device: evaluate(model, loader, device),
            default_kwargs={"data_dir": "./data", "num_workers": 4},
    ))
    if cfg.lip_min:
        _, lip_min_out = run_lipschitz_minimization_cifar10(
            T=cfg.lip_min_T,
            tolerance=cfg.lip_min_tolerance,
            L=cfg.lip_min_L,
            beta=cfg.lip_min_beta,
            batch_size=cfg.batch_size,
            stochastic=cfg.stochastic,
            seed=cfg.seed,
            eval_freq= cfg.eval_freq,
        )
        save_name = f"{save_dir}/stochastic{str(cfg.stochastic) + '_bs' + str(cfg.batch_size)}"
        save_name += f"_lip_min_T_{cfg.lip_min_T}_L_{cfg.lip_min_L}_beta_{cfg.lip_min_beta}.pkl"
        with open(save_name, "wb") as f:
            pickle.dump(lip_min_out, f)

    if cfg.infw:
        _, infw_out = run_infw_cifar10(
            T=cfg.infw_T,
            epsilon_1=cfg.infw_epsilon_1,
            batch_size=cfg.batch_size,
            stochastic=cfg.stochastic,
            seed=cfg.seed,
            data_dir=cfg.data_dir,
            eval_freq= cfg.eval_freq,
        )
        with open(f"{save_dir}/stochastic{str(cfg.stochastic) + '_bs' + str(cfg.batch_size)}_anytime_infw_epsilon_1_{cfg.infw_epsilon_1}_T_{cfg.infw_T}.pkl", "wb") as f:
            pickle.dump(infw_out, f)

    ## only call this when we use stochastic INGD implemented by Jinzhao
    if cfg.ingd_jz:
        _, ingd_out = run_stochastic_ingd_cifar10(
            T=cfg.ingd_jz_T,
            p=cfg.ingd_jz_p,
            q=cfg.ingd_jz_q,
            beta=cfg.ingd_jz_beta,
            batch_size=cfg.batch_size,
            stochastic=cfg.stochastic,
            seed=cfg.seed,
            eval_freq= cfg.eval_freq,
        )
        with open(f"{save_dir}/stochastic{str(cfg.stochastic) + '_bs' + str(cfg.batch_size)}_stochastic_ingd_p_{cfg.ingd_jz_p}_q_{cfg.ingd_jz_q}_beta_{cfg.ingd_jz_beta}_T_{cfg.ingd_jz_T}.pkl", "wb") as f:
            pickle.dump(ingd_out, f)
    if cfg.sgd:
        _, sgd_out = run_sgd_cifar10(
            T=cfg.sgd_T,
            lr=cfg.sgd_lr,
            batch_size=cfg.batch_size,
            stochastic=cfg.stochastic,
            seed=cfg.seed,
            eval_freq= cfg.eval_freq,
        )
        with open(f"{save_dir}/stochastic{str(cfg.stochastic) + '_bs' + str(cfg.batch_size)}_sgd_lr_{cfg.sgd_lr}_T_{cfg.sgd_T}.pkl", "wb") as f:
            pickle.dump(sgd_out, f)
    if cfg.ingd:
        _, ingd_out = run_ingd_cifar10(
            T=cfg.ingd_T,
            epsilon_t=cfg.ingd_epsilon_t,
            delta=cfg.ingd_delta,
            lip=cfg.ingd_lip,
            batch_size=cfg.batch_size,
            stochastic=cfg.stochastic,
            seed=cfg.seed,
            eval_freq= cfg.eval_freq,
        )
        with open(f"{save_dir}/stochastic{str(cfg.stochastic) + '_bs' + str(cfg.batch_size)}_ingd_ml_delta_{cfg.ingd_delta}_lip_{cfg.ingd_lip}_T_{cfg.ingd_T}.pkl", "wb") as f:
            pickle.dump(ingd_out, f)