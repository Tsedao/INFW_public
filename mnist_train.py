import tyro
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from typing import Tuple, Optional, Dict, Any

import matplotlib.pyplot as plt

import numpy as np
from dataclasses import dataclass

from mnist_model import MLP
from anytime_infw import AnytimeINFWConfig, AnytimeINFWResult, anytime_infw
from ingd import INGDConfig, INGDResult, ingd

from baselines import SGDConfig, SGDResult, sgd
from baselines import StochasticINGDConfig, StochasticINGDResult, stochastic_ingd
from baselines import LipschitzMinimizationConfig, LipschitzMinimizationResult, lipschitz_minimization

from oracle import Oracle, NNOracle


from plot_utils import (
    plot_f_oracle_calls, 
    plot_grad_norm_oracle_calls, 
    plot_acc_oracle_calls
)


# -------------------------
# Evaluation helpers
# -------------------------
@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def init_mnist_dataloader(
    *,
    batch_size: int = 256,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device.type=="cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=4096, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))
    return train_loader, test_loader

def init_mnist_model_and_oracle(
    *,
    hidden: int = 256,
    stochastic: bool = True,
    device: torch.device,
    train_loader: DataLoader,
) -> Tuple[nn.Module, Oracle]:
    model = MLP(input_dim=28*28, hidden_dim=hidden, output_dim=10)
    oracle = NNOracle(model, train_loader, device=device, stochastic=stochastic)
    return model, oracle
# -------------------------
# Driver: run INFW on MNIST
# -------------------------
def run_infw_mnist(
    *,
    hidden: int = 256,
    batch_size: int = 256,
    stochastic: bool = False,
    T: int = 50,
    epsilon_1: float = 1e-1,
    seed: int = 0,
    device_str: Optional[str] = None,
) -> Tuple[Dict[str, Any], AnytimeINFWResult]:
    

    # Device
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = init_mnist_dataloader(batch_size=batch_size, device=device)
    model, oracle = init_mnist_model_and_oracle(hidden=hidden, stochastic=stochastic, device=device, train_loader=train_loader)
    def callback_wrapper(x_t: np.ndarray) -> Tuple[float, str]:
        acc = evaluate_accuracy(model, test_loader, device)
        return acc, "test_acc"
     # Config
    cfg = AnytimeINFWConfig(
        T=T,
        epsilon_1=epsilon_1,
        seed=seed,
        max_inner_iters=20_000,  # can be large; mninfw will usually return early
        max_epsilon=20.0
    )
    x1_t = oracle._fatten_params().to("cpu").numpy().astype(np.float32)

    # Run INFW (in numpy param space) and then assign back to model
    out = anytime_infw(x1_t, oracle, config=cfg, callback=callback_wrapper)
    x_T = out.x_T
    oracle._set_params_from_numpy(x_T)

    # Final eval
    train_acc = evaluate_accuracy(model, train_loader, device)
    test_acc  = evaluate_accuracy(model, test_loader, device)

    result = {
        "history": out.history,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "final_params": x_T,
        "device": device_str,
    }
    print(f"INFW: Final train acc: {train_acc:.4f} | test acc: {test_acc:.4f}")
    return result, out


def run_ingd_mnist(
    *,
    hidden: int = 256,
    batch_size: int = 256,
    stochastic: bool = False,
    T: int = 100,
    epsilon_t: float = 1e-1,
    delta: float = 1e-3,
    lip: float = 1.0,
    seed: int = 0,
    device_str: Optional[str] = None,
) -> Tuple[Dict[str, Any], INGDResult]:
    # Device
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = init_mnist_dataloader(batch_size=batch_size, device=device)
    model, oracle = init_mnist_model_and_oracle(hidden=hidden, stochastic=stochastic, device=device, train_loader=train_loader)
    def callback_wrapper(x_t: np.ndarray) -> Tuple[float, str]:
        acc = evaluate_accuracy(model, test_loader, device)
        return acc, "test_acc"

    cfg = INGDConfig(
        T=T,
        epsilon_t=epsilon_t,
        delta=delta,
        seed=seed,
        Lip=lip,
    )
    x1_t = oracle._fatten_params().to("cpu").numpy().astype(np.float32)
    out = ingd(x1_t, oracle, config=cfg, callback=callback_wrapper)
    x_T = out.x_T
    oracle._set_params_from_numpy(x_T)

    # Final eval
    train_acc = evaluate_accuracy(model, train_loader, device)
    test_acc  = evaluate_accuracy(model, test_loader, device)

    result = {
        "history": out.history,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "final_params": x_T,
        "device": device_str,
    }
    print(f"INGD: Final train acc: {train_acc:.4f} | test acc: {test_acc:.4f}")
    return result, out


def run_sgd_mnist(
    *,
    hidden: int = 256,
    batch_size: int = 256,
    stochastic: bool = False,
    T: int = 100,
    lr: float = 0.01,
    seed: int = 0,
    device_str: Optional[str] = None,
) -> Tuple[Dict[str, Any], SGDResult]:
    # Device
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = init_mnist_dataloader(batch_size=batch_size, device=device)
    model, oracle = init_mnist_model_and_oracle(hidden=hidden, stochastic=stochastic, device=device, train_loader=train_loader)
    def callback_wrapper(x_t: np.ndarray) -> Tuple[float, str]:
        acc = evaluate_accuracy(model, test_loader, device)
        return acc, "test_acc"
    
    cfg = SGDConfig(
        T=T,
        lr=lr,
    )
    x1_t = oracle._fatten_params().to("cpu").numpy().astype(np.float32)
    out = sgd(x1_t, oracle, config=cfg, callback=callback_wrapper)
    x_T = out.x_T
    oracle._set_params_from_numpy(x_T)

    # Final eval
    train_acc = evaluate_accuracy(model, train_loader, device)
    test_acc  = evaluate_accuracy(model, test_loader, device)

    result = {
        "history": out.history,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "final_params": x_T,
        "device": device_str,
    }
    print(f"SGD: Final train acc: {train_acc:.4f} | test acc: {test_acc:.4f}")
    return result, out

    
    

def run_stochastic_ingd_mnist(
    *,
    hidden: int = 256,
    batch_size: int = 256,
    stochastic: bool = False,
    T: int = 100,
    p: float = 0.01,
    q: float = 0.01,
    beta: float = 0.01,
    seed: int = 0,
    device_str: Optional[str] = None,
) -> Tuple[Dict[str, Any], StochasticINGDResult]:
    # Device
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = init_mnist_dataloader(batch_size=batch_size, device=device)
    model, oracle = init_mnist_model_and_oracle(hidden=hidden, stochastic=stochastic, device=device, train_loader=train_loader)
    def callback_wrapper(x_t: np.ndarray) -> Tuple[float, str]:
        acc = evaluate_accuracy(model, test_loader, device)
        return acc, "test_acc"
    
    cfg = StochasticINGDConfig(
        T=T,
        p=p,
        q=q,
        beta=beta,
        seed=seed,
    )
    x1_t = oracle._fatten_params().to("cpu").numpy().astype(np.float32)
    out = stochastic_ingd(x1_t, oracle, config=cfg, callback=callback_wrapper)
    x_T = out.x_T
    oracle._set_params_from_numpy(x_T)

    # Final eval
    train_acc = evaluate_accuracy(model, train_loader, device)
    test_acc  = evaluate_accuracy(model, test_loader, device)

    result = {
        "history": out.history,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "final_params": x_T,
        "device": device_str,
    }
    print(f"Stochastic INGD: Final train acc: {train_acc:.4f} | test acc: {test_acc:.4f}")
    return result, out


def run_lipschitz_minimization_mnist(
    *,
    hidden: int = 256,
    batch_size: int = 256,
    stochastic: bool = False,
    T: int = 100,
    tolerance: float = 1e-6,
    L: float = 1.0,
    beta: float = 1.0,
    seed: int = 0,
    device_str: Optional[str] = None,
) -> Tuple[Dict[str, Any], LipschitzMinimizationResult]:
    # Device
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = init_mnist_dataloader(batch_size=batch_size, device=device)
    model, oracle = init_mnist_model_and_oracle(hidden=hidden, stochastic=stochastic, device=device, train_loader=train_loader)
    def callback_wrapper(x_t: np.ndarray) -> Tuple[float, str]:
        acc = evaluate_accuracy(model, test_loader, device)
        return acc, "test_acc"

    cfg = LipschitzMinimizationConfig(
        T=T,
        tolerance=tolerance,
        L=L,
        beta=beta,
        seed=seed,
    )
    x1_t = oracle._fatten_params().to("cpu").numpy().astype(np.float32)
    out = lipschitz_minimization(x1_t, oracle, config=cfg, callback=callback_wrapper)
    x_T = out.x_T
    oracle._set_params_from_numpy(x_T)

    # Final eval
    train_acc = evaluate_accuracy(model, train_loader, device)
    test_acc  = evaluate_accuracy(model, test_loader, device)

    result = {
        "history": out.history,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "final_params": x_T,
        "device": device_str,
    }
    print(f"Lipschitz Minimization: Final train acc: {train_acc:.4f} | test acc: {test_acc:.4f}")
    return result, out


@dataclass
class RunConfig:
    hidden: int = 256
    batch_size: int = 256
    stochastic: bool = True
    
    ## INFW  ############################################
    infw_T: int = 250
    infw_epsilon_1: float = 0.4
    #####################################################

    ## INGD  ############################################
    ingd_T: int = 2000
    ingd_epsilon_t: float = 1e-2
    ingd_delta: float = 1e-1
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
    

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    cfg = tyro.cli(RunConfig)
    if cfg.lip_min:
        _, lip_min_out = run_lipschitz_minimization_mnist(
            T=cfg.lip_min_T,
            tolerance=cfg.lip_min_tolerance,
            L=cfg.lip_min_L,
            beta=cfg.lip_min_beta,
            batch_size=cfg.batch_size,
            stochastic=cfg.stochastic,
            seed=cfg.seed,
        )
        save_name = f"results/mnist_example_stochastic{str(cfg.stochastic) + '_bs' + str(cfg.batch_size) if cfg.stochastic else 'False'}"
        save_name += f"_lip_min_T_{cfg.lip_min_T}_L_{cfg.lip_min_L}_beta_{cfg.lip_min_beta}.pkl"
        with open(save_name, "wb") as f:
            pickle.dump(lip_min_out, f)
    if cfg.infw:
        _, infw_out = run_infw_mnist(
            T=cfg.infw_T,
            epsilon_1=cfg.infw_epsilon_1,
            batch_size=cfg.batch_size,
            stochastic=cfg.stochastic,
            seed=cfg.seed,
        )
        with open(f"results/mnist_example_stochastic{str(cfg.stochastic) + '_bs' + str(cfg.batch_size) if cfg.stochastic else 'False'}_anytime_infw_epsilon_1_{cfg.infw_epsilon_1}_T_{cfg.infw_T}.pkl", "wb") as f:
            pickle.dump(infw_out, f)
    ## only call this when we use stochastic INGD implemented by Jinzhao
    if cfg.ingd_jz:
        _, ingd_out = run_stochastic_ingd_mnist(
            T=cfg.ingd_jz_T,
            p=cfg.ingd_jz_p,
            q=cfg.ingd_jz_q,
            beta=cfg.ingd_jz_beta,
            batch_size=cfg.batch_size,
            stochastic=cfg.stochastic,
            seed=cfg.seed,
        )
        with open(f"results/mnist_example_stochastic{str(cfg.stochastic) + '_bs' + str(cfg.batch_size) if cfg.stochastic else 'False'}_stochastic_ingd_p_{cfg.ingd_jz_p}_q_{cfg.ingd_jz_q}_beta_{cfg.ingd_jz_beta}_T_{cfg.ingd_jz_T}.pkl", "wb") as f:
            pickle.dump(ingd_out, f)
    if cfg.sgd:
        _, sgd_out = run_sgd_mnist(
            T=cfg.sgd_T,
            lr=cfg.sgd_lr,
            batch_size=cfg.batch_size,
            stochastic=cfg.stochastic,
            seed=cfg.seed,
        )
        with open(f"results/mnist_example_stochastic{str(cfg.stochastic) + '_bs' + str(cfg.batch_size) if cfg.stochastic else 'False'}_sgd_lr_{cfg.sgd_lr}_T_{cfg.sgd_T}.pkl", "wb") as f:
            pickle.dump(sgd_out, f)
    if cfg.ingd:
        _, ingd_out = run_ingd_mnist(
            T=cfg.ingd_T,
            epsilon_t=cfg.ingd_epsilon_t,
            delta=cfg.ingd_delta,
            lip=cfg.ingd_lip,
            batch_size=cfg.batch_size,
            stochastic=cfg.stochastic,
            seed=cfg.seed,
        )
        with open(f"results/mnist_example_stochastic{str(cfg.stochastic) + '_bs' + str(cfg.batch_size) if cfg.stochastic else 'False'}_ingd_ml_delta_{cfg.ingd_delta}_lip_{cfg.ingd_lip}_T_{cfg.ingd_T}.pkl", "wb") as f:
            pickle.dump(ingd_out, f)
 