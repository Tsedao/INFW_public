
# experiment_runner.py (registry-enabled)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, Tuple, Optional
import numpy as np
import torch

from registry import TaskSpec, AlgoSpec, get_task, get_algo
# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def _seed_and_device(seed: int, device_str: Optional[str]) -> torch.device:
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return device

def _flatten_params_to_numpy(oracle) -> np.ndarray:
    return oracle._fatten_params().to("cpu").numpy().astype(np.float32)

def _assign_numpy_params(oracle, x: np.ndarray) -> None:
    oracle._set_params_from_numpy(x)

# ----------------------------------------------------------------------
# The single unified runner
# ----------------------------------------------------------------------
def run_experiment(
    *,
    task: TaskSpec,
    algo: AlgoSpec,
    seed: int = 0,
    device_str: Optional[str] = None,
    batch_size: int = 256,
    stochastic: bool = False,
    task_kwargs: Optional[Dict[str, Any]] = None,
    config_kwargs: Optional[Dict[str, Any]] = None,
    callback_metric: str = "test_acc",
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Any]:
    device = _seed_and_device(seed, device_str)
    task_kwargs = {**task.default_kwargs, **(task_kwargs or {})}
    config_kwargs = {**algo.default_cfg, **(config_kwargs or {})}
    eval_freq = config_kwargs.get('eval_freq', 1)
    extra = extra or {}

    # Dataloaders
    train_loader, test_loader = task.dataloaders_builder(batch_size=batch_size, **task_kwargs)

    # Model + Oracle
    model, oracle = task.model_oracle_builder(stochastic=stochastic, device=device, train_loader=train_loader, **task_kwargs)

    # Callback uses the task's eval_fn
    def callback_wrapper(x_t: np.ndarray) -> Tuple[float, str]:
        _assign_numpy_params(oracle, x_t)
        acc = task.eval_fn(model, test_loader, device)
        return acc, callback_metric

    # Config for the algorithm
    cfg = algo.config_factory(**config_kwargs)
    print(cfg)

    # Initial params
    x1_t = _flatten_params_to_numpy(oracle)

    # Run algorithm
    out = algo.run_fn(x1_t, oracle, cfg, callback_wrapper,eval_freq=eval_freq)

    # Final params -> assign back
    x_T = out.x_T
    _assign_numpy_params(oracle, x_T)

    # Final evals
    train_acc = task.eval_fn(model, train_loader, device)
    test_acc  = task.eval_fn(model, test_loader, device)

    result = {
        "history": getattr(out, "history", None),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "final_params": x_T,
        "device": str(device),
        "task": task.name,
        "algo": algo.name,
    }
    print(f"{algo.name.upper()} on {task.name}: train acc {train_acc:.4f} | test acc {test_acc:.4f}")
    return result, out

# ----------------------------------------------------------------------
# Concrete task specs
# ----------------------------------------------------------------------


# register_task(TaskSpec(
#     name="MNIST",
#     dataloaders_builder=_mnist_dls,
#     model_oracle_builder=_mnist_model_oracle,
#     eval_fn=evaluate_accuracy,
#     default_kwargs={"hidden": 256},
# ))



# ----------------------------------------------------------------------
# Generic entrypoint + thin convenience wrappers
# ----------------------------------------------------------------------
def run_algo_on_task(
    algo_name: str,
    task_name: str,
    *,
    batch_size: int = 256,
    stochastic: bool = False,
    seed: int = 0,
    device_str: Optional[str] = None,
    task_kwargs: Optional[Dict[str, Any]] = None,
    config_kwargs: Optional[Dict[str, Any]] = None,
    callback_metric: str = "test_acc",
):
    return run_experiment(
        task=get_task(task_name),
        algo=get_algo(algo_name),
        seed=seed,
        device_str=device_str,
        batch_size=batch_size,
        stochastic=stochastic,
        task_kwargs=task_kwargs,
        config_kwargs=config_kwargs,
        callback_metric=callback_metric,
    )

# Backward-compatible helpers (MNIST)
def run_infw_mnist(*, hidden: int = 256, batch_size: int = 256, stochastic: bool = False,
                   seed: int = 0, device_str: Optional[str] = None,
                   T: int = 50, epsilon_1: float = 1e-1):
    return run_algo_on_task(
        "INFW", "MNIST",
        batch_size=batch_size, stochastic=stochastic, seed=seed, device_str=device_str,
        task_kwargs={"hidden": hidden},
        config_kwargs={"T": T, "epsilon_1": epsilon_1, "seed": seed},
    )

def run_ingd_mnist(*, hidden: int = 256, batch_size: int = 256, stochastic: bool = False,
                   seed: int = 0, device_str: Optional[str] = None,
                   T: int = 100, epsilon_t: float = 1e-1, delta: float = 1e-3, lip: float = 1.0):
    return run_algo_on_task(
        "INGD", "MNIST",
        batch_size=batch_size, stochastic=stochastic, seed=seed, device_str=device_str,
        task_kwargs={"hidden": hidden},
        config_kwargs={"T": T, "epsilon_t": epsilon_t, "delta": delta, "Lip": lip, "seed": seed},
    )

# === Thin MNIST wrappers ===
def run_sgd_mnist(*, hidden: int = 256, batch_size: int = 256, stochastic: bool = False,
                  seed: int = 0, device_str: Optional[str] = None,
                  T: int = 100, lr: float = 1e-2):
    return run_algo_on_task(
        "SGD", "MNIST",
        batch_size=batch_size, stochastic=stochastic, seed=seed, device_str=device_str,
        task_kwargs={"hidden": hidden},
        config_kwargs={"T": T, "lr": lr},
    )

def run_stochastic_ingd_mnist(*, hidden: int = 256, batch_size: int = 256, stochastic: bool = False,
                              seed: int = 0, device_str: Optional[str] = None,
                              T: int = 100, p: float = 1e-2, q: float = 1e-2, beta: float = 1e-2):
    return run_algo_on_task(
        "StochasticINGD", "MNIST",
        batch_size=batch_size, stochastic=stochastic, seed=seed, device_str=device_str,
        task_kwargs={"hidden": hidden},
        config_kwargs={"T": T, "p": p, "q": q, "beta": beta, "seed": seed},
    )

def run_lipschitz_minimization_mnist(*, hidden: int = 256, batch_size: int = 256, stochastic: bool = False,
                                     seed: int = 0, device_str: Optional[str] = None,
                                     T: int = 100, tolerance: float = 1e-6, L: float = 1.0, beta: float = 1.0):
    return run_algo_on_task(
        "LipschitzMinimization", "MNIST",
        batch_size=batch_size, stochastic=stochastic, seed=seed, device_str=device_str,
        task_kwargs={"hidden": hidden},
        config_kwargs={"T": T, "tolerance": tolerance, "L": L, "beta": beta, "seed": seed},
    )


