# registry.py
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, Tuple
import numpy as np
import torch

@dataclass
class TaskSpec:
    name: str
    dataloaders_builder: Callable[..., Tuple[Any, Any]]
    model_oracle_builder: Callable[..., Tuple[Any, Any]]
    eval_fn: Callable[[Any, Any, torch.device], float]
    default_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlgoSpec:
    name: str
    run_fn: Callable[[np.ndarray, Any, Any, Callable[[np.ndarray], tuple]], Any]
    config_factory: Callable[..., Any]
    default_cfg: Dict[str, Any] = field(default_factory=dict)

_TASKS: Dict[str, TaskSpec] = {}
_ALGOS: Dict[str, AlgoSpec] = {}

def register_task(spec: TaskSpec) -> None:
    _TASKS[spec.name.lower()] = spec  # overwrite is fine/idempotent

def register_algo(spec: AlgoSpec) -> None:
    _ALGOS[spec.name.lower()] = spec  # overwrite is fine/idempotent

def get_task(name: str) -> TaskSpec:
    return _TASKS[name.lower()]

def get_algo(name: str) -> AlgoSpec:
    return _ALGOS[name.lower()]

def list_tasks(): return sorted(_TASKS)
def list_algos(): return sorted(_ALGOS)