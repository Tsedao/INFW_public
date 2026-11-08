import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from typing import Callable, Optional, Tuple, Dict, Any, List

from dataclasses import dataclass,field

Array = np.ndarray
# -------------------------
# Oracle interface
# -------------------------
# We assume access to: f(x) and a gradient oracle grad(x).
# The algorithm also invokes a non-standard oracle that returns a gradient at y
# whose inner product with a given direction d equals the (Clarke) directional derivative.
# For C^1 functions, grad(y) already satisfies this property, so default
# implementation simply calls grad(y).

@dataclass
class Oracle:
    f: Callable[[Array], float]
    grad: Callable[[Array], Array]

    # Optional custom directional-derivative matching oracle.
    # If None, we fall back to grad(y).
    dir_grad: Optional[Callable[[Array, Array], Array]] = None
    first_oracle_call: int = field(default=0, init=False)

    def oracle_grad(self, y: Array, direction: Array = None) -> Array:
        self.first_oracle_call += 1
        if self.dir_grad is not None and direction is not None:
            return self.dir_grad(y, direction)
        # Default for smooth f
        return self.grad(y)
    
    def restart(self):
        self.first_oracle_call = 0
    

class NNOracle(Oracle):
    """
    Oracle wrapper of nn.Module for both deterministic and stochastic mini-batch cross-entropy on MNIST.

    Main Methods:
      - f(x): returns mini-batch loss at params x
      - oracle_grad(x, ghat): returns gradient wrt x computed at params x
        (exact autograd gradient, which satisfies the dir-deriv property)
    """

    def __init__(
            self, 
            model: nn.Module, 
            dataloader: DataLoader,
            device: torch.device,
            stochastic: bool = True,
        ):

        self.model = model.to(device)
        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)
        self.device = device
        self.stochastic = stochastic

        # fixed batch for non-stochastic oracle
        self._cached_batch = self._next_batch()
        self.first_oracle_call = 0
        self.zero_oracle_calls = 0

    def _next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        return x, y
    
    def _get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.stochastic:
            return self._next_batch()
        else:
            return self._cached_batch
    
    
    def _fatten_params(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.cat([p.detach().view(-1) for p in self.model.parameters()])
        
    def _assign_flat_params(self, params: torch.Tensor) -> None:
        ## inplace assignment to each layer's parameters
        offset = 0
        for p in self.model.parameters():
            num_param = p.numel()
            p.data.copy_(params[offset:offset+num_param].view_as(p.data))
            offset += num_param
    
    def _set_params_from_numpy(self, params: np.ndarray) -> None:
        self._assign_flat_params(torch.from_numpy(params).to(self.device))


    def _loss_no_grad(self, x: np.ndarray) -> float:
        self._set_params_from_numpy(x)
        self.model.eval()
        with torch.no_grad():
            Xb, yb = self._get_batch()
            logits = self.model(Xb)
            loss = F.cross_entropy(logits, yb)
        return float(loss.item())
    
    def start_outer_iter(self, reuse_batch: bool = True):
        if reuse_batch:
            self._cached_batch = self._next_batch()
        else:
            pass
    
    def f(self, x: np.ndarray) -> float:
        self.zero_oracle_calls += 1
        if self.stochastic:
            ## using miaolan's approach evaluate the loss 1/epsilon^2 times
            # epsilon = 1e-1
            # num_samples = int(1 / (epsilon ** 2))
            num_samples = 1
            loss = 0
            for _ in range(num_samples):
                loss += self._loss_no_grad(x)
            return loss / num_samples
        else:
            return self._loss_no_grad(x)

    def oracle_grad(self, x: np.ndarray, direction: np.ndarray = None) -> np.ndarray:
        # Return exact gradient at x (ghat arg (direction) is ignored; exact grad satisfies dir-deriv property)
        self.first_oracle_call += 1
        self._set_params_from_numpy(x)
        self.model.train(False)
        Xb, yb = self._get_batch()

        # Forward + backward
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        logits = self.model(Xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()

        # Flatten grads
        grads = []
        for p in self.model.parameters():
            if p.grad is None:
                grads.append(torch.zeros_like(p).view(-1))
            else:
                grads.append(p.grad.detach().view(-1))
        g_flat = torch.cat(grads).detach().to("cpu").numpy().astype(np.float32)
        return g_flat