from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Any
from collections import defaultdict
import numpy as np
from baselines import BaseConfig, BaseResult, norm, normalize, sample_uniform_on_segment
from oracle import Oracle
Array = np.ndarray

@dataclass
class INGDConfig(BaseConfig):
    T: int = 100
    epsilon_t: float = 1e-5
    seed: Optional[int] = None
    max_inner_iters: int = 1_000
    Lip: float = 1.0
    delta: float = 1e-5

@dataclass
class INGDResult(BaseResult):
    x_T: Array
    history: Dict[str, Any]

def ingd(
    x1: Array,
    oracle: Oracle,
    config: INGDConfig = INGDConfig(),
    callback: Optional[Callable[[int, Array, float], None]] = None,
    **kwargs: Any,
) -> INGDResult:
    eval_freq = kwargs.get('eval_freq', 1)
    store_path = kwargs.get('store_path', False)
    rng = np.random.default_rng(config.seed)
    x_t = x1.copy()
    hist = defaultdict(list)    
    hist["f_t"].append(oracle.f(x1))
    hist["oracle_calls"].append(0)
    hist["path"].append(x1.copy())
    hist["eps"].append(config.epsilon_t)
    

    m_tk = np.zeros_like(x_t)
    for t in range(config.T):
        m_tk = oracle.oracle_grad(x_t, normalize(m_tk))
        while norm(m_tk) > config.epsilon_t:
            find = False
            for k in range(1, config.max_inner_iters + 1):
                x_tk = x_t - config.delta * normalize(m_tk)
                if norm(m_tk) <= config.epsilon_t:
                    f_val = oracle.f(x_t)
                    hist["f_t"].append(f_val)
                    if store_path:
                        hist["path"].append(x_t.copy())
                    hist["g_norm"].append(norm(m_tk))
                    hist["oracle_calls"].append(oracle.first_oracle_call)
                    return INGDResult(x_T=x_t, history=hist)
                elif oracle.f(x_tk) - oracle.f(x_t) <= -config.delta * norm(m_tk) / 4:
                    x_t = x_tk
                    find = True
                    break
                else:
                    y_tk = sample_uniform_on_segment(x_t, x_tk, rng)
                    g_tk = oracle.oracle_grad(y_tk, normalize(m_tk))
                    beta_k = (4 * config.Lip ** 2 - norm(m_tk) ** 2) / (4 * config.Lip ** 2 + 2 * norm(m_tk) ** 2)
                    m_tk =  beta_k * m_tk + (1.0 - beta_k) * g_tk
            if find:
                ## break while loop
                break

        f_val = oracle.f(x_t)
        if callback is not None and t % eval_freq == 0:
            call_back_res, call_back_name = callback(x_t)    
            hist[call_back_name].append(call_back_res)
            hist[f"{call_back_name}_x"].append(oracle.first_oracle_call)
        print(f"INGD step {t}, function value: {f_val}, stepsize: {config.delta}, oracle calls: {oracle.first_oracle_call}")
        hist["f_t"].append(f_val)
        if store_path:
            hist["path"].append(x_t.copy())
        hist["g_norm"].append(norm(m_tk))
        hist["oracle_calls"].append(oracle.first_oracle_call)
        if oracle.first_oracle_call > 20_000:
            return INGDResult(x_T=x_t, history=hist)
    return INGDResult(x_T=x_t, history=hist)


