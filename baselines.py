import numpy as np
from typing import Callable, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict

from oracle import Oracle, NNOracle

Array = np.ndarray
# -------------------------
# Utility helpers
# -------------------------

def norm(x: Array) -> float:
    return float(np.linalg.norm(x))


def normalize(v: Array, eps: float = 1e-12) -> Array:
    n = norm(v)
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def sample_uniform_on_segment(x: Array, y: Array, rng: np.random.Generator) -> Array:
    """Return a point ~ Uniform on the line segment [x, y]."""
    u = rng.uniform(0.0, 1.0)
    return x + u * (y - x)

@dataclass
class BaseConfig:
    T: int = 100
    seed: int = 0
    eval_freq : int = 1

@dataclass
class BaseResult:
    x_T: np.ndarray
    history: Dict[str, Any]



@dataclass
class SGDConfig(BaseConfig):
    T: int = 100
    lr: float = 0.01

@dataclass
class SGDResult(BaseResult):
    pass

def sgd(
    x1: np.ndarray,
    oracle: Oracle,
    config: SGDConfig = SGDConfig(),
    callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    **kwargs: Any,
) -> SGDResult:
    eval_freq = kwargs.get('eval_freq', 1)
    x_t = x1.copy()
    history = defaultdict(list)
    # history["x_t"].append(x_t.copy())
    history["f_t"].append(oracle.f(x_t))
    history["oracle_calls"].append(0)
    history["eps"].append(config.lr)

    for t in range(config.T):
        grad = oracle.oracle_grad(x_t)
        x_t -= config.lr * grad
        # history["x_t"].append(x_t.copy())
        history["f_t"].append(oracle.f(x_t))
        history["oracle_calls"].append(oracle.first_oracle_call)
        history["g_norm"].append(np.linalg.norm(grad))
        history["eps"].append(config.lr)
        print(f"SGD: t={t}, f_t={history['f_t'][-1]}, oracle_calls={history['oracle_calls'][-1]}")
        if callback is not None and t % eval_freq == 1:
            call_back_res, call_back_name = callback(x_t)
            history[call_back_name].append(call_back_res)
            history[f"{call_back_name}_x"].append(oracle.first_oracle_call)

    return SGDResult(x_T=x_t, history=history)

@dataclass
class StochasticINGDConfig(BaseConfig):
    T: int = 100
    p: float = 0.01
    q: float = 0.01
    beta: float = 0.01
    seed: Optional[int] = None
    max_inner_iters: int = 10_000
    max_epsilon: float = 20

@dataclass
class StochasticINGDResult(BaseResult):
    pass


def stochastic_ingd(
    x1: np.ndarray,
    oracle: Union[NNOracle, Oracle],
    config: StochasticINGDConfig = StochasticINGDConfig(),
    callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    **kwargs: Any,
) -> SGDResult:
    eval_freq = kwargs.get('eval_freq', 1)
    rng = np.random.RandomState(config.seed)
    x_t = x1.copy()
    history = defaultdict(list)
    # history["x_t"].append(x_t.copy())
    history["f_t"].append(oracle.f(x_t))
    history["oracle_calls"].append(0)
    history["eps"].append(config.p)
    # oracle_calls = 0

    grad = oracle.oracle_grad(x_t)
    m_t = grad
    for t in range(config.T):
        eta_t = 1/ (config.p * np.linalg.norm(m_t) + config.q)
        x_t -= eta_t * m_t
        # history["x_t"].append(x_t.copy())
        history["f_t"].append(oracle.f(x_t))
        history["oracle_calls"].append(oracle.first_oracle_call)
        history["eps"].append(eta_t)
        history["g_norm"].append(np.linalg.norm(m_t))
        
        y_t = sample_uniform_on_segment(x_t, x_t - eta_t * m_t, rng)
        grad_y = oracle.oracle_grad(y_t)
        # oracle_calls += 1
        m_t = config.beta * m_t + (1 - config.beta) * grad_y
        print(f"Stochastic INGD: t={t}, f_t={history['f_t'][-1]}, oracle_calls={history['oracle_calls'][-1]}")
        if callback is not None and t % eval_freq == 0:
            call_back_res, call_back_name = callback(x_t)
            history[call_back_name].append(call_back_res)
            history[f"{call_back_name}_x"].append(oracle.first_oracle_call)

    return SGDResult(x_T=x_t, history=history)


#### DD22's Algorithm 2, KS25's Algorithm 8.2
def shortest_vector(g : np.ndarray, h : np.ndarray):
    """
    Find the shortest vector in the span of g and h.
    this eqivalents to solve the following problem:
        t^* = argmin_{t \in [0, 1]} ||(1 - t)g + th||
            =||g||^2 - <g, h> / ||g - h||^2
    """
    numerator = np.dot(g, g) - np.dot(g, h)
    denominator = np.dot(g - h, g - h)
    t_star = numerator / denominator if denominator > 0 else 0
    t_opt = np.clip(t_star, 0, 1)
    v_shortest = (1 - t_opt) * g + t_opt * h
    return v_shortest, t_opt


def approximate_goldstein_subgradient(
    x1: np.ndarray,
    oracle: Union[Oracle, NNOracle],
    *,
    initial_radius: float = 1.0,
    max_iters: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Approximate the subgradient of the oracle at x1.
    """
    rng = np.random.RandomState(seed)
    x_t = x1.copy()
    random_direction = np.random.randn(*x_t.shape)
    random_direction /= np.linalg.norm(random_direction) 
    grad = oracle.oracle_grad(x_t, direction=random_direction)
    gamma = np.linalg.norm(grad)
    cnt = 0
    while gamma > initial_radius and cnt < max_iters:
        y = x_t - initial_radius * grad / gamma
        if oracle.f(y) <= oracle.f(x_t) - initial_radius * gamma / 2:
            break
        ## sample z \in [x_t, y] uniformly
        z = sample_uniform_on_segment(x_t, y, rng)
        grad_z = oracle.oracle_grad(z, direction= grad / gamma)
        gamma = np.linalg.norm(grad_z)

        ## shortest vector in the span of [grad, grad_z]
        v_shortest, _ = shortest_vector(grad, grad_z)
        grad = v_shortest
        gamma = np.linalg.norm(grad)
        cnt += 1
    return grad


#### KS25's Algorithm 8.3
def goldstein_modulus_estimate(
    x1: np.ndarray,
    oracle: Union[Oracle,NNOracle],
    *,
    initial_radius: float = 1.0,
    tolerance: float = 1e-6,
    max_iters: int = 1000,
    seed: Optional[int] = None,
    history: Optional[Dict[str, Any]] = None,
) -> Tuple[float, np.ndarray, int]:
    """
    Estimate the modulus of the gradient of the oracle at x1.
    """
    x_t = x1.copy()
    random_direction = np.random.randn(*x_t.shape)
    random_direction /= np.linalg.norm(random_direction) 
    grad = oracle.oracle_grad(x_t, direction=random_direction)
    radius = initial_radius
    cnt = 0
    flag = False
    while np.linalg.norm(grad) < radius and cnt < max_iters:
        radius = radius / 2
        ## find approximate subgradient
        grad = approximate_goldstein_subgradient(x_t, oracle, initial_radius=radius, seed=seed)
        history["g_norm_previous"].append(np.linalg.norm(grad))
        history["oracle_calls_g_norm"].append(oracle.first_oracle_call)

        if np.linalg.norm(grad) < tolerance:
            ### x is approximate stationary point
            flag = True
            break
        print(oracle.first_oracle_call)
        if oracle.first_oracle_call > 50_000:
            flag = True
            break
        cnt += 1
    return radius, grad, flag

## KS25's Algorithm 8.4
@dataclass
class LipschitzMinimizationConfig(BaseConfig):
    T: int = 100
    # initial_radius: float = 1.0
    tolerance: float = 1e-6
    L: float = 1.0
    beta: float = 1.0
    seed: Optional[int] = None

@dataclass
class LipschitzMinimizationResult(BaseResult):
    pass

def lipschitz_minimization(
    x1: np.ndarray,
    oracle: Union[Oracle, NNOracle],
    config: LipschitzMinimizationConfig = LipschitzMinimizationConfig(),
    callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    **kwargs: Any,
) -> LipschitzMinimizationResult:
    eval_freq = kwargs.get('eval_freq', 1)
    store_path = kwargs.get("store_path", False)
    rng = np.random.RandomState(config.seed)
    x_t = x1.copy()
    history = defaultdict(list)
    history["f_t"].append(oracle.f(x_t))
    history["oracle_calls"].append(0)
    history["oracle_calls_g_norm"].append(0)
    history["eps"].append(config.L)

    for t in range(config.T):
        radius = config.L
        ## run goldstein_modulus_estimate
        radius, grad, flag = goldstein_modulus_estimate(
            x_t, oracle, initial_radius=radius, seed=config.seed,
            tolerance=config.tolerance, max_iters=config.T, history=history
        )
        if np.linalg.norm(grad) < config.tolerance or flag:
            return LipschitzMinimizationResult(x_T=x_t, history=history)
        radius = 2 * config.beta * radius
        ## run goldstein_modulus_estimate again
        radius, grad, flag = goldstein_modulus_estimate(
            x_t, oracle, initial_radius=radius, seed=config.seed,
            tolerance=config.tolerance, max_iters=config.T, 
            history=history
        )
        if np.linalg.norm(grad) < config.tolerance or flag:
            print(f"Reaching {config.tolerance} stationary point...")
            return LipschitzMinimizationResult(x_T=x_t, history=history)
        x_t -= radius * grad / np.linalg.norm(grad)

        history["f_t"].append(oracle.f(x_t))
        history["oracle_calls"].append(oracle.first_oracle_call)
        history["eps"].append(radius)
        history["g_norm"].append(np.linalg.norm(grad))
        if store_path:
            history['path'].append(x_t.copy())
        print(f"Lipschitz Minimization: t={t}, f_t={history['f_t'][-1]}, g_norm={np.linalg.norm(grad)}, radius={radius}, oracle_calls={history['oracle_calls'][-1]}")
        if callback is not None and t % eval_freq == 0:
            call_back_res, call_back_name = callback(x_t)
            history[call_back_name].append(call_back_res)
            history[f"{call_back_name}_x"].append(oracle.first_oracle_call)

    return LipschitzMinimizationResult(x_T=x_t, history=history)
