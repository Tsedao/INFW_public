from __future__ import annotations
"""Algorithms 1, 3 and 4 of the supplied INFW paper (bounded execution)."""
from dataclasses import dataclass
import numpy as np


@dataclass
class AnytimeINFWConfig:
    T: int = 100
    epsilon_1: float | None = None  # optional experimental override
    seed: int | None = None
    max_inner_iters: int = 10000
    max_scale_iters: int = 10000
    eval_freq: int = 1

@dataclass
class AnytimeINFWResult:
    x_T: np.ndarray
    history: dict
    certificate: tuple
    status: str

class InnerLimit(RuntimeError):
    """An inner solve ended without a valid output."""

def unit(g):
    n = np.linalg.norm(g)
    return g / n if n else np.zeros_like(g)

def ball(center, radius, rng):
    direction = rng.normal(size=center.shape)
    while np.linalg.norm(direction) == 0:
        direction = rng.normal(size=center.shape)
    return center + radius * rng.random() ** (1 / center.size) * unit(direction)

def inner_infw(x, epsilon, delta, oracle, *, rng=None, max_iters=10000, first_order=True):
    """Return (point, vector); raise InnerLimit rather than certify on timeout.

    first_order=True implements Algorithm 3; False implements Algorithm 1
    and requires the directional oracle from equation (2.2).
    """
    if epsilon <= 0 or delta <= 0:
        raise ValueError('epsilon and delta must be positive')
    rng = np.random.default_rng() if rng is None else rng
    v = np.zeros_like(x); v.flat[0] = 1
    g = oracle.oracle_grad(ball(x, delta, rng) if first_order else x, v)
    fx = oracle.f(x)
    for k in range(1, max_iters + 1):
        n = np.linalg.norm(g)
        if n <= epsilon:
            return x.copy(), g
        y = x - delta * unit(g)
        if fx - oracle.f(y) >= delta * n / 2:
            return y, g
        beta = 4 / (k + 3)
        # r = ||g|| eta / 8 satisfies the strict bound r < ||g|| eta / 4.
        z = x - delta * unit(ball(g, n * beta / 8, rng)) if first_order else y
        sample = x + rng.random() * (z - x)
        u = oracle.oracle_grad(sample, unit(g))
        g = (1 - beta) * g + beta * u
    if np.linalg.norm(g) <= epsilon:
        return x.copy(), g
    raise InnerLimit('inner iteration limit reached')

def anytime_infw(x1, oracle, config=None, callback=None, **kwargs):
    """Algorithm 4; result.certificate is (x_best, g_best, eps_best, j_best)."""
    c = config or AnytimeINFWConfig()
    if c.T < 0 or c.max_inner_iters < 1 or c.max_scale_iters < 1:
        raise ValueError('invalid iteration limits')
    if getattr(oracle, 'stochastic', False):
        raise ValueError('Algorithm 4 requires a fixed objective; use stochastic=False')
    x = np.asarray(x1, dtype=float).copy()
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError('x1 must be nonempty and finite')
    rng = np.random.default_rng(c.seed)
    g = oracle.oracle_grad(x, np.eye(1, x.size).reshape(x.shape))
    eps = float(np.linalg.norm(g))
    best = (x.copy(), g.copy(), eps, 0)
    hist = dict(f_t=[oracle.f(x)], eps=[eps], oracle_calls=[oracle.first_oracle_call],
                g_norm=[], best_eps=[eps], path=[x.copy()])
    if eps == 0:
        return AnytimeINFWResult(x, hist, best, 'stationary')
    if c.epsilon_1 is not None:
        eps = float(c.epsilon_1)
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError('epsilon_1 must be finite and positive')
    j = 0
    def probe(radius):
        nonlocal j, best
        if radius <= 0 or not np.isfinite(radius):
            raise InnerLimit('scale is outside floating point range')
        out = inner_infw(x, radius, radius, oracle, rng=rng, max_iters=c.max_inner_iters)
        j += 1
        if np.linalg.norm(out[1]) <= radius and radius < best[2]:
            best = (x.copy(), out[1].copy(), radius, j)
        return out
    status = 'outer_limit'
    try:
        for t in range(c.T):
            low = probe(eps)
            for _ in range(c.max_scale_iters):
                high = probe(2 * eps)
                lc = np.linalg.norm(low[1]) <= eps
                hc = np.linalg.norm(high[1]) <= 2 * eps
                if hc and not lc:
                    break
                if not hc:
                    eps *= 2
                    low = high
                else:
                    eps /= 2
                    low = probe(eps)
            else:
                status = 'scale_limit'
                break
            x = min((low[0], high[0]), key=oracle.f).copy()
            hist['f_t'].append(oracle.f(x))
            hist['eps'].append(eps)
            hist['oracle_calls'].append(oracle.first_oracle_call)
            hist['g_norm'].append(float(np.linalg.norm(low[1])))
            hist['best_eps'].append(best[2])
            if kwargs.get('store_path', False): hist['path'].append(x.copy())
            if callback and t % kwargs.get('eval_freq', c.eval_freq) == 0:
                value, name = callback(x)
                hist.setdefault(name, []).append(value)
                hist.setdefault(name + '_x', []).append(oracle.first_oracle_call)
    except InnerLimit:
        status = 'inner_limit'
    return AnytimeINFWResult(x, hist, best, status)
