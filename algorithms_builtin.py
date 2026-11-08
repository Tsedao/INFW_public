# algorithm registry
# convert name to function
# ----------------------------------------------------------------------
from registry import AlgoSpec, register_algo

from anytime_infw import AnytimeINFWConfig, AnytimeINFWResult, anytime_infw
from ingd import INGDConfig, INGDResult, ingd

from baselines import SGDConfig, SGDResult, sgd
from baselines import StochasticINGDConfig, StochasticINGDResult, stochastic_ingd
from baselines import LipschitzMinimizationConfig, LipschitzMinimizationResult, lipschitz_minimization


register_algo(AlgoSpec(
    name="INFW",
    run_fn=anytime_infw,
    config_factory=lambda **k: AnytimeINFWConfig(**k),
    default_cfg=dict(T=50, epsilon_1=None, seed=0, max_inner_iters=10_000),
))

## Deterministic INGD + Miaolan's Stochastic Linear Search
register_algo(AlgoSpec(
    name="INGD",
    run_fn=ingd,
    config_factory=lambda **k: INGDConfig(**k),
    default_cfg=dict(T=100, epsilon_t=1e-1, delta=1e-3, seed=0, Lip=1.0),
))

register_algo(AlgoSpec(
    name="SGD",
    run_fn=sgd,
    config_factory=lambda **k: SGDConfig(**k),
    default_cfg=dict(T=100, lr=1e-2),
))

## Jinzhao's Stochastic INGD
register_algo(AlgoSpec(
    name="StochasticINGD",
    run_fn=stochastic_ingd,
    config_factory=lambda **k: StochasticINGDConfig(**k),
    default_cfg=dict(T=100, p=1e-2, q=1e-2, beta=1e-2, seed=0),
))

## Lipschitz Minimization (KS25's Algorithm 8.4)
register_algo(AlgoSpec(
    name="LipschitzMinimization",
    run_fn=lipschitz_minimization,
    config_factory=lambda **k: LipschitzMinimizationConfig(**k),
    default_cfg=dict(T=100, tolerance=1e-6, L=1.0, beta=1.0, seed=0),
))