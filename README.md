# INFW

NumPy implementation of Inner-INFW (Algorithm 1), FO-Inner-INFW
(Algorithm 3), and Anytime-INFW (Algorithm 4) from
*A Parameter-Free Anytime* (the supplied manuscript).

## Quick start

Python 3.10 or newer:

```sh
python -m pip install numpy
PYTHONPATH=. python examples/quadratic.py
python -m unittest discover -s tests -v
```

Provide an oracle with `f(x)`, `oracle_grad(x, direction=None)`, and an
integer `first_oracle_call` counter. For Algorithm 3 the latter method
returns the gradient. Algorithm 1 requires equation (2.2)'s directional
subgradient oracle.

`anytime_infw(x, oracle, AnytimeINFWConfig(T=100, seed=0))` returns the
last point, diagnostic history, status, and the best certificate tuple
`(x_best, g_best, epsilon_best, inner_call_number)`. The certificate is
the anytime output; the last point need not be its location.

The default initial scale is the initial gradient norm, as in Algorithm 4.
`epsilon_1` is an optional experimental override. Iteration limits bound
execution; `inner_limit` and `scale_limit` indicate incomplete searches.
No certificate is manufactured from an unfinished solve.

## Optional training experiments

Install `requirements-experiments.txt` for the retained MNIST/CIFAR-10
training scripts, models, experiment registry, baseline methods and plot
helpers. Run `python mnist_train.py --help` or
`python cifar10_train.py --help` for options. Dataset downloads require
network access. Use `stochastic=False` for INFW: independent minibatch
loss comparisons do not implement the fixed-objective paper algorithm.
Deterministic training uses one cached batch, not the full training set.
The training and baseline code is retained experimental code; full model
training and numerical reproduction of paper figures have not been verified.

Generated results, datasets, checkpoints, old plots and Git history are
excluded. The original folder remains the archive. No license is inferred;
choose a license and confirm ownership before publishing.
