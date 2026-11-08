# INFW

NumPy implementation of Inner-INFW, FO-Inner-INFW, and Anytime-INFW
(Algorithms 1, 3, and 4 of the paper).

## Quick start

Uses Python 3.11 (selected by `.python-version`) and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run from the repository root:

```sh
uv sync --locked
PYTHONPATH=. uv run python examples/quadratic.py
```

## Usage

```python
from anytime_infw import AnytimeINFWConfig, anytime_infw

result = anytime_infw(x0, oracle, AnytimeINFWConfig(T=100, seed=0))
x_best, g_best, epsilon_best, call_number = result.certificate
```

The oracle provides `f(x)`, `oracle_grad(x, direction=None)`, and a
`first_oracle_call` counter. See `examples/quadratic.py` for a complete example.
The result includes the final point (`x_T`), history, status, and best
stationarity certificate. Check `status` for iteration limits.

## MNIST and CIFAR-10

```sh
uv sync --locked --extra experiments
uv run --extra experiments python mnist_train.py --help
uv run --extra experiments python cifar10_train.py --help
```

Run a short fixed-batch MNIST experiment:

```sh
uv run --locked --extra experiments python mnist_train.py --infw --infw-T 5 --hidden 256 --batch-size 256 --seed 0
```

Results are saved under `results/`; datasets are downloaded to `data/`.
INFW uses `stochastic=False` by default and trains on one fixed batch.
