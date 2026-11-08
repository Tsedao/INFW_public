from typing import Dict, Any, Union, Optional


import matplotlib.pyplot as plt
import numpy as np
from anytime_infw import AnytimeINFWResult, INGDResult

def plot_f_oracle_calls(
    *,
    results: Union[AnytimeINFWResult, INGDResult],
    ax: Optional[plt.Axes] = None,
    method_name: str,
) -> None:
    if ax is None:
        ax = plt.gca()

    ### plot function value v.s. first order oracle calls
    ln,  = ax.plot(results.history["oracle_calls"], results.history["f_t"], label=f"{method_name}")
    ax.scatter(results.history["oracle_calls"], results.history["f_t"], color=ln.get_color(), alpha=0.5, s=5)
    ax.set_xlabel("first order oracle calls")
    ax.set_ylabel("function value")
    # ax.legend()
# --- smoothing helpers ---
def ema_smooth(x, alpha=0.05):
    """
    Exponential moving average smoothing.
    alpha in (0,1): smaller => smoother (more lag), larger => less smooth.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def plot_f_oracle_calls_stochastic(
    *,
    results: Union[AnytimeINFWResult, INGDResult],
    ax: Optional[plt.Axes] = None,
    method_name: str,
) -> None:
    if ax is None:
        ax = plt.gca()
    data = results.history["f_t"]
    x = results.history["oracle_calls"]
    smoothed_data = ema_smooth(data)

    ### plot function value v.s. first order oracle calls
    (ln1,) = ax.plot(x, smoothed_data, label=f"{method_name}", linewidth=2)
    ax.plot(x, data, alpha=0.25, color=ln1.get_color())
    ax.set_xlabel("first order oracle calls")
    ax.set_ylabel("function value")

def plot_grad_norm_oracle_calls(
    *,
    results: Union[AnytimeINFWResult, INGDResult],
    ax: Optional[plt.Axes] = None,
    method_name: str,
) -> None:
    if ax is None:
        ax = plt.gca()

    ### plot gradient norm v.s. first order oracle calls
    ln, = ax.plot(results.history["oracle_calls"][1:], results.history["g_norm"], label=f"{method_name}")
    ax.scatter(results.history["oracle_calls"][1:], results.history["g_norm"], color=ln.get_color(), alpha=0.5, s=5)
    ax.set_xlabel("first order oracle calls")
    ax.set_ylabel("gradient norm")
    # ax.legend()


def plot_acc_oracle_calls(
    *,
    results: Union[AnytimeINFWResult, INGDResult],
    ax: Optional[plt.Axes] = None,
    method_name: str,
    color: Optional[str] = None,
) -> None:
    try:
        x = results.history['test_acc_x']
        y = results.history['test_acc']
    except KeyError:
        x = results.history["oracle_calls"][1:]
        y = results.history["test_acc"]
    if len(x) == 0:
        x = results.history["oracle_calls"][1:]
    if color is not None:
        ln, = ax.plot(x, y, label=f"{method_name}", color=color)
        ax.scatter(x, y, color=ln.get_color(), alpha=0.5, s=5)
    else:
        ln, = ax.plot(x, y, label=f"{method_name}")
        ax.scatter(x, y, color=ln.get_color(), alpha=0.5, s=5)
    ax.set_xlabel("first order oracle calls")
    ax.set_ylabel("test accuracy")
    # ax.legend()

def plot_eps_oracle_calls(
    *,
    results: Union[AnytimeINFWResult, INGDResult],
    ax: Optional[plt.Axes] = None,
    method_name: str,
    color: Optional[str] = None,
) -> None:
    if ax is None:
        ax = plt.gca()

    if color is not None:
        ln, = ax.plot(results.history["oracle_calls"], results.history["eps"], label=f"{method_name}", color=color)
        ax.scatter(results.history["oracle_calls"], results.history["eps"], color=ln.get_color(), alpha=0.5, s=1)
    else:
        ln, = ax.plot(results.history["oracle_calls"], results.history["eps"], label=f"{method_name}")
        ax.scatter(results.history["oracle_calls"], results.history["eps"], color=ln.get_color(), alpha=0.5, s=1)
    ax.set_xlabel("first order oracle calls")
    ax.set_ylabel("Step size")
    

        
