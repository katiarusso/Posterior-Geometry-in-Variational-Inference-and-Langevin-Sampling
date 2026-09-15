"""
Experiment 1
Bivariate Gaussian target.

Purpose
-------
Reproduce the figures and diagnostic tables for Section 3.2 of the project
in a self-contained script. The original analysis was developed in
Experiment1.ipynb; this script extracts only what is needed for the figures
that appear in the project report, with a consistent visual style shared with the
Experiment 2 (second version) script (2x2 contour grid, no matplotlib
figure-level titles, identical typographic conventions).

Model
-----
Target distributions
    pi_rho = N(0, Sigma_rho),    Sigma_rho = [[1, rho], [rho, 1]],
for rho in {0, 0.5, 0.9, 0.99}. The condition number of Sigma_rho is
    kappa = (1 + rho) / (1 - rho),
and the optimal Gaussian mean-field optimum for KL(q || pi_rho) is
    q_rho^* = N(0, (1 - rho^2) I_2).

ULA recursion
    X_{k+1} = (I - eta Sigma^{-1}) X_k + sqrt(2 eta) xi_k,    xi_k ~ N(0, I_2),
with step size
    eta = 0.8 / L,    L = lambda_max(Sigma^{-1}).
Along the slow eigendirection the chain reduces to an AR(1) with coefficient
1 - eta m = 1 - 0.8 / kappa.

Outputs
-------
1. experiment1_target_vs_meanfield.pdf : 2x2 contour grid, posterior (solid)
   versus optimal Gaussian mean-field (dashed).
2. experiment1_ula_acf_slow_direction.pdf : empirical ACF along the slow
   eigendirection with the AR(1) theoretical curves.
3. experiment1_theoretical_quantities.csv : analytic per-rho table.
4. experiment1_ula_diagnostics.csv : ULA per-rho table.
5. experiment1_stationary_covariance_comparison.csv : empirical vs theoretical
   ULA stationary variance and correlation.

Dependencies: numpy, pandas, matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ExperimentConfig:
    rhos: tuple[float, ...] = (0.0, 0.5, 0.9, 0.99)
    n_steps: int = 50_000
    burn_in: int = 10_000
    step_constant: float = 0.8
    seed: int = 123
    max_acf_lag: int = 200
    output_dir: str = "results/experiment_1"


def covariance_matrix(rho: float) -> np.ndarray:
    """Sigma_rho = [[1, rho], [rho, 1]]."""
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must lie in (-1, 1).")
    return np.array([[1.0, rho], [rho, 1.0]])


def precision_matrix(rho: float) -> np.ndarray:
    """Lambda_rho = Sigma_rho^{-1}."""
    return np.linalg.inv(covariance_matrix(rho))


def mean_field_covariance(rho: float) -> np.ndarray:
    """Optimal Gaussian mean-field covariance for KL(q || pi_rho)."""
    return (1.0 - rho ** 2) * np.eye(2)


def optimal_mean_field_kl(rho: float) -> float:
    """Closed-form KL(q* || pi_rho) for the bivariate target."""
    return -0.5 * np.log(1.0 - rho ** 2)


def ula_step_size(rho: float, step_constant: float) -> float:
    """eta = step_constant / L with L = lambda_max(Sigma^{-1})."""
    Lambda = precision_matrix(rho)
    L = float(np.linalg.eigvalsh(Lambda)[-1])
    return step_constant / L


def simulate_ula_gaussian(
    rho: float,
    n_steps: int,
    burn_in: int,
    step_constant: float,
    rng: np.random.Generator,
) -> dict:
    """Simulate ULA for N(0, Sigma_rho) and return diagnostics."""
    Sigma = covariance_matrix(rho)
    Lambda = np.linalg.inv(Sigma)
    eigvals, eigvecs = np.linalg.eigh(Lambda)
    L = float(eigvals[-1])
    m = float(eigvals[0])
    eta = step_constant / L
    sqrt_2eta = np.sqrt(2.0 * eta)

    x = rng.normal(size=2) * 0.0  # start at the mean for stationarity diagnostics
    chain = np.empty((n_steps, 2))
    for k in range(n_steps):
        x = x - eta * (Lambda @ x) + sqrt_2eta * rng.normal(size=2)
        chain[k] = x

    post_burn = chain[burn_in:]
    return {
        "rho": rho,
        "Sigma": Sigma,
        "Lambda": Lambda,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "L": L,
        "m": m,
        "eta": eta,
        "chain": chain,
        "post_burn": post_burn,
    }


def autocorrelation_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased empirical ACF for lags 0 through ``max_lag``."""
    z = np.asarray(x, dtype=float) - np.mean(x)
    denom = float(np.dot(z, z))
    if denom <= 0:
        return np.full(max_lag + 1, np.nan)
    acf = np.empty(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = float(np.dot(z[:-lag], z[lag:])) / denom
    return acf


def ess_until_first_nonpositive(acf: np.ndarray, sample_size: int) -> float:
    """ESS estimate truncating at the first non-positive autocorrelation.

    This is the estimator used in the original project output. It is intentionally named
    precisely: unlike Geyer's IPS estimator, it does not group autocovariances
    into adjacent pairs.
    """
    positive_sum = 0.0
    for value in acf[1:]:
        if value <= 0:
            break
        positive_sum += float(value)
    return sample_size / (1.0 + 2.0 * positive_sum)


def ellipse_points(
    mu: np.ndarray, cov: np.ndarray, levels: tuple[float, ...], n_grid: int = 300
) -> list[np.ndarray]:
    """Quadratic-level contour points for N(mu, cov)."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    theta = np.linspace(0.0, 2.0 * np.pi, n_grid)
    unit_circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipses = []
    for level in levels:
        transform = eigvecs @ np.diag(np.sqrt(level * eigvals))
        pts = mu[:, None] + transform @ unit_circle
        ellipses.append(pts.T)
    return ellipses


def plot_target_vs_meanfield(cfg: ExperimentConfig, out_dir: Path) -> None:
    """Two-by-two grid of target / mean-field contour comparisons."""
    n_panels = len(cfg.rhos)
    assert n_panels == 4, "this plot is designed for the four-rho grid"
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 9.0))
    axes = axes.flatten()
    levels = (1.0, 2.5, 5.0)

    for ax, rho in zip(axes, cfg.rhos):
        Sigma = covariance_matrix(rho)
        D_star = mean_field_covariance(rho)
        kappa = (1.0 + rho) / (1.0 - rho)
        for pts in ellipse_points(np.zeros(2), Sigma, levels):
            ax.plot(pts[:, 0], pts[:, 1], linewidth=1.5)
        for pts in ellipse_points(np.zeros(2), D_star, levels):
            ax.plot(pts[:, 0], pts[:, 1], linestyle="--", linewidth=1.5)
        ax.scatter([0.0], [0.0], s=12, color="tab:blue")
        ax.set_title(fr"$\rho={rho:g}$, $\kappa={kappa:.1f}$")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)

    # Figure-level title intentionally omitted: the LaTeX caption describes the figure.
    fig.tight_layout()
    fig.savefig(out_dir / "experiment1_target_vs_meanfield.pdf")
    fig.savefig(out_dir / "experiment1_target_vs_meanfield.png", dpi=200)
    plt.close(fig)


def plot_ula_slow_acf(
    cfg: ExperimentConfig,
    out_dir: Path,
    simulation_results: list[dict],
) -> None:
    """Single-panel ACF plot: empirical + theoretical AR(1)."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    lags = np.arange(cfg.max_acf_lag + 1)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, res in enumerate(simulation_results):
        rho = res["rho"]
        eigvecs = res["eigvecs"]
        rotated = res["post_burn"] @ eigvecs
        # eigh returns eigenvalues in ascending order, so column 0 = slow direction.
        slow_chain = rotated[:, 0]
        acf_slow = autocorrelation_1d(slow_chain, cfg.max_acf_lag)
        slow_ar = 1.0 - res["eta"] * res["m"]
        kappa = (1.0 + rho) / (1.0 - rho)
        color = color_cycle[i % len(color_cycle)]
        ax.plot(lags, acf_slow, color=color, linewidth=1.4,
                label=fr"empirical $\rho={rho:g}$, $\kappa={kappa:.1f}$")
        ax.plot(lags, slow_ar ** lags, color=color, linestyle="--", linewidth=1.0)

    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    # Figure-level title intentionally omitted.
    ax.set_ylim(-0.1, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "experiment1_ula_acf_slow_direction.pdf")
    fig.savefig(out_dir / "experiment1_ula_acf_slow_direction.png", dpi=200)
    plt.close(fig)


def make_tables(
    cfg: ExperimentConfig,
    out_dir: Path,
    simulation_results: list[dict],
) -> None:
    theory_rows = []
    ula_rows = []
    stat_rows = []
    for res in simulation_results:
        rho = res["rho"]
        Sigma = res["Sigma"]
        Lambda = res["Lambda"]
        eigvals = res["eigvals"]
        eta = res["eta"]
        L = res["L"]
        m = res["m"]
        kappa = (1.0 + rho) / (1.0 - rho)
        D_star = mean_field_covariance(rho)

        # Empirical ULA diagnostics
        rotated = res["post_burn"] @ res["eigvecs"]
        slow_chain = rotated[:, 0]
        fast_chain = rotated[:, -1]
        acf_slow = autocorrelation_1d(slow_chain, cfg.max_acf_lag)
        acf_fast = autocorrelation_1d(fast_chain, cfg.max_acf_lag)
        sample_size = len(slow_chain)
        ess_slow = ess_until_first_nonpositive(acf_slow, sample_size)
        ess_fast = ess_until_first_nonpositive(acf_fast, sample_size)

        emp_cov = np.cov(res["post_burn"].T)
        theo_var_1 = float(eta / (1.0 - (1.0 - eta * eigvals[0]) ** 2)) \
            * 0  # placeholder, computed below to keep formula visible
        # ULA stationary variance along direction i is 1 / (lambda_i - eta lambda_i^2 / 2)
        ula_stationary_vars = 1.0 / (eigvals - 0.5 * eta * eigvals ** 2)
        # Express back in the original basis: rotated diag(ula_stationary_vars) by eigvecs
        ula_stationary_cov = res["eigvecs"] @ np.diag(ula_stationary_vars) @ res["eigvecs"].T

        emp_corr = emp_cov[0, 1] / np.sqrt(emp_cov[0, 0] * emp_cov[1, 1])
        theo_corr = ula_stationary_cov[0, 1] / np.sqrt(ula_stationary_cov[0, 0] * ula_stationary_cov[1, 1])

        theory_rows.append({
            "rho": rho,
            "kappa": kappa,
            "m": m,
            "L": L,
            "eta": eta,
            "MF variance": float(D_star[0, 0]),
            "KL(q* || pi)": float(optimal_mean_field_kl(rho)),
        })
        ula_rows.append({
            "rho": rho,
            "kappa": kappa,
            "eta": eta,
            "target corr": rho,
            "empirical corr ULA": float(emp_corr),
            "MF variance": float(D_star[0, 0]),
            "KL(q* || pi)": float(optimal_mean_field_kl(rho)),
            "ESS slow direction": ess_slow,
            "ESS fast direction": ess_fast,
            "ACF slow lag 50": float(acf_slow[50]),
            "ACF slow lag 100": float(acf_slow[100]),
        })
        stat_rows.append({
            "rho": rho,
            "kappa": kappa,
            "eta": eta,
            "target var 1": 1.0,
            "empirical ULA var 1": float(emp_cov[0, 0]),
            "theoretical ULA var 1": float(ula_stationary_cov[0, 0]),
            "target corr": rho,
            "empirical ULA corr": float(emp_corr),
            "theoretical ULA corr": float(theo_corr),
        })

    pd.DataFrame(theory_rows).to_csv(out_dir / "experiment1_theoretical_quantities.csv", index=False)
    pd.DataFrame(ula_rows).to_csv(out_dir / "experiment1_ula_diagnostics.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(out_dir / "experiment1_stationary_covariance_comparison.csv", index=False)


def main() -> None:
    cfg = ExperimentConfig()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot the analytical contour comparison (does not depend on the simulation).
    plot_target_vs_meanfield(cfg, out_dir)

    # Independent deterministic sub-seeds make every rho reproducible in isolation.
    simulation_results = []
    for rho in cfg.rhos:
        rng = np.random.default_rng(cfg.seed + int(round(10_000 * rho)))
        simulation_results.append(
            simulate_ula_gaussian(
                rho=rho,
                n_steps=cfg.n_steps,
                burn_in=cfg.burn_in,
                step_constant=cfg.step_constant,
                rng=rng,
            )
        )

    plot_ula_slow_acf(cfg, out_dir, simulation_results)
    make_tables(cfg, out_dir, simulation_results)

    print(f"Saved outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
