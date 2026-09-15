"""
Experiment 2
Bayesian linear regression with collinearity and prior regularization.

Purpose
-------
This script supports Section 3.3 of the thesis.
It is not meant to reproduce the bivariate Gaussian toy experiment inside a
regression model. Instead, it shows how posterior geometry arises naturally
from a Bayesian linear regression model with collinear predictors, and how the
prior regularizes the weakly identified direction.

Model
-----
y | beta ~ N(X beta, sigma^2 I_n)
beta     ~ N(0, tau^2 I_p), with p = 2

The design matrix is constructed so that
X'X = n [[1, rho_X], [rho_X, 1]]
exactly. This isolates the role of design collinearity. The posterior is
Gaussian with precision
Lambda_post = X'X / sigma^2 + I / tau^2.

Main quantities reported
------------------------
1. Posterior condition number kappa_post.
2. Posterior correlation corr(beta_1, beta_2).
3. Exact posterior marginal variances.
4. Optimal Gaussian mean-field marginal variances for KL(q || pi).
5. Mean-field variance ratio Var_q(beta_j) / Var_pi(beta_j).
6. ULA slow-direction AR coefficient and optional ESS diagnostics.

Interpretation
--------------
For highly collinear designs, the exact posterior recognizes weak
identification of individual coefficients through large marginal variance and
strong negative posterior correlation. The Gaussian mean-field optimum removes
this dependence and can severely underestimate marginal uncertainty.

Author: generated for thesis experiment workflow
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ExperimentConfig:
    n: int = 100
    sigma: float = 1.0
    beta_true_1: float = 1.0
    beta_true_2: float = -1.0
    seed: int = 12345
    output_dir: str = "results/experiment_2"

    # The grid is deliberately separated into two conceptual axes:
    # rho_grid studies the effect of design collinearity;
    # tau_grid studies prior regularization of that collinearity.
    rho_grid: tuple[float, ...] = (0.0, 0.5, 0.9, 0.99)
    tau_grid: tuple[float, ...] = (10.0, 1.0, 0.3)

    # ULA diagnostics are secondary here. They are included only to connect the
    # weakly identified direction to the slow eigendirection from Chapter 3.
    ula_iterations: int = 60000
    ula_burnin: int = 10000
    max_acf_lag: int = 200


def build_exact_design(n: int, rho: float) -> np.ndarray:
    """Construct an n x 2 design matrix with X.T @ X exactly equal to
    n [[1, rho], [rho, 1]].

    The columns are deterministic and centered. Exact construction avoids
    Monte Carlo noise in the geometry, so changes in posterior quantities are
    attributable to rho and tau rather than random design fluctuations.
    """
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must lie in (-1, 1).")
    if n < 2:
        raise ValueError("n must be at least 2.")

    z1 = np.linspace(-1.0, 1.0, n)
    z1 = z1 - z1.mean()
    z1 = z1 / np.linalg.norm(z1) * np.sqrt(n)

    # Use a deterministic vector and orthogonalize it against z1.
    z2_raw = np.cos(np.linspace(0.0, 2.0 * np.pi, n, endpoint=False))
    z2_raw = z2_raw - z2_raw.mean()
    z2_raw = z2_raw - (z2_raw @ z1) / (z1 @ z1) * z1
    z2 = z2_raw / np.linalg.norm(z2_raw) * np.sqrt(n)

    x1 = z1
    x2 = rho * z1 + np.sqrt(1.0 - rho**2) * z2
    X = np.column_stack([x1, x2])

    target = n * np.array([[1.0, rho], [rho, 1.0]])
    if not np.allclose(X.T @ X, target, atol=1e-10, rtol=1e-10):
        raise RuntimeError("Design construction failed to match the target Gram matrix.")
    return X


def posterior_from_regression(
    X: np.ndarray,
    y: np.ndarray,
    sigma: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return posterior mean, covariance and precision for Gaussian regression."""
    p = X.shape[1]
    precision = (X.T @ X) / sigma**2 + np.eye(p) / tau**2
    covariance = np.linalg.inv(precision)
    mean = covariance @ (X.T @ y / sigma**2)
    return mean, covariance, precision


def gaussian_mf_forward_kl_optimum(
    mean: np.ndarray,
    precision: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimal Gaussian mean-field approximation for KL(q || pi).

    If pi = N(mean, Sigma) and Lambda = Sigma^{-1}, then
    q* = N(mean, D*) with D*_{jj} = 1 / Lambda_{jj}.
    """
    d_mf = 1.0 / np.diag(precision)
    D_mf = np.diag(d_mf)
    return mean.copy(), D_mf


def kl_gaussian(mean_q: np.ndarray, cov_q: np.ndarray, mean_p: np.ndarray, cov_p: np.ndarray) -> float:
    """KL(N_q || N_p)."""
    d = mean_q.shape[0]
    precision_p = np.linalg.inv(cov_p)
    diff = mean_p - mean_q
    sign_p, logdet_p = np.linalg.slogdet(cov_p)
    sign_q, logdet_q = np.linalg.slogdet(cov_q)
    if sign_p <= 0 or sign_q <= 0:
        raise ValueError("Covariance matrices must be positive definite.")
    value = 0.5 * (
        np.trace(precision_p @ cov_q)
        + diff.T @ precision_p @ diff
        - d
        + logdet_p
        - logdet_q
    )
    return float(value)


def autocorrelation_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Empirical autocorrelation for lags 0,...,max_lag."""
    z = np.asarray(x, dtype=float)
    z = z - z.mean()
    denom = np.dot(z, z)
    if denom <= 0:
        return np.full(max_lag + 1, np.nan)
    acf = np.empty(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.dot(z[:-lag], z[lag:]) / denom
    return acf


def ess_from_acf(acf: np.ndarray, n_samples: int) -> float:
    """ESS estimate truncating at the first non-positive autocorrelation.

    This preserves the estimator used for the thesis tables; it is not the
    paired-autocovariance IPS estimator introduced by Geyer.
    """
    positive_sum = 0.0
    for lag in range(1, len(acf)):
        if acf[lag] <= 0:
            break
        positive_sum += acf[lag]
    return float(n_samples / (1.0 + 2.0 * positive_sum))


def simulate_ula_gaussian(
    mean: np.ndarray,
    precision: np.ndarray,
    eta: float,
    iterations: int,
    burnin: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate ULA for a Gaussian target with quadratic potential."""
    d = mean.shape[0]
    x = mean.copy()
    samples = np.empty((iterations - burnin, d))
    sqrt_2eta = np.sqrt(2.0 * eta)
    out_index = 0
    for k in range(iterations):
        x = x - eta * (precision @ (x - mean)) + sqrt_2eta * rng.normal(size=d)
        if k >= burnin:
            samples[out_index] = x
            out_index += 1
    return samples


def summarize_one_case(
    rho: float,
    tau: float,
    cfg: ExperimentConfig,
    y_noise: np.ndarray,
    run_ula: bool,
) -> dict[str, float]:
    """Compute all analytical quantities and optional ULA diagnostics for one case."""
    X = build_exact_design(cfg.n, rho)
    beta_true = np.array([cfg.beta_true_1, cfg.beta_true_2])
    y = X @ beta_true + cfg.sigma * y_noise

    mean, cov, precision = posterior_from_regression(X, y, cfg.sigma, tau)
    _, D_mf = gaussian_mf_forward_kl_optimum(mean, precision)

    eigvals, eigvecs = np.linalg.eigh(precision)
    m = float(eigvals[0])
    L = float(eigvals[-1])
    kappa = float(L / m)
    eta = float(0.8 / L)
    slow_ar = float(1.0 - eta * m)

    post_corr = float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
    kl_mf = kl_gaussian(mean, D_mf, mean, cov)

    row: dict[str, float] = {
        "rho_x": rho,
        "tau": tau,
        "prior_precision": 1.0 / tau**2,
        "posterior_mean_beta1": float(mean[0]),
        "posterior_mean_beta2": float(mean[1]),
        "posterior_corr_beta1_beta2": post_corr,
        "lambda_min_m": m,
        "lambda_max_L": L,
        "kappa_post": kappa,
        "eta_ula": eta,
        "slow_ar_coefficient": slow_ar,
        "posterior_var_beta1": float(cov[0, 0]),
        "posterior_var_beta2": float(cov[1, 1]),
        "mf_var_beta1": float(D_mf[0, 0]),
        "mf_var_beta2": float(D_mf[1, 1]),
        "mf_to_post_var_ratio_beta1": float(D_mf[0, 0] / cov[0, 0]),
        "mf_to_post_var_ratio_beta2": float(D_mf[1, 1] / cov[1, 1]),
        "kl_mf_to_posterior": kl_mf,
    }

    if run_ula:
        rng = np.random.default_rng(cfg.seed + int(10000 * rho) + int(1000 * tau))
        samples = simulate_ula_gaussian(
            mean=mean,
            precision=precision,
            eta=eta,
            iterations=cfg.ula_iterations,
            burnin=cfg.ula_burnin,
            rng=rng,
        )
        centered = samples - mean
        rotated = centered @ eigvecs
        slow = rotated[:, 0]
        fast = rotated[:, -1]
        acf_slow = autocorrelation_1d(slow, cfg.max_acf_lag)
        acf_fast = autocorrelation_1d(fast, cfg.max_acf_lag)
        row.update(
            {
                "ess_slow": ess_from_acf(acf_slow, len(slow)),
                "ess_fast": ess_from_acf(acf_fast, len(fast)),
                "acf_slow_lag_100": float(acf_slow[100]) if cfg.max_acf_lag >= 100 else np.nan,
                "theoretical_acf_slow_lag_100": float(slow_ar**100) if cfg.max_acf_lag >= 100 else np.nan,
            }
        )
    else:
        row.update(
            {
                "ess_slow": np.nan,
                "ess_fast": np.nan,
                "acf_slow_lag_100": np.nan,
                "theoretical_acf_slow_lag_100": np.nan,
            }
        )

    return row


def make_summary_tables(results: pd.DataFrame, output_dir: Path) -> None:
    """Write compact CSV tables for thesis drafting."""
    results.to_csv(output_dir / "experiment2_full_results.csv", index=False)

    geometry_cols = [
        "rho_x",
        "tau",
        "posterior_corr_beta1_beta2",
        "lambda_min_m",
        "lambda_max_L",
        "kappa_post",
        "kl_mf_to_posterior",
    ]
    results[geometry_cols].to_csv(output_dir / "experiment2_geometry_summary.csv", index=False)

    variance_cols = [
        "rho_x",
        "tau",
        "posterior_var_beta1",
        "mf_var_beta1",
        "mf_to_post_var_ratio_beta1",
        "posterior_var_beta2",
        "mf_var_beta2",
        "mf_to_post_var_ratio_beta2",
    ]
    results[variance_cols].to_csv(output_dir / "experiment2_variance_summary.csv", index=False)

    ula_cols = [
        "rho_x",
        "tau",
        "kappa_post",
        "eta_ula",
        "slow_ar_coefficient",
        "ess_slow",
        "ess_fast",
        "acf_slow_lag_100",
        "theoretical_acf_slow_lag_100",
    ]
    results[ula_cols].to_csv(output_dir / "experiment2_ula_summary.csv", index=False)


def plot_kappa_vs_rho(results: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for tau, group in results.groupby("tau", sort=True):
        group = group.sort_values("rho_x")
        ax.plot(group["rho_x"], group["kappa_post"], marker="o", label=fr"$\tau={tau:g}$")
    ax.set_xlabel(r"design correlation $\rho_X$")
    ax.set_ylabel(r"posterior condition number $\kappa_{post}$")
    # Figure-level title removed: the LaTeX caption already describes the figure.
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "experiment2_kappa_vs_rho.pdf")
    fig.savefig(output_dir / "experiment2_kappa_vs_rho.png", dpi=200)
    plt.close(fig)


def plot_variance_ratio_vs_rho(results: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for tau, group in results.groupby("tau", sort=True):
        group = group.sort_values("rho_x")
        ax.plot(
            group["rho_x"],
            group["mf_to_post_var_ratio_beta1"],
            marker="o",
            label=fr"$\tau={tau:g}$",
        )
    ax.set_xlabel(r"design correlation $\rho_X$")
    ax.set_ylabel(r"$\mathrm{Var}_{q^*}(\beta_1) / \mathrm{Var}_{\pi}(\beta_1)$")
    # Figure-level title removed: the LaTeX caption already describes the figure.
    ax.set_ylim(bottom=0.0, top=1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "experiment2_variance_ratio_vs_rho.pdf")
    fig.savefig(output_dir / "experiment2_variance_ratio_vs_rho.png", dpi=200)
    plt.close(fig)


def plot_posterior_vs_mf_variance(results: pd.DataFrame, output_dir: Path) -> None:
    """Plot exact posterior and MF variances for beta_1 as rho changes."""
    for tau, group in results.groupby("tau", sort=True):
        group = group.sort_values("rho_x")
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(group["rho_x"], group["posterior_var_beta1"], marker="o", label=r"exact posterior")
        ax.plot(group["rho_x"], group["mf_var_beta1"], marker="s", linestyle="--", label=r"mean-field optimum")
        ax.set_xlabel(r"design correlation $\rho_X$")
        ax.set_ylabel(r"marginal variance of $\beta_1$")
        ax.set_title(fr"Posterior vs mean-field variance, $\tau={tau:g}$")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        stem = f"experiment2_variance_beta1_tau_{tau:g}".replace(".", "p")
        fig.savefig(output_dir / f"{stem}.pdf")
        fig.savefig(output_dir / f"{stem}.png", dpi=200)
        plt.close(fig)


def plot_posterior_correlation(results: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for tau, group in results.groupby("tau", sort=True):
        group = group.sort_values("rho_x")
        ax.plot(group["rho_x"], group["posterior_corr_beta1_beta2"], marker="o", label=fr"$\tau={tau:g}$")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel(r"design correlation $\rho_X$")
    ax.set_ylabel(r"posterior correlation $\mathrm{corr}(\beta_1,\beta_2 \mid y)$")
    ax.set_title("Positive design collinearity induces negative coefficient dependence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "experiment2_posterior_correlation.pdf")
    fig.savefig(output_dir / "experiment2_posterior_correlation.png", dpi=200)
    plt.close(fig)


def plot_contours_for_selected_tau(
    cfg: ExperimentConfig,
    output_dir: Path,
    selected_tau: float = 1.0,
) -> None:
    """Optional contour plot for a representative prior scale.

    This is intentionally secondary. Its role is visual intuition, not the main
    evidence of the experiment.
    """
    rng = np.random.default_rng(cfg.seed)
    y_noise = rng.normal(size=cfg.n)
    beta_true = np.array([cfg.beta_true_1, cfg.beta_true_2])

    # Layout: 2x2 grid when there are exactly four collinearity levels, otherwise
    # fall back to a single row. The 2x2 grid gives each panel roughly six
    # centimetres at \includegraphics[width=\textwidth] in the thesis, which is
    # what makes the contour comparison legible.
    n_panels = len(cfg.rho_grid)
    if n_panels == 4:
        nrows, ncols = 2, 2
        # figsize (9, 9) keeps the 2x2 contour grid visually balanced with the
        # rest of Chapter 4. With \includegraphics[width=0.85\textwidth] this
        # lands at roughly 11 cm in the thesis, similar in vertical span to the
        # single-panel ACF and kappa figures.
        fig, axes = plt.subplots(nrows, ncols, figsize=(9.0, 9.0))
        axes = axes.flatten()
    else:
        nrows, ncols = 1, n_panels
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * n_panels, 4.0))
        if n_panels == 1:
            axes = [axes]
        else:
            axes = np.asarray(axes).flatten()

    for ax, rho in zip(axes, cfg.rho_grid):
        X = build_exact_design(cfg.n, rho)
        y = X @ beta_true + cfg.sigma * y_noise
        mean, cov, precision = posterior_from_regression(X, y, cfg.sigma, selected_tau)
        _, D_mf = gaussian_mf_forward_kl_optimum(mean, precision)

        # Build a local grid around the exact posterior.
        std1 = np.sqrt(cov[0, 0])
        std2 = np.sqrt(cov[1, 1])
        x1 = np.linspace(mean[0] - 3.0 * std1, mean[0] + 3.0 * std1, 220)
        x2 = np.linspace(mean[1] - 3.0 * std2, mean[1] + 3.0 * std2, 220)
        X1, X2 = np.meshgrid(x1, x2)
        grid = np.column_stack([X1.ravel(), X2.ravel()])

        def quad_density(points: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
            prec = np.linalg.inv(Sigma)
            centered = points - mu
            q = np.sum((centered @ prec) * centered, axis=1)
            return np.exp(-0.5 * q).reshape(X1.shape)

        Z_post = quad_density(grid, mean, cov)
        Z_mf = quad_density(grid, mean, D_mf)
        levels = np.exp(-0.5 * np.array([4.0, 2.0, 1.0]))
        ax.contour(X1, X2, Z_post, levels=levels)
        ax.contour(X1, X2, Z_mf, levels=levels, linestyles="--")
        ax.scatter([mean[0]], [mean[1]], s=15)
        ax.set_title(fr"$\rho_X={rho:g}$")
        ax.set_xlabel(r"$\beta_1$")
        ax.set_ylabel(r"$\beta_2$")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    # Figure-level suptitle removed: the LaTeX caption already describes the figure.
    fig.tight_layout()
    stem = f"experiment2_contours_tau_{selected_tau:g}".replace(".", "p")
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png", dpi=200)
    plt.close(fig)


def plot_ula_acf_for_selected_tau(
    cfg: ExperimentConfig,
    output_dir: Path,
    selected_tau: float = 1.0,
) -> None:
    """Optional ULA ACF plot for a representative prior scale."""
    rng_global = np.random.default_rng(cfg.seed)
    y_noise = rng_global.normal(size=cfg.n)
    beta_true = np.array([cfg.beta_true_1, cfg.beta_true_2])

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    lags = np.arange(cfg.max_acf_lag + 1)
    for rho in cfg.rho_grid:
        X = build_exact_design(cfg.n, rho)
        y = X @ beta_true + cfg.sigma * y_noise
        mean, cov, precision = posterior_from_regression(X, y, cfg.sigma, selected_tau)
        eigvals, eigvecs = np.linalg.eigh(precision)
        m = float(eigvals[0])
        L = float(eigvals[-1])
        kappa = L / m
        eta = 0.8 / L
        slow_ar = 1.0 - eta * m

        rng = np.random.default_rng(cfg.seed + int(10000 * rho) + int(1000 * selected_tau))
        samples = simulate_ula_gaussian(
            mean=mean,
            precision=precision,
            eta=eta,
            iterations=cfg.ula_iterations,
            burnin=cfg.ula_burnin,
            rng=rng,
        )
        rotated = (samples - mean) @ eigvecs
        acf_slow = autocorrelation_1d(rotated[:, 0], cfg.max_acf_lag)
        ax.plot(lags, acf_slow, label=fr"empirical $\rho_X={rho:g}$, $\kappa={kappa:.1f}$")
        ax.plot(lags, slow_ar**lags, linestyle="--", linewidth=1.0)

    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    # Figure-level title removed: the LaTeX caption already describes the figure.
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    stem = f"experiment2_ula_acf_tau_{selected_tau:g}".replace(".", "p")
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png", dpi=200)
    plt.close(fig)


def run_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    # Use the same noise vector across cases so that changes in geometry are due
    # to rho and tau. Headline geometric quantities do not depend on y, but the
    # posterior mean and contour locations do.
    y_noise = rng.normal(size=cfg.n)

    rows = []
    for tau in cfg.tau_grid:
        for rho in cfg.rho_grid:
            # ULA is run for all cases because p=2 and it is cheap. If runtime is
            # an issue, set run_ula=False here and rely on analytical AR factors.
            rows.append(summarize_one_case(rho, tau, cfg, y_noise, run_ula=True))

    results = pd.DataFrame(rows).sort_values(["tau", "rho_x"]).reset_index(drop=True)
    make_summary_tables(results, output_dir)

    plot_kappa_vs_rho(results, output_dir)
    plot_variance_ratio_vs_rho(results, output_dir)
    plot_posterior_vs_mf_variance(results, output_dir)
    plot_posterior_correlation(results, output_dir)

    # Secondary visualizations for a representative prior scale.
    # tau=1.0 is chosen because it is more Bayesian than tau=10 but still lets
    # collinearity remain visible. Change selected_tau if needed.
    plot_contours_for_selected_tau(cfg, output_dir, selected_tau=1.0)
    plot_ula_acf_for_selected_tau(cfg, output_dir, selected_tau=1.0)

    return results


def main() -> None:
    cfg = ExperimentConfig()
    results = run_experiment(cfg)

    print("Experiment completed.")
    print(f"Outputs written to: {Path(cfg.output_dir).resolve()}")
    print("\nMain summary:")
    display_cols = [
        "rho_x",
        "tau",
        "posterior_corr_beta1_beta2",
        "kappa_post",
        "posterior_var_beta1",
        "mf_var_beta1",
        "mf_to_post_var_ratio_beta1",
        "kl_mf_to_posterior",
        "ess_slow",
    ]
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(results[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
