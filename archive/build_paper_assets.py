from __future__ import annotations

import json
import math
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "crypto_stage1_processed"
ASSET_DIR = PROJECT_DIR / "paper_assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)


EXPERIMENTS = [
    {
        "experiment": "crypto_student_t_k3_raw",
        "label": "Crypto Student-t, K=3, raw returns",
        "root": DATA_DIR / "sgmcmc_results",
        "methods": {
            "uniform": "uniform/L10_B10_M4_steps2500_burn750_save10_eta0p0001_gamma0p55_t0100_gclip2500",
            "young_static": "young_static/L10_B10_Me2_Mt2_steps2000_burn500_save10_eta0p0001_gamma0p55_t0100_gclip2500",
            "online_feature": "online_feature/L10_B10_Me2_Mt2_steps2000_burn500_save10_warm300_eta0p0001_gamma0p55_t0100_eps0p05_ridge5_disc0p995_gclip2500",
        },
    },
    {
        "experiment": "crypto_gaussian_k2_rollstd",
        "label": "Crypto Gaussian, K=2, 30-day rolling standardized",
        "root": DATA_DIR / "gaussian_k2_rollstd_comparison",
        "methods": {
            "uniform": "uniform/uniform_k2_l10_b10_n_steps500_burn_in150_save_every10_eta00p0001_gamma0p55_t0100",
            "young_static": "young_static/young_static_k2_l10_b10_n_steps500_burn_in150_save_every10_eta00p0001_gamma0p55_t0100",
            "online_feature": "online_feature/online_feature_k2_l10_b10_n_steps500_burn_in150_save_every10_eta00p0001_gamma0p55_t0100",
        },
    },
    {
        "experiment": "crypto_btc_eth_gaussian_k2_fullcov",
        "label": "BTC/ETH Gaussian, K=2, full covariance",
        "root": DATA_DIR / "gaussian_k2_btc_eth_rollstd_fullcov_comparison",
        "methods": {
            "uniform": "uniform/uniform_k2_l10_b10_n_steps500_burn_in150_save_every10_eta00p0001_gamma0p55_t0100",
            "young_static": "young_static/young_static_k2_l10_b10_n_steps500_burn_in150_save_every10_eta00p0001_gamma0p55_t0100",
            "online_feature": "online_feature/online_feature_k2_l10_b10_n_steps500_burn_in150_save_every10_eta00p0001_gamma0p55_t0100",
        },
    },
    {
        "experiment": "simulated_gaussian_k2_fullcov",
        "label": "Simulated Gaussian, K=2, two-dimensional full covariance",
        "root": DATA_DIR / "simulated_k2_gaussian_fullcov_part_a",
        "methods": {
            "uniform": "uniform/uniform_k2_l8_b8_n_steps1200_burn_in300_save_every10_eta07em05_gamma0p55_t0100",
            "young_static": "young_static/young_static_k2_l8_b8_n_steps1200_burn_in300_save_every10_eta07em05_gamma0p55_t0100",
            "online_feature": "online_feature/online_feature_k2_l8_b8_n_steps1200_burn_in300_save_every10_eta07em05_gamma0p55_t0100",
        },
    },
    {
        "experiment": "simulated_gaussian_k2_univariate",
        "label": "Simulated Gaussian, K=2, univariate",
        "root": DATA_DIR / "simulated_k2_univariate_part_a",
        "methods": {
            "uniform": "uniform/uniform_k2_l10_b10_n_steps1500_burn_in400_save_every10_eta05em05_gamma0p55_t0100",
            "young_static": "young_static/young_static_k2_l10_b10_n_steps1500_burn_in400_save_every10_eta05em05_gamma0p55_t0100",
            "online_feature": "online_feature/online_feature_k2_l10_b10_n_steps1500_burn_in400_save_every10_eta05em05_gamma0p55_t0100",
        },
    },
]


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table type: {path}")


def chain_frame_to_idata(df: pd.DataFrame, variables: list[str]) -> az.InferenceData:
    chains = sorted(df["chain"].unique())
    min_draws = min((df.loc[df["chain"] == chain].shape[0] for chain in chains))
    posterior = {}
    for var in variables:
        arr = []
        for chain in chains:
            sub = df.loc[df["chain"] == chain].sort_values("step")
            arr.append(sub[var].to_numpy(dtype=float)[:min_draws])
        posterior[var] = np.asarray(arr)
    return az.from_dict(posterior=posterior)


def xarray_scalar_table(dataset, value_name: str) -> pd.DataFrame:
    rows = []
    for var in dataset.data_vars:
        value = np.asarray(dataset[var].values).reshape(-1)
        if value.size == 1 and np.isfinite(value[0]):
            rows.append({"variable": str(var), value_name: float(value[0])})
    return pd.DataFrame(rows)


def compute_sg_diagnostics(samples: pd.DataFrame) -> pd.DataFrame:
    exclude = {"step", "accepted", "bad_step_streak", "chain"}
    variables = [
        col
        for col in samples.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(samples[col])
    ]
    idata = chain_frame_to_idata(samples, variables)
    rhat_df = xarray_scalar_table(az.rhat(idata), "rhat")
    ess_bulk_df = xarray_scalar_table(az.ess(idata, method="bulk"), "ess_bulk")
    ess_tail_df = xarray_scalar_table(az.ess(idata, method="tail"), "ess_tail")
    out = rhat_df.merge(ess_bulk_df, on="variable", how="outer")
    out = out.merge(ess_tail_df, on="variable", how="outer")
    return out.sort_values(["rhat", "variable"], ascending=[False, True]).reset_index(drop=True)


def summarize_samples(samples: pd.DataFrame) -> pd.DataFrame:
    exclude = {"step", "accepted", "bad_step_streak", "chain"}
    variables = [
        col
        for col in samples.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(samples[col])
    ]
    rows = []
    for var in variables:
        values = samples[var].to_numpy(dtype=float)
        rows.append(
            {
                "variable": var,
                "mean": np.nanmean(values),
                "sd": np.nanstd(values, ddof=1),
                "q05": np.nanquantile(values, 0.05),
                "median": np.nanquantile(values, 0.50),
                "q95": np.nanquantile(values, 0.95),
            }
        )
    return pd.DataFrame(rows)


def summarize_occupancy(smoothed: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    prob_cols = [c for c in smoothed.columns if c.startswith("state_") and c.endswith("_prob")]
    for col in prob_cols:
        state = col.removeprefix("state_").removesuffix("_prob")
        out[f"avg_state_{state}_prob"] = float(smoothed[col].mean())
    if "ml_state" in smoothed.columns:
        shares = smoothed["ml_state"].value_counts(normalize=True)
        for state, value in shares.items():
            out[f"ml_state_{int(state)}_share"] = float(value)
    return out


def collect_sg_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diag_rows = []
    posterior_rows = []
    occupancy_rows = []

    for experiment in EXPERIMENTS:
        for method, rel in experiment["methods"].items():
            run_dir = experiment["root"] / rel
            samples_path = run_dir / "summaries" / "posterior_summary_samples_all_chains.pkl"
            runtime_path = run_dir / "summaries" / "chain_runtime_summary.pkl"
            smooth_path = run_dir / "summaries" / "smoothed_state_probs.pkl"
            if not samples_path.exists():
                continue

            samples = load_table(samples_path)
            diagnostics = compute_sg_diagnostics(samples)
            posterior = summarize_samples(samples)
            runtime_mean = np.nan
            if runtime_path.exists():
                runtime = load_table(runtime_path)
                runtime_mean = float(runtime["runtime_seconds"].mean())

            chains = int(samples["chain"].nunique())
            draws_per_chain = int(samples.groupby("chain").size().min())
            diag_rows.append(
                {
                    "experiment": experiment["experiment"],
                    "label": experiment["label"],
                    "method": method,
                    "chains": chains,
                    "draws_per_chain": draws_per_chain,
                    "runtime_sec_mean": runtime_mean,
                    "max_rhat": float(diagnostics["rhat"].max()),
                    "median_rhat": float(diagnostics["rhat"].median()),
                    "min_ess_bulk": float(diagnostics["ess_bulk"].min()),
                    "median_ess_bulk": float(diagnostics["ess_bulk"].median()),
                    "min_ess_tail": float(diagnostics["ess_tail"].min()),
                    "diagnostic_file": str(run_dir / "summaries" / "computed_diagnostics.csv"),
                }
            )
            diagnostics.to_csv(run_dir / "summaries" / "computed_diagnostics.csv", index=False)

            posterior.insert(0, "method", method)
            posterior.insert(0, "experiment", experiment["experiment"])
            posterior_rows.append(posterior)

            if smooth_path.exists():
                occ = summarize_occupancy(load_table(smooth_path))
                occ.update({"experiment": experiment["experiment"], "method": method})
                occupancy_rows.append(occ)

    diagnostics_all = pd.DataFrame(diag_rows)
    posterior_all = pd.concat(posterior_rows, ignore_index=True)
    occupancy_all = pd.DataFrame(occupancy_rows)
    return diagnostics_all, posterior_all, occupancy_all


def collect_stan_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    post_rows = []
    occ_rows = []
    stan_specs = [
        ("stan_k2_rollstd", "Stan Gaussian, K=2, rolling standardized", DATA_DIR / "stan_k2_rollstd_comparison"),
        ("stan_k3_rollstd", "Stan Gaussian, K=3, rolling standardized", DATA_DIR / "stan_k3_rollstd_comparison"),
    ]
    for experiment, label, root in stan_specs:
        summary_path = root / "stan_summary.csv"
        posterior_path = root / "stan_posterior_summary.csv"
        smooth_path = root / "stan_smoothed_state_probs.pkl"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path).rename(columns={"Unnamed: 0": "variable"})
        core_mask = summary["variable"].str.match(r"^(A|pi|mu|log_sigma_level|sigma)\[")
        core = summary.loc[core_mask].copy()
        if core.empty:
            core = summary.loc[~summary["variable"].str.startswith(("logalpha", "gamma", "state_prob"))].copy()
        rows.append(
            {
                "experiment": experiment,
                "label": label,
                "method": "stan_nuts",
                "n_parameters_reported": int(core.shape[0]),
                "max_rhat": float(core["R_hat"].max()),
                "median_rhat": float(core["R_hat"].median()),
                "min_ess_bulk": float(core["ESS_bulk"].min()),
                "median_ess_bulk": float(core["ESS_bulk"].median()),
                "min_ess_tail": float(core["ESS_tail"].min()),
            }
        )
        if posterior_path.exists():
            posterior = pd.read_csv(posterior_path)
            posterior.insert(0, "method", "stan_nuts")
            posterior.insert(0, "experiment", experiment)
            post_rows.append(posterior)
        if smooth_path.exists():
            occ = summarize_occupancy(pd.read_pickle(smooth_path))
            occ.update({"experiment": experiment, "method": "stan_nuts"})
            occ_rows.append(occ)
    return pd.DataFrame(rows), pd.concat(post_rows, ignore_index=True), pd.DataFrame(occ_rows)


def latex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def save_latex_table(df: pd.DataFrame, path: Path, columns: list[str], float_fmt: str = "%.2f") -> None:
    formatted = df.loc[:, columns].copy()
    for col in formatted.columns:
        if formatted[col].dtype == object:
            formatted[col] = formatted[col].map(lambda x: latex_escape(str(x)))
    formatted = formatted.replace({np.nan: "--"})
    formatted.columns = [latex_escape(str(col)) for col in formatted.columns]
    path.write_text(
        formatted.to_latex(index=False, escape=False, float_format=lambda x: float_fmt % x),
        encoding="utf-8",
    )


def plot_k2_crypto_state_probs() -> None:
    specs = [
        ("Stan", DATA_DIR / "stan_k2_rollstd_comparison" / "stan_smoothed_state_probs.pkl"),
        (
            "Uniform",
            DATA_DIR
            / "gaussian_k2_rollstd_comparison/uniform/uniform_k2_l10_b10_n_steps500_burn_in150_save_every10_eta00p0001_gamma0p55_t0100/summaries/smoothed_state_probs.pkl",
        ),
        (
            "TASS-static",
            DATA_DIR
            / "gaussian_k2_rollstd_comparison/young_static/young_static_k2_l10_b10_n_steps500_burn_in150_save_every10_eta00p0001_gamma0p55_t0100/summaries/smoothed_state_probs.pkl",
        ),
        (
            "Online",
            DATA_DIR
            / "gaussian_k2_rollstd_comparison/online_feature/online_feature_k2_l10_b10_n_steps500_burn_in150_save_every10_eta00p0001_gamma0p55_t0100/summaries/smoothed_state_probs.pkl",
        ),
    ]
    fig, axes = plt.subplots(len(specs), 1, figsize=(10, 6.8), sharex=True, sharey=True)
    for ax, (name, path) in zip(axes, specs):
        df = pd.read_pickle(path)
        date = pd.to_datetime(df["date"])
        prob_col = "state_1_prob" if "state_1_prob" in df.columns else [c for c in df.columns if c.endswith("_prob")][-1]
        ax.plot(date, df[prob_col], lw=0.9, color="#2f5d62")
        ax.fill_between(date, 0, df[prob_col].to_numpy(dtype=float), color="#9cc5a1", alpha=0.35)
        ax.set_ylabel(name)
        ax.grid(alpha=0.2, linewidth=0.5)
    axes[-1].set_xlabel("Date")
    axes[0].set_title("Posterior smoothed probability of the high-volatility crypto regime")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "crypto_k2_state_probabilities.pdf")
    plt.close(fig)


def simulate_univariate_truth() -> pd.DataFrame:
    rng = np.random.default_rng(221)
    T = 800
    true_a = np.array([[0.985, 0.015], [0.025, 0.975]], dtype=float)
    true_pi = np.array([0.65, 0.35], dtype=float)
    true_mu = np.array([[0.05], [-0.35]], dtype=float)
    true_sigma = np.array([[[0.35**2]], [[1.10**2]]], dtype=float)
    z = np.zeros(T, dtype=int)
    z[0] = rng.choice(2, p=true_pi)
    for t in range(1, T):
        z[t] = rng.choice(2, p=true_a[z[t - 1]])
    y = np.stack([rng.multivariate_normal(true_mu[state], true_sigma[state]) for state in z])[:, 0]
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=T, freq="D"), "y": y, "true_state": z})


def plot_simulated_univariate_recovery() -> None:
    truth = simulate_univariate_truth()
    specs = [
        (
            "Uniform",
            DATA_DIR
            / "simulated_k2_univariate_part_a/uniform/uniform_k2_l10_b10_n_steps1500_burn_in400_save_every10_eta05em05_gamma0p55_t0100/summaries/smoothed_state_probs.pkl",
        ),
        (
            "TASS-static",
            DATA_DIR
            / "simulated_k2_univariate_part_a/young_static/young_static_k2_l10_b10_n_steps1500_burn_in400_save_every10_eta05em05_gamma0p55_t0100/summaries/smoothed_state_probs.pkl",
        ),
        (
            "Online",
            DATA_DIR
            / "simulated_k2_univariate_part_a/online_feature/online_feature_k2_l10_b10_n_steps1500_burn_in400_save_every10_eta05em05_gamma0p55_t0100/summaries/smoothed_state_probs.pkl",
        ),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(truth["date"], truth["y"], lw=0.8, color="#555555")
    axes[0].scatter(
        truth.loc[truth["true_state"] == 1, "date"],
        truth.loc[truth["true_state"] == 1, "y"],
        s=8,
        color="#d95f02",
        alpha=0.45,
        label="true stress state",
    )
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].set_title("Univariate simulation: observed series and inferred stress probabilities")
    for ax, (name, path) in zip(axes[1:], specs):
        df = pd.read_pickle(path)
        date = pd.to_datetime(df["date"])
        ax.plot(date, df["state_1_prob"], lw=0.9, color="#1b4965")
        ax.fill_between(date, 0, truth["true_state"], color="#d95f02", alpha=0.12, step="mid")
        ax.set_ylabel(name)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2, linewidth=0.5)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "simulated_univariate_recovery.pdf")
    plt.close(fig)


def plot_mixing_summary(diagnostics_all: pd.DataFrame, stan_diag: pd.DataFrame) -> None:
    sg = diagnostics_all.loc[
        diagnostics_all["experiment"].isin(
            [
                "crypto_gaussian_k2_rollstd",
                "crypto_btc_eth_gaussian_k2_fullcov",
                "simulated_gaussian_k2_fullcov",
                "simulated_gaussian_k2_univariate",
            ]
        )
    ].copy()
    sg["short"] = sg["experiment"].map(
        {
            "crypto_gaussian_k2_rollstd": "Crypto K=2",
            "crypto_btc_eth_gaussian_k2_fullcov": "BTC/ETH K=2",
            "simulated_gaussian_k2_fullcov": "Sim 2D",
            "simulated_gaussian_k2_univariate": "Sim 1D",
        }
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(sg.shape[0])
    colors = sg["method"].map({"uniform": "#4c78a8", "young_static": "#f58518", "online_feature": "#54a24b"}).fillna("#999999")
    ax.bar(x, sg["max_rhat"], color=colors)
    ax.axhline(1.01, color="#222222", linestyle="--", lw=1, label="Stan-style convergence target")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{b}" for a, b in zip(sg["short"], sg["method"])], rotation=45, ha="right")
    ax.set_ylabel("Maximum rank-normalized R-hat")
    ax.set_title("SGLD chains remained poorly mixed across application and simulation benchmarks")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "sgmcmc_mixing_summary.pdf")
    plt.close(fig)


def plot_online_usefulness_region() -> None:
    cv = np.linspace(0, 3, 250)
    eps = np.linspace(0, 1.2, 250)
    cv_grid, eps_grid = np.meshgrid(cv, eps)
    explore = 0.05
    useful = (np.exp(2 * eps_grid) / (1 - explore)) < (1 + cv_grid**2)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.contourf(cv_grid, eps_grid, useful.astype(float), levels=[-0.1, 0.5, 1.1], colors=["#f3d8c7", "#c7e9c0"], alpha=0.85)
    boundary = 0.5 * np.log((1 - explore) * (1 + cv**2))
    boundary[boundary < 0] = np.nan
    ax.plot(cv, boundary, color="#2f3e46", lw=2)
    ax.set_xlabel("Coefficient of variation of block gradient norms")
    ax.set_ylabel("Uniform log-norm prediction error bound $\\epsilon$")
    ax.set_title("When feature-informed online sampling can beat uniform sampling")
    ax.text(1.85, 0.25, "online useful", fontsize=10)
    ax.text(0.35, 0.85, "uniform safer", fontsize=10)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "online_usefulness_region.pdf")
    plt.close(fig)


def make_tables(diagnostics_all: pd.DataFrame, posterior_all: pd.DataFrame, occupancy_all: pd.DataFrame, stan_diag: pd.DataFrame, stan_post: pd.DataFrame, stan_occ: pd.DataFrame) -> None:
    diagnostics_all.to_csv(ASSET_DIR / "sgmcmc_diagnostics_all.csv", index=False)
    posterior_all.to_csv(ASSET_DIR / "sgmcmc_posterior_summaries_all.csv", index=False)
    occupancy_all.to_csv(ASSET_DIR / "sgmcmc_occupancy_all.csv", index=False)
    stan_diag.to_csv(ASSET_DIR / "stan_diagnostics.csv", index=False)
    stan_post.to_csv(ASSET_DIR / "stan_posterior_summaries.csv", index=False)
    stan_occ.to_csv(ASSET_DIR / "stan_occupancy.csv", index=False)

    display_diag = diagnostics_all[
        diagnostics_all["experiment"].isin(
            ["crypto_gaussian_k2_rollstd", "simulated_gaussian_k2_fullcov", "simulated_gaussian_k2_univariate"]
        )
    ].copy()
    display_diag["experiment"] = display_diag["experiment"].map(
        {
            "crypto_gaussian_k2_rollstd": "Crypto K=2",
            "simulated_gaussian_k2_fullcov": "Simulated 2D",
            "simulated_gaussian_k2_univariate": "Simulated 1D",
        }
    )
    display_diag = display_diag[
        ["experiment", "method", "draws_per_chain", "runtime_sec_mean", "max_rhat", "median_rhat", "min_ess_bulk"]
    ]
    save_latex_table(
        display_diag,
        ASSET_DIR / "table_sgmcmc_diagnostics.tex",
        ["experiment", "method", "draws_per_chain", "runtime_sec_mean", "max_rhat", "median_rhat", "min_ess_bulk"],
    )

    stan_display = stan_diag[["experiment", "method", "max_rhat", "median_rhat", "min_ess_bulk", "median_ess_bulk"]].copy()
    stan_display["experiment"] = stan_display["experiment"].map(
        {"stan_k2_rollstd": "Stan K=2", "stan_k3_rollstd": "Stan K=3"}
    )
    save_latex_table(
        stan_display,
        ASSET_DIR / "table_stan_diagnostics.tex",
        ["experiment", "method", "max_rhat", "median_rhat", "min_ess_bulk", "median_ess_bulk"],
    )

    occ = pd.concat([occupancy_all, stan_occ], ignore_index=True, sort=False)
    occ_display = occ[
        occ["experiment"].isin(
            ["crypto_student_t_k3_raw", "crypto_gaussian_k2_rollstd", "stan_k2_rollstd", "stan_k3_rollstd"]
        )
    ].copy()
    keep = [c for c in ["experiment", "method", "avg_state_0_prob", "avg_state_1_prob", "avg_state_2_prob", "ml_state_0_share", "ml_state_1_share", "ml_state_2_share"] if c in occ_display.columns]
    save_latex_table(occ_display[keep], ASSET_DIR / "table_occupancy.tex", keep, float_fmt="%.3f")

    k2_post = posterior_all[
        (posterior_all["experiment"] == "crypto_gaussian_k2_rollstd")
        & (posterior_all["variable"].isin(["pi_0", "pi_1", "A_00", "A_11", "trace_Sigma_0", "trace_Sigma_1", "mu_0_0", "mu_1_0"]))
    ].copy()
    stan_k2 = stan_post[
        (stan_post["experiment"] == "stan_k2_rollstd")
        & (stan_post["variable"].isin(["pi_0", "pi_1", "A_00", "A_11", "mean_sigma_0", "mean_sigma_1", "mu_0_0", "mu_1_0"]))
    ].copy()
    combined = pd.concat([k2_post, stan_k2], ignore_index=True, sort=False)
    save_latex_table(
        combined[["method", "variable", "mean", "sd", "q05", "median", "q95"]],
        ASSET_DIR / "table_crypto_k2_posterior.tex",
        ["method", "variable", "mean", "sd", "q05", "median", "q95"],
        float_fmt="%.3f",
    )


def write_summary_json(diagnostics_all: pd.DataFrame, occupancy_all: pd.DataFrame, stan_diag: pd.DataFrame, stan_occ: pd.DataFrame) -> None:
    summary = {
        "n_sgmcmc_runs": int(diagnostics_all.shape[0]),
        "sgmcmc_worst_max_rhat": float(diagnostics_all["max_rhat"].max()),
        "sgmcmc_best_max_rhat": float(diagnostics_all["max_rhat"].min()),
        "stan": stan_diag.to_dict(orient="records"),
        "key_occupancies": pd.concat([occupancy_all, stan_occ], ignore_index=True, sort=False).to_dict(orient="records"),
    }
    (ASSET_DIR / "paper_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    diagnostics_all, posterior_all, occupancy_all = collect_sg_results()
    stan_diag, stan_post, stan_occ = collect_stan_results()
    make_tables(diagnostics_all, posterior_all, occupancy_all, stan_diag, stan_post, stan_occ)
    plot_k2_crypto_state_probs()
    plot_simulated_univariate_recovery()
    plot_mixing_summary(diagnostics_all, stan_diag)
    plot_online_usefulness_region()
    write_summary_json(diagnostics_all, occupancy_all, stan_diag, stan_occ)
    print("wrote", ASSET_DIR)
    print(diagnostics_all[["experiment", "method", "max_rhat", "min_ess_bulk", "runtime_sec_mean"]].to_string(index=False))
    print(stan_diag[["experiment", "max_rhat", "min_ess_bulk"]].to_string(index=False))


if __name__ == "__main__":
    main()
