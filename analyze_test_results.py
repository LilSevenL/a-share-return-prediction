"""只读分析永久Test结果，生成风险归因表与论文图表。

本脚本不会训练模型、生成新预测或修改永久Test目录中的原始结果。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = PROJECT_DIR / "quant_data" / "processed" / "permanent_test" / "evaluation"
DEFAULT_OUTPUT = PROJECT_DIR / "quant_data" / "processed" / "result_analysis"
MODEL_LABELS = {
    "mlp_primary": "MLP (Primary)",
    "linear_baseline": "Elastic Net (Baseline)",
    "lightgbm_challenger": "LightGBM (Challenger)",
}
BENCHMARK_LABELS = {
    "000300.SH": "CSI 300",
    "000905.SH": "CSI 500",
    "000852.SH": "CSI 1000",
}
COLORS = {
    "mlp_primary": "#2563eb",
    "linear_baseline": "#059669",
    "lightgbm_challenger": "#d97706",
    "000300.SH": "#7c3aed",
    "000905.SH": "#dc2626",
    "000852.SH": "#475569",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def configure_plotting() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
    })


def load_inputs() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    summary = json.loads((EVALUATION_DIR / "permanent_test_summary.json").read_text(encoding="utf-8"))
    nav = pd.read_parquet(EVALUATION_DIR / "daily_nav.parquet")
    nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    nav = nav.sort_values(["trade_date", "model"])

    index_data = pd.read_csv(
        PROJECT_DIR / "quant_data" / "combined" / "index_daily.csv",
        usecols=["ts_code", "trade_date", "close"],
    )
    index_data = index_data[index_data["ts_code"].isin(BENCHMARK_LABELS)].copy()
    index_data["trade_date"] = pd.to_datetime(index_data["trade_date"].astype(str))
    index_data = index_data.sort_values(["trade_date", "ts_code"])
    return summary, nav, index_data


def max_drawdown_episode(nav: pd.Series) -> dict[str, Any]:
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    trough = drawdown.idxmin()
    peak = nav.loc[:trough].idxmax()
    after = nav.loc[trough:]
    recovered = after[after >= nav.loc[peak]]
    recovery = recovered.index[0] if len(recovered) else pd.NaT
    end = recovery if pd.notna(recovery) else nav.index[-1]
    return {
        "peak_date": peak.strftime("%Y-%m-%d"),
        "trough_date": trough.strftime("%Y-%m-%d"),
        "recovery_date": recovery.strftime("%Y-%m-%d") if pd.notna(recovery) else None,
        "max_drawdown": float(drawdown.loc[trough]),
        "peak_to_trough_trading_days": int(nav.loc[peak:trough].shape[0] - 1),
        "underwater_trading_days": int(nav.loc[peak:end].shape[0] - 1),
    }


def standalone_risk_metrics(returns: pd.Series, nav: pd.Series) -> dict[str, Any]:
    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=1) * math.sqrt(252)) if len(downside) > 1 else float("nan")
    sortino = float(returns.mean() * 252 / downside_vol) if downside_vol > 0 else float("nan")
    q05 = float(returns.quantile(0.05))
    cvar = float(returns[returns <= q05].mean())
    return {
        "annualized_downside_volatility": downside_vol,
        "sortino_zero_mar": sortino,
        "historical_var_95_daily_loss": -q05,
        "historical_cvar_95_daily_loss": -cvar,
        "best_day_return": float(returns.max()),
        "best_day": returns.idxmax().strftime("%Y-%m-%d"),
        "worst_day_return": float(returns.min()),
        "worst_day": returns.idxmin().strftime("%Y-%m-%d"),
        **max_drawdown_episode(nav),
    }


def benchmark_exposure(portfolio: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    frame = pd.concat([portfolio.rename("p"), benchmark.rename("b")], axis=1).dropna()
    p = frame["p"].to_numpy(float)
    b = frame["b"].to_numpy(float)
    beta = float(np.cov(p, b, ddof=0)[0, 1] / np.var(b))
    alpha = float((p.mean() - beta * b.mean()) * 252)
    active = p - b
    corr = float(np.corrcoef(p, b)[0, 1])
    positive = b > 0
    negative = b < 0
    upside = float(p[positive].mean() / b[positive].mean()) if positive.any() else float("nan")
    downside = float(p[negative].mean() / b[negative].mean()) if negative.any() else float("nan")
    return {
        "observations": int(len(frame)),
        "beta": beta,
        "annualized_jensen_alpha_zero_rf": alpha,
        "correlation": corr,
        "r_squared_univariate": corr * corr,
        "information_ratio": float(active.mean() / active.std(ddof=1) * math.sqrt(252)),
        "up_capture": upside,
        "down_capture": downside,
    }


def multi_index_exposure(portfolio: pd.Series, benchmarks: pd.DataFrame) -> dict[str, float]:
    frame = pd.concat([portfolio.rename("p"), benchmarks], axis=1).dropna()
    y = frame.pop("p").to_numpy(float)
    x = frame.to_numpy(float)
    design = np.column_stack([np.ones(len(x)), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    residual = y - fitted
    tss = float(np.sum((y - y.mean()) ** 2))
    result = {
        "observations": int(len(y)),
        "annualized_intercept_zero_rf": float(coef[0] * 252),
        "r_squared": float(1 - np.sum(residual**2) / tss),
        "annualized_residual_volatility": float(residual.std(ddof=1) * math.sqrt(252)),
    }
    for i, code in enumerate(frame.columns, start=1):
        result[f"beta_{code}"] = float(coef[i])
    return result


def build_analysis(
    summary: dict[str, Any], nav_long: pd.DataFrame, index_long: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nav = nav_long.pivot(index="trade_date", columns="model", values="nav").sort_index()
    returns = nav.pct_change().fillna(0.0)
    benchmark_close = index_long.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    benchmark_returns = benchmark_close.pct_change().reindex(nav.index)

    standalone: dict[str, Any] = {}
    exposures: list[dict[str, Any]] = []
    multi: dict[str, Any] = {}
    for model in MODEL_LABELS:
        standalone[model] = standalone_risk_metrics(returns[model], nav[model])
        for code in BENCHMARK_LABELS:
            exposures.append({
                "model": model,
                "benchmark": code,
                **benchmark_exposure(returns[model], benchmark_returns[code]),
            })
        multi[model] = multi_index_exposure(returns[model], benchmark_returns[list(BENCHMARK_LABELS)])

    monthly = (1 + returns).resample("ME").prod() - 1
    monthly.index.name = "month"
    yearly = (1 + returns).groupby(returns.index.year).prod() - 1
    yearly.index.name = "year"
    correlation = returns.corr()
    exposure_df = pd.DataFrame(exposures)

    predictive = {
        model: summary["predictive_metrics"][model]["full_test"] for model in MODEL_LABELS
    }
    portfolio = {row["model"]: row for row in summary["portfolio_results"]}
    analysis = {
        "source_status": summary["status"],
        "test_period": summary["test_period"],
        "interpretation_only": True,
        "no_post_test_tuning": True,
        "standalone_risk": standalone,
        "benchmark_exposure": exposures,
        "multi_index_exposure": multi,
        "return_correlation": correlation.to_dict(),
        "predictive_metrics": predictive,
        "execution_and_turnover": {
            model: {"impact": portfolio[model]["impact"], "diagnostics": portfolio[model]["diagnostics"]}
            for model in MODEL_LABELS
        },
        "method_notes": [
            "Alpha and beta use daily total returns and assume a zero risk-free rate.",
            "Benchmark regressions are descriptive exposures, not causal factor attribution.",
            "The three-index regression is affected by benchmark collinearity.",
            "2026 is partial through 2026-08-27.",
        ],
    }
    return analysis, nav, returns, benchmark_returns, exposure_df, monthly, yearly, correlation


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / name, bbox_inches="tight")
    plt.close(fig)


def plot_nav_and_drawdown(nav: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]})
    for model in MODEL_LABELS:
        label = MODEL_LABELS[model]
        axes[0].plot(nav.index, nav[model], label=label, color=COLORS[model], linewidth=2)
        drawdown = nav[model] / nav[model].cummax() - 1
        axes[1].plot(nav.index, drawdown * 100, label=label, color=COLORS[model], linewidth=1.7)
    axes[0].set_title("Permanent Test: cumulative net value")
    axes[0].set_ylabel("Net value")
    axes[0].legend(ncol=3, loc="upper left")
    axes[1].set_title("Drawdown path")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30)
    save_figure(fig, output, "01_nav_and_drawdown.png")


def plot_model_metrics(summary: dict[str, Any], output: Path) -> None:
    portfolio = {row["model"]: row["full_test"] for row in summary["portfolio_results"]}
    predictive = {m: summary["predictive_metrics"][m]["full_test"] for m in MODEL_LABELS}
    metrics = [
        ("Daily Rank IC", [predictive[m]["daily_rank_ic_mean"] for m in MODEL_LABELS], 1),
        ("Annualized return (%)", [portfolio[m]["annualized_return"] * 100 for m in MODEL_LABELS], 1),
        ("Sharpe ratio", [portfolio[m]["sharpe"] for m in MODEL_LABELS], 1),
        ("Maximum drawdown (%)", [portfolio[m]["max_drawdown"] * 100 for m in MODEL_LABELS], 1),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    labels = [MODEL_LABELS[m].replace(" (", "\n(") for m in MODEL_LABELS]
    colors = [COLORS[m] for m in MODEL_LABELS]
    for ax, (title, values, direction) in zip(axes.ravel(), metrics):
        bars = ax.bar(labels, values, color=colors, alpha=0.9)
        ax.set_title(title)
        ax.axhline(0, color="#64748b", linewidth=0.8)
        best = int(np.argmax(np.asarray(values) * direction))
        bars[best].set_hatch("//")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    save_figure(fig, output, "02_model_metric_comparison.png")


def plot_risk_return(summary: dict[str, Any], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for row in summary["portfolio_results"]:
        model = row["model"]
        x = row["full_test"]["annualized_volatility"] * 100
        y = row["full_test"]["annualized_return"] * 100
        sharpe = row["full_test"]["sharpe"]
        ax.scatter(x, y, s=130, color=COLORS[model], edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(f"{MODEL_LABELS[model]}\nSharpe {sharpe:.3f}", (x, y), xytext=(7, 7), textcoords="offset points", fontsize=9)
    ax.set_title("Risk-return comparison")
    ax.set_xlabel("Annualized volatility (%)")
    ax.set_ylabel("Annualized return (%)")
    save_figure(fig, output, "03_risk_return_scatter.png")


def plot_monthly_heatmap(monthly: pd.DataFrame, output: Path) -> None:
    values = monthly[list(MODEL_LABELS)].T * 100
    labels = [d.strftime("%Y-%m") for d in values.columns]
    limit = max(1.0, float(np.nanmax(np.abs(values.to_numpy()))))
    fig, ax = plt.subplots(figsize=(14, 3.8))
    image = ax.imshow(values.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
    ax.set_yticks(range(len(values.index)), [MODEL_LABELS[m] for m in values.index])
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_title("Monthly portfolio returns (%)")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values.iloc[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7.5, color="black")
    fig.colorbar(image, ax=ax, shrink=0.75, label="Monthly return (%)")
    save_figure(fig, output, "04_monthly_return_heatmap.png")


def plot_rolling_risk(returns: pd.DataFrame, benchmarks: pd.DataFrame, output: Path) -> None:
    window = 63
    rolling_vol = returns[list(MODEL_LABELS)].rolling(window).std() * math.sqrt(252) * 100
    mlp = returns["mlp_primary"]
    rolling_beta = pd.DataFrame(index=returns.index)
    for code in BENCHMARK_LABELS:
        rolling_beta[code] = mlp.rolling(window).cov(benchmarks[code]) / benchmarks[code].rolling(window).var()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for model in MODEL_LABELS:
        axes[0].plot(rolling_vol.index, rolling_vol[model], label=MODEL_LABELS[model], color=COLORS[model], linewidth=1.7)
    axes[0].set_title("63-day rolling annualized volatility")
    axes[0].set_ylabel("Volatility (%)")
    axes[0].legend(ncol=3, loc="upper left")
    for code in BENCHMARK_LABELS:
        axes[1].plot(rolling_beta.index, rolling_beta[code], label=f"MLP vs {BENCHMARK_LABELS[code]}", color=COLORS[code], linewidth=1.6)
    axes[1].axhline(1, color="#64748b", linewidth=0.8, linestyle="--")
    axes[1].set_title("MLP 63-day rolling benchmark beta")
    axes[1].set_ylabel("Beta")
    axes[1].set_xlabel("Date")
    axes[1].legend(ncol=3, loc="upper left")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30)
    save_figure(fig, output, "05_rolling_risk.png")


def plot_year_and_execution(summary: dict[str, Any], yearly: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(yearly.index))
    width = 0.24
    for offset, model in enumerate(MODEL_LABELS):
        axes[0].bar(x + (offset - 1) * width, yearly[model].to_numpy() * 100, width, label=MODEL_LABELS[model], color=COLORS[model])
    year_labels = ["2026 YTD\n(to Aug 27)" if int(year) == 2026 else str(year) for year in yearly.index]
    axes[0].set_xticks(x, year_labels)
    axes[0].axhline(0, color="#64748b", linewidth=0.8)
    axes[0].set_title("Calendar-year return")
    axes[0].set_ylabel("Return (%)")
    axes[0].legend(fontsize=8)

    rows = {row["model"]: row for row in summary["portfolio_results"]}
    models = list(MODEL_LABELS)
    mean_impact = [rows[m]["impact"]["mean_estimated_roundtrip_impact_bps"] for m in models]
    p95_impact = [rows[m]["impact"]["p95_estimated_roundtrip_impact_bps"] for m in models]
    xi = np.arange(len(models))
    axes[1].bar(xi - 0.18, mean_impact, 0.36, label="Mean impact", color="#2563eb")
    axes[1].bar(xi + 0.18, p95_impact, 0.36, label="95th percentile", color="#d97706")
    axes[1].axhline(20, color="#64748b", linewidth=1, linestyle="--", label="Fixed roundtrip cost")
    axes[1].set_xticks(xi, [MODEL_LABELS[m].replace(" (", "\n(") for m in models])
    axes[1].set_title("Estimated roundtrip market impact")
    axes[1].set_ylabel("Basis points")
    axes[1].legend(fontsize=8)
    save_figure(fig, output, "06_year_and_execution.png")


def write_outputs(
    output: Path,
    analysis: dict[str, Any],
    exposure: pd.DataFrame,
    monthly: pd.DataFrame,
    yearly: pd.DataFrame,
    correlation: pd.DataFrame,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "risk_attribution.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    exposure.to_csv(output / "benchmark_exposure.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_returns.csv", encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_returns.csv", encoding="utf-8-sig")
    correlation.to_csv(output / "model_return_correlation.csv", encoding="utf-8-sig")
    pd.DataFrame.from_dict(analysis["standalone_risk"], orient="index").rename_axis("model").to_csv(
        output / "standalone_risk_metrics.csv", encoding="utf-8-sig"
    )


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    configure_plotting()
    summary, nav_long, index_long = load_inputs()
    analysis, nav, returns, benchmarks, exposure, monthly, yearly, correlation = build_analysis(
        summary, nav_long, index_long
    )
    write_outputs(output, analysis, exposure, monthly, yearly, correlation)
    plot_nav_and_drawdown(nav, output)
    plot_model_metrics(summary, output)
    plot_risk_return(summary, output)
    plot_monthly_heatmap(monthly, output)
    plot_rolling_risk(returns, benchmarks, output)
    plot_year_and_execution(summary, yearly, output)
    print(json.dumps({
        "output_dir": str(output),
        "figures": 6,
        "tables": 5,
        "interpretation_only": True,
        "no_post_test_tuning": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
