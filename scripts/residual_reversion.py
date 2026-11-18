"""Run from the project root: python -m scripts.residual_reversion.

The existing Engine and LIST collection drive every experiment. This research
Module keeps scalar diagnostics and uses self-financing post-cost targets; the
project Simulator is unchanged. Daily close execution is idealized. Delay and
borrowing scenarios are explicit, not estimates of actual implementation costs.
"""

import argparse
import csv
import json
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import rankdata

from src.data import DailyReturns, load_daily_returns
from src.engine import Engine, ModelEntry
from src.models.factor import FactorModel
from src.models.pca import PCAModel
from src.models.residual_calibration import ResidualCalibrationModel
from src.models.residual_signal import ResidualSignalModel
from src.modules.module import Module
from src.strategies.residual_reversion import ResidualReversion


class Evaluation(Module):
    """Scalar accounting and diagnostics, with explicit execution delay."""

    def __init__(
        self,
        name: str,
        strategy: ResidualReversion,
        start: str,
        fee: float = 0.0005,
        delay: int = 0,
        borrow: float = 0.0,
        rebalance: int = 1,
    ) -> None:
        assert fee >= 0 and delay >= 0 and borrow >= 0 and rebalance >= 1
        self.name = name
        self.strategy = strategy
        self.start = np.datetime64(start)
        self.fee = fee
        self.delay = delay
        self.borrow = borrow
        self.rebalance = rebalance
        self.trading_days = 0
        self.positions: np.ndarray | None = None
        self.pending: deque[np.ndarray] = deque()
        self.rows: list[tuple[float, ...]] = []

    def update(
        self,
        timestamp: np.datetime64,
        returns: np.ndarray,
        model_outputs: dict[str, Any],
    ) -> None:
        if self.positions is None:
            self.positions = np.zeros(returns.shape[1])

        gross_return = float(self.positions @ returns[-1])
        borrow_cost = self.borrow / 252 * np.maximum(-self.positions, 0).sum()
        growth = 1 + gross_return - borrow_cost

        if growth <= 0:
            raise RuntimeError(
                "Portfolio insolvent; this run cannot be summarized with log returns."
            )

        drifted = self.positions * (1 + returns[-1]) / growth
        target = self.strategy.update(timestamp, model_outputs, drifted.copy())
        self.pending.append(target)
        target = (
            self.pending.popleft()
            if len(self.pending) > self.delay
            else np.zeros_like(target)
        )

        if timestamp < self.start:
            target = np.zeros_like(target)
        else:
            if self.trading_days % self.rebalance:
                target = drifted
            self.trading_days += 1

        # V_after / V_before = 1 - fee * |target * V_after/V_before - drifted|.
        def balance(ratio):
            return ratio + self.fee * np.abs(target * ratio - drifted).sum() - 1

        ratio = brentq(balance, 0, 1, xtol=1e-14) if self.fee else 1.0
        turnover = float(np.abs(target * ratio - drifted).sum())
        betas = model_outputs["factor"][-1].betas
        exposure = (
            float(np.max(np.abs(betas.T @ target))) if np.isfinite(betas).all() else 0.0
        )
        self.rows.append(
            (
                growth * ratio - 1,
                gross_return,
                turnover,
                np.abs(target).sum(),
                target.sum(),
                exposure,
                growth * (1 - ratio),
                borrow_cost,
                float(
                    self.trading_days > 0
                    and (self.trading_days - 1) % self.rebalance == 0
                ),
            )
        )
        self.positions = target


def metrics(values: np.ndarray, market: np.ndarray) -> dict[str, float]:
    deviation = values.std(ddof=1)
    wealth = np.cumprod(1 + values)
    peak = np.maximum.accumulate(np.r_[1, wealth])[1:]
    alpha, beta = np.linalg.lstsq(
        np.column_stack((np.ones(len(market)), market)), values, rcond=None
    )[0]
    active = values - beta * market
    return {
        "cumulative_return": float(wealth[-1] - 1),
        "pnl_per_million": float((wealth[-1] - 1) * 1_000_000),
        "cagr": float(wealth[-1] ** (252 / len(values)) - 1),
        "sharpe": float(values.mean() / deviation * np.sqrt(252))
        if deviation > 1e-12
        else 0.0,
        "volatility": float(deviation * np.sqrt(252)),
        "drawdown": float(np.min(wealth / peak - 1)),
        "alpha_annual": float(alpha * 252),
        "market_beta": float(beta),
        "information_ratio": float(active.mean() / active.std(ddof=1) * np.sqrt(252))
        if active.std() > 1e-12
        else 0.0,
    }


def configurations(full: bool) -> list[tuple[int, int, int]]:
    base = (5, 500, 100)
    groups = [base]

    if full:
        groups += [
            (3, 500, 100),
            (10, 500, 100),
            (5, 252, 100),
            (5, 756, 100),
            (5, 500, 60),
            (5, 500, 252),
        ]

    return groups


@dataclass(frozen=True)
class Configuration:
    factors: int = 5
    pca_window: int = 500
    regression_window: int = 100
    score_window: int = 60
    mode: str = "rank"
    fraction: float = 0.2
    entry: float = 1.5
    exit: float = 0.5
    horizon: int = 1
    rebalance: int = 1
    fee: float = 0.0005
    delay: int = 0
    borrow: float = 0.0
    use_ou: bool = False

    @property
    def key(self) -> str:
        return "_".join(str(value) for value in self.__dict__.values())

    @property
    def label(self) -> str:
        signal = "OU" if self.use_ou else self.mode.capitalize()
        return (
            f"{signal}: K={self.factors}, PCA={self.pca_window}d, "
            f"regression={self.regression_window}d, score={self.score_window}d, "
            f"rebalance={self.rebalance}d"
        )


def group_configurations(
    parameters: tuple[int, int, int], full: bool
) -> list[Configuration]:
    base = Configuration(
        factors=parameters[0], pca_window=parameters[1], regression_window=parameters[2]
    )
    configs = []
    for window in (20, 60, 120) if full else (60,):
        setting = replace(base, score_window=window)
        configs.extend(
            [
                setting,
                replace(setting, mode="threshold"),
                replace(setting, mode="threshold", entry=2.0),
            ]
        )
    configs.extend(
        [
            replace(base, mode="alpha"),
            replace(base, mode="threshold", use_ou=True),
            replace(base, mode="calibrated"),
        ]
    )
    if parameters == (5, 500, 100):
        configs.extend(replace(base, fee=fee) for fee in (0, 0.0002, 0.001))
        configs.extend([replace(base, delay=1), replace(base, borrow=0.03)])
        for interval in (5, 20):
            configs.extend(
                [
                    replace(base, rebalance=interval),
                    replace(base, mode="threshold", rebalance=interval),
                    replace(
                        base, mode="calibrated", horizon=interval, rebalance=interval
                    ),
                ]
            )
        configs.extend(replace(base, fraction=fraction) for fraction in (0.1, 0.3))
        configs.extend(
            replace(base, mode="threshold", entry=entry) for entry in (1.0, 2.5)
        )
    return configs


def evaluation_start(data: DailyReturns, full: bool) -> str:
    """First full calendar year after every candidate can finish its warm-up."""
    ready = []
    for group in configurations(full):
        for config in group_configurations(group, full):
            index = (
                max(config.pca_window, config.regression_window)
                - 1
                + config.score_window
            )
            if config.mode == "calibrated":
                index += config.horizon + 125  # 126 matured training dates
            ready.append(index)
    last_warmup = data.dates[max(ready)]
    return str(last_warmup.astype("datetime64[Y]") + np.timedelta64(1, "Y")) + "-01-01"


def run_group(
    data: DailyReturns,
    parameters: tuple[int, int, int],
    full: bool,
) -> tuple[dict[str, np.ndarray], list[Configuration], dict[str, np.ndarray]]:
    configs = group_configurations(parameters, full)
    start = evaluation_start(data, full)
    k, pca_window, regression_window = parameters
    models = [
        ModelEntry("pca", PCAModel(k, pca_window, 5)),
        ModelEntry("factor", FactorModel("pca", k, regression_window, 5)),
    ]
    for window, use_ou in sorted({(c.score_window, c.use_ou) for c in configs}):
        models.append(
            ModelEntry(
                f"signal_{window}_{use_ou}",
                ResidualSignalModel("factor", window, use_ou),
            )
        )
    for horizon in sorted({c.horizon for c in configs if c.mode == "calibrated"}):
        models.append(
            ModelEntry(
                f"forecast_{horizon}",
                ResidualCalibrationModel(
                    "signal_60_False",
                    "factor",
                    "pca",
                    horizon=horizon,
                ),
            )
        )
    modules = []
    for config in configs:
        strategy = ResidualReversion(
            "factor",
            f"signal_{config.score_window}_{config.use_ou}",
            mode=config.mode,
            forecast_name=f"forecast_{config.horizon}",
            fraction=config.fraction,
            entry=config.entry,
            exit=config.exit,
            trade_fee=config.fee,
        )
        modules.append(
            Evaluation(
                config.key,
                strategy,
                start,
                config.fee,
                config.delay,
                config.borrow,
                config.rebalance,
            )
        )
    diagnostics = CalibrationDiagnostics() if parameters == (5, 500, 100) else None
    observers = modules + ([diagnostics] if diagnostics is not None else [])
    Engine(data, models, observers).run()
    samples = diagnostics.arrays() if diagnostics is not None else {}
    return (
        {module.name: np.asarray(module.rows) for module in modules},
        configs,
        samples,
    )


def bootstrap_sharpe(
    values: np.ndarray, seed: int = 0, samples: int = 500
) -> list[float]:
    rng = np.random.default_rng(seed)
    estimates = []
    block = 21

    for _ in range(samples):
        starts = rng.integers(0, len(values), size=(len(values) + block - 1) // block)
        indices = ((starts[:, None] + np.arange(block)) % len(values)).ravel()[
            : len(values)
        ]
        draw = values[indices]
        estimates.append(draw.mean() / max(draw.std(ddof=1), 1e-12) * np.sqrt(252))

    return np.percentile(estimates, [2.5, 97.5]).tolist()


@dataclass
class DiagnosticOrigin:
    date: np.datetime64
    score: np.ndarray
    forecasts: dict[int, np.ndarray]
    betas: np.ndarray
    weights: np.ndarray
    growth: np.ndarray
    age: int = 0


class CalibrationDiagnostics(Module):
    """Record forward outcomes without feeding them back into a strategy.

    Quantiles are formed cross-sectionally at the prediction date. Forward
    labels use frozen-origin hedges and buy-and-hold stock growth. Observations
    crossing the development/holdout boundary are excluded when reporting.
    """

    def __init__(self) -> None:
        self.pending: deque[DiagnosticOrigin] = deque()
        self.bins: list[tuple[float, ...]] = []
        self.daily: list[tuple[float, ...]] = []

    def update(self, timestamp, returns, model_outputs) -> None:
        for origin in self.pending:
            origin.growth *= 1 + returns[-1]
            origin.age += 1
            if origin.age not in (1, 5, 20):
                continue

            raw = origin.growth - 1
            hedged = raw - (raw @ origin.weights) @ origin.betas.T
            prediction = origin.forecasts[origin.age]
            dates = (
                float(origin.date.astype("datetime64[D]").astype(int)),
                float(timestamp.astype("datetime64[D]").astype(int)),
                origin.age,
            )
            valid = np.isfinite(origin.score) & np.isfinite(hedged)
            signal_ic = np.nan
            forecast_ic = np.nan
            rmse = np.nan
            zero_rmse = np.nan

            if valid.sum() >= 20:
                signal_ic = correlation(
                    rankdata(-origin.score[valid]), rankdata(hedged[valid])
                )
            matched = valid & np.isfinite(prediction)
            if matched.sum() >= 20:
                forecast_ic = correlation(prediction[matched], hedged[matched])
                rmse = float(np.mean((prediction[matched] - hedged[matched]) ** 2))
                zero_rmse = float(np.mean(hedged[matched] ** 2))
            self.daily.append((*dates, signal_ic, forecast_ic, rmse, zero_rmse))

            for kind, signal in enumerate((origin.score, prediction)):
                indices = np.flatnonzero(np.isfinite(signal) & np.isfinite(hedged))
                if len(indices) < 20:
                    continue
                order = indices[np.argsort(signal[indices], kind="stable")]
                for quantile, stocks in enumerate(np.array_split(order, 10)):
                    self.bins.append(
                        (
                            *dates,
                            kind,
                            quantile + 1,
                            len(stocks),
                            float(signal[stocks].mean()),
                            float(hedged[stocks].mean()),
                            float(raw[stocks].mean()),
                        )
                    )

        while self.pending and self.pending[0].age >= 20:
            self.pending.popleft()

        pca = model_outputs["pca"][-1]
        factor = model_outputs["factor"][-1]
        score = model_outputs["signal_60_False"][-1].score
        if (
            pca is not None
            and np.isfinite(score).sum() >= 20
            and np.isfinite(factor.betas).all()
        ):
            self.pending.append(
                DiagnosticOrigin(
                    date=timestamp,
                    score=score.copy(),
                    forecasts={
                        h: model_outputs[f"forecast_{h}"][-1].expected_return.copy()
                        for h in (1, 5, 20)
                    },
                    betas=factor.betas.copy(),
                    weights=pca.components / pca.std_devs[:, None],
                    growth=np.ones_like(score),
                )
            )

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "quantiles": np.asarray(self.bins),
            "daily_calibration": np.asarray(self.daily),
        }


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def mean_interval(values: np.ndarray, lag: int = 21) -> tuple[float, float]:
    """Approximate 95% interval using daily means and Newey-West covariances."""
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    centered = values - values.mean()
    variance = np.mean(centered**2)
    for offset in range(1, min(lag + 1, len(values))):
        variance += (
            2
            * (1 - offset / (lag + 1))
            * np.dot(centered[offset:], centered[:-offset])
            / len(values)
        )
    return float(values.mean()), float(1.96 * np.sqrt(max(variance, 0) / len(values)))


def diagnostic_mask(values: np.ndarray, period: str, start: str) -> np.ndarray:
    start = np.datetime64(start).astype(int)
    boundary = np.datetime64("2019-01-01").astype(int)
    if period == "Overall":
        return values[:, 0] >= start
    if period == "Holdout":
        return values[:, 0] >= boundary
    return (values[:, 0] >= start) & (values[:, 1] < boundary)


def calibration_plots(
    samples: dict[str, np.ndarray], destination: Path, start: str, whole: bool = False
) -> tuple[list[str], list[dict[str, Any]]]:
    bins = samples["quantiles"]
    daily = samples["daily_calibration"]
    paths, records = [], []
    names = ["Overall"] if whole else ["Development", "Holdout"]

    for kind, name in [(0, "residual_signal"), (1, "predicted_return")]:
        for raw, outcome in [(False, "factor_hedged"), (True, "raw_stock")]:
            if kind == 1 and raw:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            for axis, horizon in zip(axes, (1, 5, 20)):
                for holdout, period in enumerate(names):
                    selected = bins[
                        (bins[:, 2] == horizon)
                        & (bins[:, 3] == kind)
                        & diagnostic_mask(bins, period, start)
                    ]
                    means, errors = [], []
                    for quantile in range(1, 11):
                        bucket = selected[selected[:, 4] == quantile]
                        mean, interval = mean_interval(bucket[:, 8 if raw else 7])
                        means.append(mean * 1e4)
                        errors.append(interval * 1e4)
                        records.append(
                            {
                                "period": period,
                                "horizon_days": horizon,
                                "sort": name,
                                "outcome": outcome,
                                "quantile": quantile,
                                "dates": len(bucket),
                                "mean_signal": float(bucket[:, 6].mean())
                                if len(bucket)
                                else np.nan,
                                "mean_forward_return_bps": mean * 1e4,
                                "interval_95_bps": interval * 1e4,
                            }
                        )
                    axis.errorbar(
                        range(1, 11),
                        means,
                        yerr=errors,
                        marker="o",
                        capsize=3,
                        label=period,
                    )
                axis.axhline(0, color="grey")
                axis.set(
                    title=f"{horizon}-day forward return",
                    xlabel="Cross-sectional decile (low to high)",
                    ylabel="Mean forward return (basis points)",
                )
            axes[0].legend(fontsize=8)
            path = f"calibration/{name}_deciles_vs_{outcome}_returns.png"
            path = path.replace(".png", "_whole_period.png") if whole else path
            save_figure(
                fig,
                destination,
                path,
                f"{name.replace('_', ' ').capitalize()} versus realized {outcome.replace('_', ' ')} returns",
            )
            paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, horizon in zip(axes, (1, 5, 20)):
        for holdout, period in enumerate(names):
            selected = bins[
                (bins[:, 2] == horizon)
                & (bins[:, 3] == 1)
                & diagnostic_mask(bins, period, start)
            ]
            predicted, realized = [], []
            for quantile in range(1, 11):
                bucket = selected[selected[:, 4] == quantile]
                predicted.append(bucket[:, 6].mean() * 1e4)
                realized.append(bucket[:, 7].mean() * 1e4)
            axis.plot(predicted, realized, marker="o", label=period)
        limits = axis.get_xlim()
        axis.plot(
            limits, limits, linestyle="--", color="grey", label="Perfect calibration"
        )
        axis.set(
            title=f"{horizon}-day matched horizon",
            xlabel="Mean predicted return (bp)",
            ylabel="Mean realized hedged return (bp)",
        )
    axes[0].legend(fontsize=8)
    path = "calibration/predicted_vs_realized_return_reliability.png"
    path = path.replace(".png", "_whole_period.png") if whole else path
    save_figure(fig, destination, path, "Forecast reliability by prediction decile")
    paths.append(path)

    if whole:
        return paths, records

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for horizon in (1, 5, 20):
        selected = daily[
            (daily[:, 2] == horizon) & (daily[:, 0] >= np.datetime64(start).astype(int))
        ]
        dates = selected[:, 0].astype(int).astype("datetime64[D]")
        for axis, column in zip(axes, (3, 4)):
            series = selected[:, column]
            rolling = np.array(
                [np.nanmean(series[i - 125 : i + 1]) for i in range(125, len(series))]
            )
            axis.plot(dates[125:], rolling, label=f"{horizon}-day horizon")
            axis.axhline(0, color="grey")
    axes[0].set_ylabel("Reversal signal Spearman IC")
    axes[1].set_ylabel("Predicted-return Pearson IC")
    axes[1].set_xlabel("Signal date")
    axes[0].legend()
    path = "calibration/rolling_information_coefficients.png"
    path = path.replace(".png", "_whole_period.png") if whole else path
    save_figure(
        fig, destination, path, "126-day mean cross-sectional predictive correlation"
    )
    paths.append(path)
    return paths, records


def periods(dates: np.ndarray, start: str) -> dict[str, np.ndarray]:
    return {
        "Overall": dates >= np.datetime64(start),
        "Development": (dates >= np.datetime64(start))
        & (dates < np.datetime64("2019-01-01")),
        "Holdout": dates >= np.datetime64("2019-01-01"),
    }


def save_figure(fig, destination: Path, relative: str, title: str) -> None:
    fig.suptitle(title, fontsize=14, fontweight="bold")
    for axis in fig.axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    path = destination / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def rolling_sharpe(values: np.ndarray, window: int = 252) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    deviation = windows.std(axis=1, ddof=1)
    return np.divide(
        windows.mean(axis=1) * np.sqrt(252),
        deviation,
        out=np.full(len(windows), np.nan),
        where=deviation > 1e-12,
    )


def summarize(
    data: DailyReturns,
    results: dict[str, np.ndarray],
    configs: list[Configuration],
    start: str,
) -> list[dict[str, Any]]:
    market = data.returns.mean(axis=1)
    records = []
    for config in configs:
        rows = results[config.key]
        for period, mask in periods(data.dates, start).items():
            record = {"configuration": config.key, "period": period, **config.__dict__}
            record.update(metrics(rows[mask, 0], market[mask]))
            rebalance_days = mask & (rows[:, 8] == 1)
            period_rows = rows[mask]
            starting_equity = np.r_[1.0, np.cumprod(1 + period_rows[:-1, 0])]
            traded_notional = np.sum(
                starting_equity
                * (1 + period_rows[:, 1] - period_rows[:, 7])
                * period_rows[:, 2]
            )
            record.update(
                traded_notional_per_million=float(traded_notional * 1_000_000),
                average_gross=float(rows[mask, 3].mean()),
                daily_turnover=float(rows[mask, 2].mean()),
                trading_cost_annual=float(rows[mask, 6].mean() * 252),
                max_factor_exposure=float(rows[mask, 5].max()),
                max_rebalance_factor_exposure=float(rows[rebalance_days, 5].max()),
            )
            records.append(record)
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def performance_plots(
    data, results, configs, selected, destination, start
) -> list[str]:
    base = Configuration()
    comparisons = [base, next(c for c in configs if c.key == selected)]
    comparisons += [
        c
        for c in configs
        if (c.factors, c.pca_window, c.regression_window, c.score_window)
        == (5, 500, 100, 60)
        and c.mode in ("threshold", "calibrated", "alpha")
        and c.entry == 1.5
        and c.rebalance == 1
    ]
    comparisons = list({c.key: c for c in comparisons}.values())
    mask = data.dates >= np.datetime64(start)
    dates = data.dates[mask]
    paths = []

    def label(config):
        if config == base:
            return "Rank baseline"
        if config.key == selected:
            return "Development-selected: " + config.label
        return "OU threshold" if config.use_ou else config.mode.capitalize()

    for suffix, logarithmic in [
        ("cumulative_returns", False),
        ("wealth_log_scale", True),
    ]:
        fig, axis = plt.subplots(figsize=(13, 6))
        for config in comparisons:
            wealth = np.cumprod(1 + results[config.key][mask, 0])
            axis.plot(
                dates,
                wealth if logarithmic else (wealth - 1) * 100,
                label=label(config),
            )
        if logarithmic:
            axis.set_yscale("log")
        axis.set(
            xlabel="Date",
            ylabel="Wealth from $1 (log scale)"
            if logarithmic
            else "Cumulative net return (%)",
        )
        axis.axvline(
            np.datetime64("2019-01-01"),
            color="grey",
            linestyle="--",
            label="Holdout begins",
        )
        axis.legend(fontsize=8)
        path = f"performance/{suffix}.png"
        save_figure(fig, destination, path, "Strategy comparison after trading costs")
        paths.append(path)

    fig, axis = plt.subplots(figsize=(13, 5))
    for config in comparisons:
        axis.plot(
            dates[251:],
            rolling_sharpe(results[config.key][mask, 0]),
            label=label(config),
        )
    axis.axhline(0, color="grey")
    axis.set(xlabel="Date", ylabel="Annualized Sharpe (252-day trailing window)")
    axis.legend(fontsize=8)
    path = "performance/rolling_252_day_sharpe.png"
    save_figure(fig, destination, path, "Rolling risk-adjusted performance after costs")
    paths.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for config in comparisons[:2]:
        rows = results[config.key][mask]
        wealth = np.cumprod(1 + rows[:, 0])
        axes[0].plot(
            dates,
            (wealth / np.maximum.accumulate(np.r_[1, wealth])[1:] - 1) * 100,
            label=label(config),
        )
        axes[1].plot(dates, rows[:, 3], alpha=0.7)
        axes[2].plot(dates, rows[:, 2] * 100, alpha=0.6)
    for axis, text in zip(
        axes, ["Drawdown (%)", "Gross exposure / equity", "One-way daily turnover (%)"]
    ):
        axis.set_ylabel(text)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Date")
    path = "risk/drawdowns_exposure_and_turnover.png"
    save_figure(
        fig, destination, path, "Baseline and development-selected portfolio risk"
    )
    paths.append(path)

    years = np.unique(dates.astype("datetime64[Y]"))
    fig, axis = plt.subplots(figsize=(13, 5))
    for index, config in enumerate(comparisons[:2]):
        annual = [
            np.prod(
                1 + results[config.key][data.dates.astype("datetime64[Y]") == year, 0]
            )
            - 1
            for year in years
        ]
        axis.bar(
            np.arange(len(years)) + index * 0.35,
            np.array(annual) * 100,
            width=0.35,
            label=label(config),
        )
    axis.set_xticks(np.arange(len(years)), years.astype(str), rotation=45)
    axis.set(xlabel="Calendar year", ylabel="Net return (%)")
    axis.legend(fontsize=8)
    path = "performance/annual_net_returns.png"
    save_figure(fig, destination, path, "Calendar-year returns after trading costs")
    paths.append(path)

    fig, axis = plt.subplots(figsize=(13, 5))
    for column, text in [(0, "Net of fees"), (1, "Before fees")]:
        axis.plot(
            dates,
            (np.cumprod(1 + results[base.key][mask, column]) - 1) * 100,
            label=text,
        )
    axis.set(xlabel="Date", ylabel="Cumulative return (%)")
    axis.legend()
    path = "performance/baseline_gross_vs_net_returns.png"
    save_figure(fig, destination, path, "Baseline: effect of trading costs")
    paths.append(path)
    fig, axis = plt.subplots(figsize=(12, 9))
    calendar_years = np.unique(dates.astype("datetime64[Y]"))
    monthly = np.full((len(calendar_years), 12), np.nan)
    for i, year in enumerate(calendar_years):
        for month in range(12):
            date = year.astype("datetime64[M]") + np.timedelta64(month, "M")
            selected_days = data.dates.astype("datetime64[M]") == date
            if selected_days.any():
                monthly[i, month] = (
                    np.prod(1 + results[base.key][selected_days, 0]) - 1
                ) * 100
    limit = max(np.nanmax(np.abs(monthly)), 1)
    axis.imshow(monthly, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    for i, j in np.ndindex(monthly.shape):
        if np.isfinite(monthly[i, j]):
            axis.text(
                j, i, f"{monthly[i, j]:.1f}", ha="center", va="center", fontsize=8
            )
    axis.set_xticks(
        range(12),
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )
    axis.set_yticks(range(len(calendar_years)), calendar_years.astype(str))
    axis.set(xlabel="Calendar month", ylabel="Calendar year")
    path = "performance/baseline_monthly_net_return_heatmap.png"
    save_figure(fig, destination, path, "Baseline monthly net returns (%)")
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for interval in (1, 5, 20):
        config = replace(base, rebalance=interval)
        rows = results[config.key][mask]
        axes[0].plot(
            dates,
            rows[:, 5],
            label=f"Rebalance every {interval} trading days",
            alpha=0.7,
        )
        axes[1].plot(dates, rows[:, 4] * 100, alpha=0.7)
    axes[0].set_ylabel("Maximum absolute estimated factor exposure")
    axes[1].set_ylabel("Net dollar exposure / equity (%)")
    axes[1].set_xlabel("Date")
    axes[0].legend()
    path = "risk/exposure_drift_between_rebalances.png"
    save_figure(
        fig,
        destination,
        path,
        "Holding-period risk: neutrality holds at rebalance, exposures drift afterward",
    )
    paths.append(path)

    return paths


def comparison_plots(
    records: list[dict[str, Any]],
    configs: list[Configuration],
    destination: Path,
    whole: bool = False,
) -> list[str]:
    paths = []
    lookup = {(r["configuration"], r["period"]): r for r in records}
    period_names = ["Overall"] if whole else ["Development", "Holdout"]
    baseline = Configuration()

    for field, values, title in [
        ("factors", [3, 5, 10], "Number of PCA factors"),
        ("pca_window", [252, 500, 756], "PCA estimation window (days)"),
        ("regression_window", [60, 100, 252], "Factor regression window (days)"),
    ]:
        fig, axes = plt.subplots(
            1, len(period_names), figsize=(6 * len(period_names), 5)
        )
        axes = np.atleast_1d(axes)
        for axis, period in zip(axes, period_names):
            matrix = np.full((len(values), 3), np.nan)
            for i, value in enumerate(values):
                for j, window in enumerate((20, 60, 120)):
                    config = replace(baseline, **{field: value, "score_window": window})
                    record = lookup.get((config.key, period))
                    if record:
                        matrix[i, j] = record["sharpe"]
            limit = max(1.0, max(abs(record["sharpe"]) for record in records))
            axis.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit)
            for i, j in np.ndindex(matrix.shape):
                axis.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}" if np.isfinite(matrix[i, j]) else "Not run",
                    ha="center",
                    va="center",
                )
            axis.set_xticks(range(3), [20, 60, 120])
            axis.set_yticks(range(len(values)), values)
            axis.set(xlabel="Residual score window (days)", ylabel=title, title=period)
        path = f"parameter_sweeps/sharpe_by_{field}_and_score_window.png"
        path = path.replace(".png", "_whole_period.png") if whole else path
        save_figure(
            fig,
            destination,
            path,
            "Net Sharpe sensitivity (other parameters held at baseline)",
        )
        paths.append(path)

    candidates = [
        c for c in configs if c.fee == 0.0005 and c.delay == 0 and c.borrow == 0
    ]
    fig, axis = plt.subplots(figsize=(8, 7))
    modes = ("rank", "threshold", "calibrated", "alpha")
    if whole:
        distributions = [
            [lookup[(c.key, "Overall")]["sharpe"] for c in candidates if c.mode == mode]
            for mode in modes
        ]
        axis.boxplot(distributions, tick_labels=[mode.capitalize() for mode in modes])
        axis.set(
            xlabel="Strategy family (distribution across configurations)",
            ylabel="Whole-period net Sharpe",
        )
        path = "parameter_sweeps/whole_period_sharpe_by_strategy_family.png"
        title = "Whole-period Sharpe across tested configurations"
    else:
        for mode in modes:
            group = [c for c in candidates if c.mode == mode]
            axis.scatter(
                [lookup[(c.key, "Development")]["sharpe"] for c in group],
                [lookup[(c.key, "Holdout")]["sharpe"] for c in group],
                label=mode.capitalize(),
                alpha=0.75,
            )
        axis.axvline(0, color="grey")
        axis.set(xlabel="Development net Sharpe", ylabel="Holdout net Sharpe")
        axis.legend()
        path = "parameter_sweeps/development_vs_holdout_sharpe.png"
        title = "Parameter stability across periods (all tested candidates)"
    axis.axhline(0, color="grey")
    save_figure(fig, destination, path, title)
    paths.append(path)

    subsets = {
        "rebalance_intervals": [
            c
            for c in configs
            if (c.factors, c.pca_window, c.regression_window, c.score_window)
            == (5, 500, 100, 60)
            and c.entry == 1.5
            and c.fraction == 0.2
            and not c.use_ou
            and c.mode != "alpha"
            and c.fee == 0.0005
            and c.delay == 0
            and c.borrow == 0
        ],
        "entry_thresholds": [
            c
            for c in configs
            if c.factors == 5
            and c.pca_window == 500
            and c.regression_window == 100
            and c.score_window == 60
            and c.mode == "threshold"
            and not c.use_ou
            and c.rebalance == 1
        ],
        "rank_fractions": [
            c
            for c in configs
            if c.factors == 5
            and c.pca_window == 500
            and c.regression_window == 100
            and c.score_window == 60
            and c.mode == "rank"
            and c.rebalance == 1
            and c.fee == 0.0005
            and c.delay == 0
            and c.borrow == 0
        ],
        "execution_costs_and_delay": [baseline]
        + [c for c in configs if c.fee != 0.0005 or c.delay or c.borrow],
    }
    for name, subset in subsets.items():
        fig, axis = plt.subplots(figsize=(13, 6))
        labels = []
        for c in subset:
            if name == "rebalance_intervals":
                labels.append(f"{c.mode.capitalize()} / {c.rebalance}d")
            elif name == "entry_thresholds":
                labels.append(f"Entry |z| > {c.entry}")
            elif name == "rank_fractions":
                labels.append(f"Long/short {c.fraction:.0%} each")
            else:
                labels.append(
                    f"{c.fee * 1e4:g} bp fee / {c.delay}d delay / {c.borrow:.0%} borrow"
                )
        for index, period in enumerate(period_names):
            axis.bar(
                np.arange(len(subset)) + index * 0.35,
                [lookup[(c.key, period)]["sharpe"] for c in subset],
                width=0.35,
                label=period,
            )
        axis.set_xticks(np.arange(len(subset)) + 0.175, labels, rotation=25, ha="right")
        axis.set(ylabel="Net annualized Sharpe", xlabel="Trading configuration")
        axis.legend()
        path = f"trading_rules/{name}.png"
        path = path.replace(".png", "_whole_period.png") if whole else path
        save_figure(fig, destination, path, name.replace("_", " ").capitalize())
        paths.append(path)
    return paths


def save_report(
    data: DailyReturns,
    results: dict[str, np.ndarray],
    configs: list[Configuration],
    samples: dict[str, np.ndarray],
    destination: Path,
    settings: dict[str, Any],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "data").mkdir(exist_ok=True)
    (destination / "tables").mkdir(exist_ok=True)
    start = settings["evaluation_start"]
    records = summarize(data, results, configs, start)
    eligible = [
        r
        for r in records
        if r["period"] == "Development"
        and r["fee"] == 0.0005
        and r["delay"] == 0
        and r["borrow"] == 0
    ]
    selected = max(eligible, key=lambda r: r["sharpe"])["configuration"]
    paths = performance_plots(data, results, configs, selected, destination, start)
    paths += comparison_plots(records, configs, destination)
    paths += comparison_plots(records, configs, destination, whole=True)
    calibration_paths, calibration_records = calibration_plots(
        samples, destination, start
    )
    paths += calibration_paths
    whole_paths, whole_records = calibration_plots(
        samples, destination, start, whole=True
    )
    paths += whole_paths
    calibration_records += whole_records
    write_csv(destination / "tables" / "strategy_statistics.csv", records)
    write_csv(
        destination / "tables" / "calibration_statistics.csv", calibration_records
    )
    interval = bootstrap_sharpe(
        results[selected][data.dates >= np.datetime64("2019-01-01"), 0]
    )
    settings.update(
        selected=selected,
        candidate_count=len(eligible),
        selected_holdout_sharpe_interval=interval,
    )
    settings["daily_columns"] = [
        "net_return",
        "gross_return",
        "turnover",
        "gross_exposure",
        "net_exposure",
        "factor_exposure",
        "fee_return",
        "borrow_return",
        "rebalance_day",
    ]
    settings["quantile_columns"] = [
        "signal_date_epoch_days",
        "maturity_date_epoch_days",
        "horizon",
        "sort_kind",
        "decile",
        "stocks",
        "mean_signal",
        "hedged_return",
        "raw_return",
    ]
    settings["calibration_daily_columns"] = [
        "signal_date_epoch_days",
        "maturity_date_epoch_days",
        "horizon",
        "reversal_rank_ic",
        "forecast_ic",
        "forecast_mse",
        "zero_forecast_mse",
    ]
    (destination / "data" / "run_configuration.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    lookup = {(r["configuration"], r["period"]): r for r in records}
    base = Configuration()
    chosen = next(c for c in configs if c.key == selected)
    lines = [
        "# Residual reversion - results",
        "",
        f"**{len(data.stock_ids):,} stocks - {len(configs)} configurations - {start} to {data.dates[-1]}**",
        "",
        "Net results include trading fees. Dollar figures illustrate a $1 million starting portfolio, rebased at the start of each listed period.",
        "",
        "## Baseline",
        "",
        "5 PCA factors; 500-day PCA window; 100-day factor regression; 60-day residual score; long/short 20% each; daily rebalancing; 5 bp one-way fees.",
        "",
        "| Period | Sharpe | Net return | P&L | Max drawdown | Daily turnover | Notional traded |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for period in ("Overall", "Development", "Holdout"):
        r = lookup[(base.key, period)]
        lines.append(
            f"| {period} | {r['sharpe']:.2f} | {r['cumulative_return']:.1%} | ${r['pnl_per_million']:,.0f} | "
            f"{r['drawdown']:.1%} | {r['daily_turnover']:.1%} | ${r['traded_notional_per_million'] / 1e6:,.1f}m |"
        )
    lines += [
        "",
        "Notional traded is buys plus sells, not share volume or market liquidity. It grows with simulated portfolio equity.",
        "",
        "## Main comparisons",
        "",
        "| Configuration | Overall Sharpe | Holdout Sharpe | Overall net return | Holdout CAGR |",
        "|---|---:|---:|---:|---:|",
    ]
    comparisons = [
        ("Baseline", base),
        ("Development-selected", chosen),
        ("OU threshold", replace(base, mode="threshold", use_ou=True)),
        ("Calibrated, 1 day", replace(base, mode="calibrated")),
        ("Alpha benchmark", replace(base, mode="alpha")),
        ("Extra execution day", replace(base, delay=1)),
        ("10 bp per side", replace(base, fee=0.001)),
    ]
    for label, config in comparisons:
        overall, holdout = (
            lookup[(config.key, "Overall")],
            lookup[(config.key, "Holdout")],
        )
        lines.append(
            f"| {label} | {overall['sharpe']:.2f} | {holdout['sharpe']:.2f} | "
            f"{overall['cumulative_return']:.1%} | {holdout['cagr']:.1%} |"
        )
    lines += [
        "",
        f"Development-selected configuration: **{chosen.label}**. Selection used development Sharpe only.",
        "",
        "## Rebalance / forecast horizons",
        "",
        "| Rule | Interval | Overall Sharpe | Holdout Sharpe | Overall average gross |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ("rank", "threshold", "calibrated"):
        for horizon in (1, 5, 20):
            config = replace(
                base,
                mode=mode,
                rebalance=horizon,
                horizon=horizon if mode == "calibrated" else 1,
            )
            overall, holdout = (
                lookup[(config.key, "Overall")],
                lookup[(config.key, "Holdout")],
            )
            lines.append(
                f"| {mode.capitalize()} | {horizon} days | {overall['sharpe']:.2f} | "
                f"{holdout['sharpe']:.2f} | {overall['average_gross']:.2f} |"
            )
    daily = samples["daily_calibration"]
    lines += [
        "",
        "## Calibration check - whole evaluation period",
        "",
        "| Forward horizon | Reversal rank IC | Forecast IC | Forecast MSE / zero-forecast MSE |",
        "|---|---:|---:|---:|",
    ]
    for horizon in (1, 5, 20):
        group = daily[
            (daily[:, 2] == horizon) & diagnostic_mask(daily, "Overall", start)
        ]
        ratio = np.nanmean(group[:, 5]) / np.nanmean(group[:, 6])
        lines.append(
            f"| {horizon} days | {np.nanmean(group[:, 3]):.3f} | {np.nanmean(group[:, 4]):.3f} | {ratio:.6f} |"
        )
    lines += [
        "",
        "Positive IC indicates predictive alignment; an MSE ratio below 1 improves on a zero-return forecast. These are gross forecast diagnostics, not net trading profits.",
        "",
        "## Interpretation",
        "",
        f"- **Dates:** development is {start} to 2018-12-31; holdout is 2019-01-01 to 2023-12-29. The start is the first full year after the longest required warm-up, replacing the arbitrary 2005 cutoff.",
        "- **Regression:** ordinary rolling OLS (`numpy.linalg.lstsq`) of each stock on an intercept and PCA factor returns. It is fully refitted every five days, or when PCA changes; coefficients are reused between refits. It is not recursive regression. The separate calibration spline efficiently maintains rolling X-transpose-X and X-transpose-y statistics and solves ridge every 21 matured training dates.",
        "- **Timing:** signals use today's information for subsequent returns. Forecast labels mature after 1/5/20 days and use frozen-origin buy-and-hold hedges. Calibrated trading uses the matching rebalance interval.",
        "- **Neutrality:** beta and dollar neutrality apply at rebalance. Exposures drift between scheduled trades. Gross and position limits also apply at rebalance; sparse signals may leave cash idle.",
        "- **Execution:** signal-close fills are idealized. Compare the delay and cost rows before treating performance as executable. Cash earns zero interest; no impact or stock-specific borrow constraints are modeled.",
        "- **Research limits:** the historical holdout has already been inspected. No nested selection or deflated-Sharpe correction is claimed. Universe survivorship is unverified; the market benchmark is an equal-weight proxy.",
        "",
        "## Detailed outputs",
        "",
        "- [All configuration statistics](tables/strategy_statistics.csv)",
        "- [Signal and prediction quantile statistics](tables/calibration_statistics.csv)",
        "- [Performance plots](performance/) - [Parameter heatmaps](parameter_sweeps/) - [Trading-rule comparisons](trading_rules/)",
        "- [Calibration plots](calibration/) - [Risk plots](risk/)",
        "- [Run configuration and data definitions](data/run_configuration.json)",
        "",
        "Calibration deciles are formed at the signal date; crossing-period labels are excluded from split-period views. Error bars use daily group means and approximate 95% Newey-West intervals with 21 lags. Whole-period views are descriptive.",
    ]
    (destination / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Saved {len(paths)} plots and {len(configs)} configurations to {destination}",
        flush=True,
    )


def load_data(stocks: int) -> DailyReturns:
    data = load_daily_returns()
    if stocks:
        if not 20 <= stocks <= len(data.stock_ids):
            raise ValueError(
                "--stocks must be between 20 and the available count, or 0 for all."
            )
        indices = np.sort(
            np.random.default_rng(42).choice(len(data.stock_ids), stocks, replace=False)
        )
        data = DailyReturns(
            dates=data.dates,
            stock_ids=data.stock_ids[indices],
            returns=data.returns[:, indices],
        )
    return data


def worker(stocks: int, parameters: tuple[int, int, int], full: bool):
    started = perf_counter()
    result = run_group(load_data(stocks), parameters, full)
    return parameters, perf_counter() - started, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Baseline factor group only; retain trading-rule comparisons.",
    )
    parser.add_argument(
        "--stocks",
        type=int,
        default=0,
        help="Deterministic subset size; 0 uses all stocks.",
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Processes with one BLAS thread each."
    )
    parser.add_argument("--output", type=Path, default=Path("plots/residual_reversion"))
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate from saved results, without backtesting.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data").mkdir(exist_ok=True)
    if args.plots_only:
        settings = json.loads(
            (args.output / "data" / "run_configuration.json").read_text()
        )
        data = load_data(settings["stocks"])
        configs = [Configuration(**c) for c in settings["configurations"]]
        with np.load(args.output / "data" / "daily_results.npz") as archive:
            results = {key: archive[key] for key in archive.files}
        with np.load(args.output / "data" / "calibration_observations.npz") as archive:
            samples = {key: archive[key] for key in archive.files}
    else:
        results, configs, samples = {}, [], {}
        groups = configurations(not args.quick)
        settings = {
            "stocks": args.stocks,
            "quick": args.quick,
            "workers": min(args.workers, len(groups)),
        }
        with ProcessPoolExecutor(max_workers=settings["workers"]) as pool:
            jobs = [
                pool.submit(worker, args.stocks, group, not args.quick)
                for group in groups
            ]
            for job in as_completed(jobs):
                parameters, elapsed, (group_results, group_configs, group_samples) = (
                    job.result()
                )
                results.update(group_results)
                configs.extend(group_configs)
                samples.update(group_samples)
                print(
                    f"Completed K/PCA/regression={parameters} in {elapsed:.1f}s ({len(group_configs)} strategies)",
                    flush=True,
                )
        configs.sort(key=lambda c: c.key)
        data = load_data(args.stocks)
        settings["configurations"] = [c.__dict__ for c in configs]
        settings["stock_ids"] = data.stock_ids.tolist()
        settings["evaluation_start"] = evaluation_start(data, not args.quick)
        settings["dates"] = [str(data.dates[0]), str(data.dates[-1])]
        np.savez_compressed(args.output / "data" / "daily_results.npz", **results)
        np.savez_compressed(
            args.output / "data" / "calibration_observations.npz", **samples
        )
    save_report(data, results, configs, samples, args.output, settings)


if __name__ == "__main__":
    main()
