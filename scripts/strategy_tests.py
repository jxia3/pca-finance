import multiprocessing
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src import data
from src.data import DailyReturns
from src.engine import Collection, Engine, ModelEntry
from src.models.factor import FactorModel, ReturnAttribution
from src.models.pairs import PairSignals, PairsModel
from src.models.pca import CorrelationPCA, PCAModel, correlation_pca
from src.modules.simulator import PortfolioHistory, Simulator
from src.strategies.alpha_quintile import AlphaQuintile
from src.strategies.long_short_snapback import LongShortSnapback
from src.strategies.pairs_trading import PairsTrading
from src.strategies.residual_snapback import ResidualSnapback
from src.strategies.strategy import Strategy
from src.strategies.treynor_black import TreynorBlack

TRADE_FEE = 0.0005
TRADING_DAYS = 252
PCA_WINDOW = 500
SNAPBACK_WINDOW = 100
HOLDOUT_START = 2018
K_VALUES = (1, 2, 3, 5, 8, 10, 15, 20)
CACHE_FILE = "results/strategy_tests.pkl"
REPORT_FILE = "results/strategy_tests.md"
# Residual snapback reads up to volatility + signal days of factor outputs (125 for the longest variant),
# and the lagged factor model reads the previous PCA
HISTORY_DAYS = 200

class LatestEngine(Engine):
    # Keep only recent outputs of each model so many models fit in memory
    def _collect_output(self, collections: dict[str, Any], entry: ModelEntry, index: int, output: Any) -> None:
        assert entry.collection == Collection.LIST
        recent = collections.setdefault(entry.name, [])
        recent.append(output)
        del recent[:-HISTORY_DAYS]

class StaticModel:
    value: Any
    empty: Any
    start: int

    # Returns a fixed full sample estimate, used for the in-sample (look-ahead) comparison
    def __init__(self, value: Any, empty: Any, start: int) -> None:
        self.value = value
        self.empty = empty
        self.start = start

    def update(self, timestamp: np.datetime64, returns: np.ndarray, model_outputs: dict[str, Any]) -> Any:
        return self.value if len(returns) >= self.start else self.empty

class AlphaRecorder(Strategy):
    inner: TreynorBlack
    alphas: np.ndarray
    residual_variance: np.ndarray
    day: int

    def __init__(self, inner: TreynorBlack, num_days: int, num_stocks: int) -> None:
        self.inner = inner
        self.alphas = np.full((num_days, num_stocks), np.nan)
        self.residual_variance = np.full((num_days, num_stocks), np.nan)
        self.day = 0

    def update(self, timestamp: np.datetime64, model_outputs: dict[str, Any], positions: np.ndarray) -> np.ndarray:
        attribution: ReturnAttribution = model_outputs[self.inner.factor_name][-1]
        self.alphas[self.day] = attribution.factor_intercept
        self.residual_variance[self.day] = attribution.residual_variance
        self.day += 1
        return self.inner.update(timestamp, model_outputs, positions)

class PairRecorder(Strategy):
    inner: PairsTrading
    pairs: np.ndarray
    z_scores: np.ndarray
    day: int

    def __init__(self, inner: PairsTrading, num_days: int, num_pairs: int) -> None:
        self.inner = inner
        self.pairs = np.full((num_days, num_pairs, 2), -1, dtype=np.int64)
        self.z_scores = np.full((num_days, num_pairs), np.nan)
        self.day = 0

    def update(self, timestamp: np.datetime64, model_outputs: dict[str, Any], positions: np.ndarray) -> np.ndarray:
        signals: PairSignals | None = model_outputs[self.inner.pairs_name][-1]
        if signals is not None:
            self.pairs[self.day] = signals.pairs
            self.z_scores[self.day] = signals.z_scores
        self.day += 1
        return self.inner.update(timestamp, model_outputs, positions)

class ResidualRecorder(Strategy):
    inner: ResidualSnapback
    residuals: np.ndarray
    day: int

    def __init__(self, inner: ResidualSnapback, num_days: int, num_stocks: int) -> None:
        self.inner = inner
        self.residuals = np.full((num_days, num_stocks), np.nan)
        self.day = 0

    def update(self, timestamp: np.datetime64, model_outputs: dict[str, Any], positions: np.ndarray) -> np.ndarray:
        self.residuals[self.day] = model_outputs[self.inner.factor_name][-1].residuals
        self.day += 1
        return self.inner.update(timestamp, model_outputs, positions)

class EqualWeightBenchmark(Strategy):
    rebalance_dates: frozenset

    def __init__(self, rebalance_dates: np.ndarray) -> None:
        self.rebalance_dates = frozenset(rebalance_dates)

    def update(self, timestamp: np.datetime64, model_outputs: dict[str, Any], positions: np.ndarray) -> np.ndarray:
        if timestamp in self.rebalance_dates:
            return np.full_like(positions, 1.0 / positions.size)
        return positions

class InSampleFactor:
    weights: np.ndarray
    coefficients: np.ndarray
    start: int

    # Full sample PCA and regression, residuals still computed day by day for the in-sample (look-ahead) comparison
    def __init__(self, returns: np.ndarray, num_components: int, start: int) -> None:
        pca = correlation_pca(returns, num_components)
        self.weights = pca.components / pca.std_devs[:, np.newaxis]
        factors = returns @ self.weights
        features = np.hstack((np.ones((len(factors), 1)), factors))
        self.coefficients = np.linalg.lstsq(features, returns, rcond=None)[0]
        self.start = start

    def update(self, timestamp: np.datetime64, returns: np.ndarray, model_outputs: dict[str, Any]) -> ReturnAttribution:
        num_components, num_stocks = self.weights.shape[1], returns.shape[1]
        if len(returns) < self.start:
            return empty_attribution(num_components, num_stocks)
        factor_returns = returns[-1] @ self.weights
        projection = factor_returns @ self.coefficients[1:]
        return ReturnAttribution(
            factor_returns=factor_returns,
            factor_intercept=self.coefficients[0],
            factor_projection=projection,
            residuals=returns[-1] - self.coefficients[0] - projection,
            betas=self.coefficients[1:],
            residual_variance=np.full(num_stocks, np.nan),
        )

def weekly_rebalance_dates(dates: np.ndarray) -> np.ndarray:
    # Last trading date of each Monday-to-Sunday week, as in the residual snapback notebook
    weeks = (dates.astype("datetime64[D]") - np.datetime64("1970-01-05")) // np.timedelta64(7, "D")
    return dates[np.r_[weeks[:-1] != weeks[1:], True]]

def snapback(factor_name: str, daily_returns: DailyReturns, hedged: bool = False, *, daily: bool = False,
             original: float | None = None, dynamic: bool = False, projection: bool = False,
             **settings: Any) -> ResidualSnapback:
    # original = fraction kept in the unhedged portfolio for the hybrid variants, settings override class defaults
    # The hedged portfolio is the long-short book, projection = the earlier version that hedges the long picks only
    dates = daily_returns.dates if daily else weekly_rebalance_dates(daily_returns.dates)
    strategy = (ResidualSnapback if projection else LongShortSnapback)(
        factor_name, dates, hedge_out_components=hedged, hybrid=original is not None,
        hybrid_parameter=0.5 if original is None else original, dynamic_vol=dynamic, daily_returns=daily_returns,
    )
    for name, value in settings.items():
        assert hasattr(ResidualSnapback, name)
        setattr(strategy, name, value)
    return strategy

def snapback_variants(factor_name: str, daily_returns: DailyReturns, key: str) -> dict[str, Strategy]:
    hedged = key == "snaph"
    variants: dict[str, Strategy] = {}
    for days in (1, 5):
        variants[f"{key}|signal days|{days}"] = snapback(factor_name, daily_returns, hedged, signal_days=days)
    for days in (20, 120):
        variants[f"{key}|volatility window|{days}"] = snapback(factor_name, daily_returns, hedged, volatility_days=days)
    for threshold in (-1.5, -2.5, -3.0):
        variants[f"{key}|score threshold|{threshold:g}"] = snapback(factor_name, daily_returns, hedged,
                                                                    score_threshold=threshold)
    for count in (10, 50):
        variants[f"{key}|max positions|{count}"] = snapback(factor_name, daily_returns, hedged, max_positions=count)
    variants[f"{key}|rebalance|daily"] = snapback(factor_name, daily_returns, hedged, daily=True)
    return variants

@dataclass(kw_only=True)
class Series:
    growth: np.ndarray
    turnover: np.ndarray
    start: int

def to_series(history: PortfolioHistory) -> Series:
    # growth[j] and turnover[j] belong to day j, fees can be applied afterwards since trades do not depend on them
    # A strategy that never trades (e.g. hedged snapback with K >= its 20 positions) starts after the last day
    active = np.flatnonzero(np.any(history.positions != 0.0, axis=1))
    growth = history.total_value[1:] / history.total_value[:-1]
    return Series(
        growth=growth,
        turnover=np.sum(np.abs(history.trades[1:]), axis=1),
        start=int(active[0]) if len(active) else len(growth),
    )

def in_sample_estimates(returns: np.ndarray, num_components: int) -> tuple[CorrelationPCA, ReturnAttribution]:
    pca = correlation_pca(returns, num_components)
    factors = returns @ (pca.components / pca.std_devs[:, np.newaxis])
    features = np.hstack((np.ones((len(factors), 1)), factors))
    coefficients = np.linalg.lstsq(features, returns, rcond=None)[0]
    residuals = returns - features @ coefficients
    num_stocks = returns.shape[1]
    attribution = ReturnAttribution(
        factor_returns=np.zeros(num_components),
        factor_intercept=coefficients[0],
        factor_projection=np.zeros(num_stocks),
        residuals=np.zeros(num_stocks),
        betas=coefficients[1:],
        residual_variance=np.sum(residuals ** 2, axis=0) / (len(returns) - features.shape[1]),
    )
    return pca, attribution

def empty_attribution(num_components: int, num_stocks: int) -> ReturnAttribution:
    return ReturnAttribution(
        factor_returns=np.full(num_components, np.nan),
        factor_intercept=np.full(num_stocks, np.nan),
        factor_projection=np.full(num_stocks, np.nan),
        residuals=np.full(num_stocks, np.nan),
        betas=np.full((num_components, num_stocks), np.nan),
        residual_variance=np.full(num_stocks, np.nan),
    )

def base_job(daily_returns: DailyReturns) -> tuple[list[ModelEntry], dict[str, Strategy]]:
    num_days, num_stocks = daily_returns.returns.shape
    models = [
        ModelEntry("pca", PCAModel(5, PCA_WINDOW, 5)),
        ModelEntry("factor", FactorModel("pca", 5, 250, 1)),
        ModelEntry("pairs", PairsModel("pca")),
    ]
    strategies: dict[str, Strategy] = {
        "tb|base|": AlphaRecorder(TreynorBlack("factor", contrarian=True), num_days, num_stocks),
        "pairs|base|": PairRecorder(PairsTrading("pairs"), num_days, 50),
    }
    for limit in (2.0, 5.0):
        strategies[f"tb|score cap|{limit:g}"] = TreynorBlack("factor", contrarian=True, score_limit=limit)
    for entry in (1.5, 2.5, 3.0):
        strategies[f"pairs|entry z|{entry:g}"] = PairsTrading("pairs", entry_threshold=entry)
    for exit in (0.0, 1.0):
        strategies[f"pairs|exit z|{exit:g}"] = PairsTrading("pairs", exit_threshold=exit)
    return models, strategies

def alpha_window_job(daily_returns: DailyReturns) -> tuple[list[ModelEntry], dict[str, Strategy]]:
    models = [ModelEntry("pca", PCAModel(5, PCA_WINDOW, 5))]
    strategies: dict[str, Strategy] = {}
    for window in (60, 125, 500):
        models.append(ModelEntry(f"factor{window}", FactorModel("pca", 5, window, 1)))
        strategies[f"tb|alpha window|{window}"] = TreynorBlack(f"factor{window}", contrarian=True)
    return models, strategies

def pca_window_job(daily_returns: DailyReturns) -> tuple[list[ModelEntry], dict[str, Strategy]]:
    models = []
    strategies: dict[str, Strategy] = {}
    for window in (250, 1000):
        models += [
            ModelEntry(f"pca{window}", PCAModel(5, window, 5)),
            ModelEntry(f"factor{window}", FactorModel(f"pca{window}", 5, 250, 1)),
            ModelEntry(f"pairs{window}", PairsModel(f"pca{window}")),
        ]
        strategies[f"tb|PCA window|{window}"] = TreynorBlack(f"factor{window}", contrarian=True)
        strategies[f"pairs|PCA window|{window}"] = PairsTrading(f"pairs{window}")
    return models, strategies

def pairs_model_job(daily_returns: DailyReturns) -> tuple[list[ModelEntry], dict[str, Strategy]]:
    models = [ModelEntry("pca", PCAModel(5, PCA_WINDOW, 5))]
    variants = {
        "spread window": [(value, {"spread_window": value}) for value in (20, 40, 120)],
        "num pairs": [(value, {"num_pairs": value}) for value in (20, 100)],
        "reform days": [(value, {"formation_interval": value}) for value in (5, 63)],
        "selection": [("correlation", {"selection": "correlation"})],
    }
    strategies: dict[str, Strategy] = {}
    for param, settings in variants.items():
        for value, kwargs in settings:
            name = f"pairs_{param}_{value}".replace(" ", "_")
            models.append(ModelEntry(name, PairsModel("pca", **kwargs)))
            strategies[f"pairs|{param}|{value}"] = PairsTrading(name)
    return models, strategies

def k_job(num_components: int) -> Callable[[DailyReturns], tuple[list[ModelEntry], dict[str, Strategy]]]:
    def build(daily_returns: DailyReturns) -> tuple[list[ModelEntry], dict[str, Strategy]]:
        num_stocks = daily_returns.returns.shape[1]
        pca, attribution = in_sample_estimates(daily_returns.returns, num_components)
        models = [
            ModelEntry("pca", PCAModel(num_components, PCA_WINDOW, 5)),
            ModelEntry("factor", FactorModel("pca", num_components, 250, 1)),
            ModelEntry("pairs", PairsModel("pca")),
            ModelEntry("is_pca", StaticModel(pca, None, PCA_WINDOW)),
            ModelEntry("is_factor", StaticModel(attribution, empty_attribution(num_components, num_stocks), PCA_WINDOW)),
            ModelEntry("is_pairs", PairsModel("is_pca")),
        ]
        strategies: dict[str, Strategy] = {
            f"k{num_components}|tb|oos": TreynorBlack("factor", contrarian=True),
            f"k{num_components}|pairs|oos": PairsTrading("pairs"),
            f"k{num_components}|tb|is": TreynorBlack("is_factor", contrarian=True),
            f"k{num_components}|tb follow|is": TreynorBlack("is_factor", contrarian=False),
            f"k{num_components}|pairs|is": PairsTrading("is_pairs"),
        }
        return models, strategies
    return build

DEVELOPMENT = {
    "Long top 20% alpha, short bottom 20%": "dev|quintile|",
    "Contrarian 20% sort": "dev|contrarian quintile|",
    "Contrarian 20% sort, factor-hedged": "dev|contrarian quintile hedged|",
    "Contrarian Treynor-Black, not hedged": "dev|tb unhedged|",
    "Contrarian Treynor-Black, factor-hedged": "tb|base|",
}

def development_job(daily_returns: DailyReturns) -> tuple[list[ModelEntry], dict[str, Strategy]]:
    # The steps that led to contrarian Treynor-Black (Table 1), the last one is the base run
    models = [ModelEntry("pca", PCAModel(5, PCA_WINDOW, 5)), ModelEntry("factor", FactorModel("pca", 5, 250, 1))]
    strategies: dict[str, Strategy] = {
        "dev|quintile|": AlphaQuintile("factor"),
        "dev|contrarian quintile|": AlphaQuintile("factor", contrarian=True),
        "dev|contrarian quintile hedged|": AlphaQuintile("factor", factor_neutral=True, contrarian=True),
        "dev|tb unhedged|": TreynorBlack("factor", factor_neutral=False, contrarian=True),
    }
    return models, strategies

def snapback_models(pca_window: int = PCA_WINDOW, regression_window: int = SNAPBACK_WINDOW,
                    num_components: int = 5, suffix: str = "") -> list[ModelEntry]:
    # PCA and regression both fit through the prior day, today's return only enters today's residual
    return [
        ModelEntry(f"pca{suffix}", PCAModel(num_components, pca_window, 5)),
        ModelEntry(f"lagged{suffix}", FactorModel(f"pca{suffix}", num_components, regression_window, 1, fit_lag=1)),
    ]

def snapback_job(daily_returns: DailyReturns) -> tuple[list[ModelEntry], dict[str, Strategy]]:
    num_days, num_stocks = daily_returns.returns.shape
    strategies: dict[str, Strategy] = {
        "snap|base|": ResidualRecorder(snapback("lagged", daily_returns), num_days, num_stocks),
        "bench|base|": EqualWeightBenchmark(weekly_rebalance_dates(daily_returns.dates)),
        **snapback_variants("lagged", daily_returns, "snap"),
    }
    for original in (0.7, 0.5, 0.3):
        strategies[f"snap|hedge mix|{original:.0%} unhedged"] = snapback("lagged", daily_returns, original=original)
    strategies["snap|hedge mix|dynamic"] = snapback("lagged", daily_returns, dynamic=True)
    # Earlier versions, for comparison: projection hedge of the long picks, raw (unsmoothed) volatility percentile
    strategies["cmp|projection hedge|"] = snapback("lagged", daily_returns, hedged=True, projection=True)
    strategies["cmp|dynamic unsmoothed|"] = snapback("lagged", daily_returns, dynamic=True, projection=True)
    return snapback_models(), strategies

def snapback_hedged_job(daily_returns: DailyReturns) -> tuple[list[ModelEntry], dict[str, Strategy]]:
    strategies: dict[str, Strategy] = {
        "snaph|base|": snapback("lagged", daily_returns, hedged=True),
        **snapback_variants("lagged", daily_returns, "snaph"),
    }
    return snapback_models(), strategies

def snapback_window_job(daily_returns: DailyReturns) -> tuple[list[ModelEntry], dict[str, Strategy]]:
    models = snapback_models()[:1]
    strategies: dict[str, Strategy] = {}
    for window in (60, 250):
        models.append(ModelEntry(f"regression{window}", FactorModel("pca", 5, window, 1, fit_lag=1)))
        for key in ("snap", "snaph"):
            strategies[f"{key}|regression window|{window}"] = snapback(f"regression{window}", daily_returns,
                                                                       key == "snaph")
    for window in (250, 1000):
        models += snapback_models(pca_window=window, suffix=str(window))
        for key in ("snap", "snaph"):
            strategies[f"{key}|PCA window|{window}"] = snapback(f"lagged{window}", daily_returns, key == "snaph")
    return models, strategies

def snapback_k_job(num_components: int) -> Callable[[DailyReturns], tuple[list[ModelEntry], dict[str, Strategy]]]:
    def build(daily_returns: DailyReturns) -> tuple[list[ModelEntry], dict[str, Strategy]]:
        models = snapback_models(num_components=num_components) + [
            ModelEntry("is_factor", InSampleFactor(daily_returns.returns, num_components, PCA_WINDOW)),
        ]
        strategies: dict[str, Strategy] = {}
        for key in ("snap", "snaph"):
            strategies[f"k{num_components}|{key}|oos"] = snapback("lagged", daily_returns, key == "snaph")
            strategies[f"k{num_components}|{key}|is"] = snapback("is_factor", daily_returns, key == "snaph")
        return models, strategies
    return build

JOBS: dict[str, Callable[[DailyReturns], tuple[list[ModelEntry], dict[str, Strategy]]]] = {
    "base": base_job,
    "alpha_window": alpha_window_job,
    "pca_window": pca_window_job,
    "pairs_model": pairs_model_job,
    **{f"k{k}": k_job(k) for k in K_VALUES},
    "development": development_job,
    "snapback": snapback_job,
    "snapback_hedged": snapback_hedged_job,
    "snapback_window": snapback_window_job,
    **{f"snapk{k}": snapback_k_job(k) for k in K_VALUES},
}
# Slowest jobs first so the pool stays busy
JOB_ORDER = ["pairs_model", "k20", "k15", "base", "pca_window", "k10", "k8", "alpha_window", "k5", "k3", "k2", "k1",
             "development", "snapback", "snapback_hedged", "snapback_window", *(f"snapk{k}" for k in reversed(K_VALUES))]

def run_job(job: str) -> dict[str, Any]:
    daily_returns = data.load_daily_returns()
    models, strategies = JOBS[job](daily_returns)
    # Run without fees, fee levels are applied exactly afterwards from the turnover
    simulators = {name: Simulator(daily_returns, strategy, 0.0) for name, strategy in strategies.items()}
    LatestEngine(daily_returns, models, list(simulators.values())).run()

    # Plain tuples pickle reliably between worker processes
    output: dict[str, Any] = {"series": {}}
    for name, simulator in simulators.items():
        series = to_series(simulator.get_history())
        output["series"][name] = (series.growth, series.turnover, series.start)
    for strategy in strategies.values():
        if isinstance(strategy, AlphaRecorder):
            output["alphas"] = strategy.alphas
            output["residual_variance"] = strategy.residual_variance
        elif isinstance(strategy, PairRecorder):
            output["pairs"] = strategy.pairs
            output["pair_z_scores"] = strategy.z_scores
        elif isinstance(strategy, ResidualRecorder):
            output["snapback_residuals"] = strategy.residuals
    print(f"finished {job}", flush=True)
    return output

def net_returns(series: Series, fee: float, mask: np.ndarray | None = None) -> np.ndarray:
    keep = np.arange(len(series.growth)) >= series.start
    if mask is not None:
        keep &= mask
    return series.growth[keep] * (1.0 - fee * series.turnover[keep]) - 1.0

def sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or np.std(returns, ddof=1) == 0.0:
        return float("nan")
    return float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS))

def annual_return(returns: np.ndarray) -> float:
    return float(np.prod(1.0 + returns) ** (TRADING_DAYS / len(returns)) - 1.0)

def max_drawdown(returns: np.ndarray) -> float:
    values = np.cumprod(1.0 + returns)
    return float(np.min(values / np.maximum.accumulate(values)) - 1.0)

def newey_west_t(values: np.ndarray, lags: int) -> float:
    values = values[np.isfinite(values)]
    errors = values - np.mean(values)
    variance = errors @ errors / len(values)
    for lag in range(1, lags + 1):
        variance += 2.0 * (1.0 - lag / (lags + 1)) * (errors[lag:] @ errors[:-lag]) / len(values)
    return float(np.mean(values) / np.sqrt(variance / len(values)))

def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)

def pct(value: float) -> str:
    return f"{value:.1%}"

def log_price_paths(returns: np.ndarray) -> np.ndarray:
    # Row t is the log price before day t, so the move over days t + 1 to t + h is row t + 1 + h minus row t + 1
    return np.vstack((np.zeros((1, returns.shape[1])), np.cumsum(np.log1p(returns), axis=0)))

def alpha_calibration(results: dict[str, Any], log_prices: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    alphas, residual_variance = results["alphas"], results["residual_variance"]
    num_days = len(alphas)
    valid = np.all(np.isfinite(alphas), axis=1) & np.all(residual_variance > 0, axis=1)
    days = np.flatnonzero(valid & (np.arange(num_days) + horizon < num_days))
    raw = alphas[days] / residual_variance[days]
    scores = -np.clip((raw - raw.mean(axis=1, keepdims=True)) / raw.std(axis=1, keepdims=True), -3.0, 3.0)
    forward = np.exp(log_prices[days + 1 + horizon] - log_prices[days + 1]) - 1.0
    forward -= forward.mean(axis=1, keepdims=True)

    standardized = (scores - scores.mean(axis=1, keepdims=True)) / scores.std(axis=1, keepdims=True)
    slopes = np.mean(standardized * forward, axis=1)
    deciles = np.argsort(np.argsort(scores, axis=1), axis=1) * 10 // scores.shape[1]
    decile_means = np.array([
        np.mean(np.sum(forward * (deciles == d), axis=1) / np.sum(deciles == d, axis=1)) for d in range(10)
    ])
    return decile_means, slopes

Z_EDGES = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
Z_LABELS = ("z < -3", "-3 to -2", "-2 to -1", "-1 to 0", "0 to 1", "1 to 2", "2 to 3", "z > 3")

def pair_calibration(results: dict[str, Any], log_prices: np.ndarray, horizon: int) -> tuple[list[np.ndarray], np.ndarray]:
    pairs, z_scores = results["pairs"], results["pair_z_scores"]
    num_days = len(pairs)
    days = np.flatnonzero(np.all(pairs[:, :, 0] >= 0, axis=1) & (np.arange(num_days) + horizon < num_days))
    first, second = pairs[days, :, 0], pairs[days, :, 1]
    rows = days[:, np.newaxis]
    forward = (log_prices[rows + 1 + horizon, first] - log_prices[rows + 1, first]) \
        - (log_prices[rows + 1 + horizon, second] - log_prices[rows + 1, second])
    z = z_scores[days]

    signal = -z
    standardized = (signal - signal.mean(axis=1, keepdims=True)) / signal.std(axis=1, keepdims=True)
    slopes = np.mean(standardized * (forward - forward.mean(axis=1, keepdims=True)), axis=1)
    buckets = np.digitize(z, Z_EDGES)
    return [forward[buckets == index] for index in range(len(Z_LABELS))], slopes

def snapback_scores(residuals: np.ndarray, signal_days: int = ResidualSnapback.signal_days,
                    volatility_days: int = ResidualSnapback.volatility_days) -> np.ndarray:
    # Same score as ResidualSnapback for every stock and day, NaN where any residual in the window is missing
    finite = np.isfinite(residuals)
    filled = np.where(finite, residuals, 0.0)
    zero = np.zeros((1, residuals.shape[1]))
    total = np.vstack((zero, np.cumsum(filled, axis=0)))
    squares = np.vstack((zero, np.cumsum(filled ** 2, axis=0)))
    count = np.vstack((zero, np.cumsum(finite, axis=0)))
    window = signal_days + volatility_days
    days = np.arange(window - 1, len(residuals))
    # Prior residuals are rows days - window + 1 to days - signal_days, recent ones the last signal_days rows
    low, middle, high = days - window + 1, days - signal_days + 1, days + 1
    prior_sum = total[middle] - total[low]
    variance = (squares[middle] - squares[low] - prior_sum ** 2 / volatility_days) / (volatility_days - 1)
    recent = total[high] - total[middle]
    valid = (count[high] - count[low] == window) & (variance > 0)
    scores = np.full(residuals.shape, np.nan)
    scores[days] = np.where(valid, recent / np.sqrt(np.where(valid, variance, 1.0) * signal_days), np.nan)
    return scores

def rolling_residual_means(residuals: np.ndarray, window: int = 60) -> tuple[np.ndarray, np.ndarray]:
    # Each stock's mean residual over the trailing window, NaN unless all window residuals exist
    finite = np.isfinite(residuals)
    zero = np.zeros((1, residuals.shape[1]))
    total = np.vstack((zero, np.cumsum(np.where(finite, residuals, 0.0), axis=0)))
    count = np.vstack((zero, np.cumsum(finite, axis=0)))
    days = np.arange(window - 1, len(residuals))
    full = count[days + 1] - count[days + 1 - window] == window
    means = np.where(full, (total[days + 1] - total[days + 1 - window]) / window, np.nan)
    keep = np.sum(full, axis=1) > 100
    return days[keep], means[keep]

def residual_diagnostics(residuals: np.ndarray, signal_days: int = ResidualSnapback.signal_days) -> dict[str, float]:
    # Checks the two assumptions behind the score: residuals average zero, and they are uncorrelated over time,
    # so the sum of signal_days residuals has signal_days times the variance of one
    days = np.flatnonzero(np.sum(np.isfinite(residuals), axis=1) > 100)
    values = residuals[days]
    cross_mean = np.nanmean(values, axis=1)
    _, rolling = rolling_residual_means(residuals)
    centered = values - np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0, ddof=1)
    autocorrelation = {}
    for lag in (1, 2):
        product = centered[lag:] * centered[:-lag]
        autocorrelation[lag] = float(np.nanmedian(np.nanmean(product, axis=0) / std ** 2))
    sums = np.lib.stride_tricks.sliding_window_view(values, signal_days, axis=0).sum(axis=-1)
    ratio = np.nanvar(sums, axis=0, ddof=1) / (signal_days * std ** 2)
    scores = snapback_scores(residuals)
    return {
        "mean": float(np.nanmean(values)),
        "t": newey_west_t(cross_mean, 5),
        "std": float(np.nanmedian(std)),
        "rolling p10": float(np.nanmean(np.nanpercentile(rolling, 10, axis=1))),
        "rolling p90": float(np.nanmean(np.nanpercentile(rolling, 90, axis=1))),
        "autocorrelation 1": autocorrelation[1],
        "autocorrelation 2": autocorrelation[2],
        "variance ratio": float(np.nanmedian(ratio)),
        "score std": float(np.nanstd(scores)),
    }

def snapback_calibration(scores: np.ndarray, log_prices: np.ndarray,
                         horizon: int) -> tuple[list[np.ndarray], np.ndarray]:
    # Target = each stock's return over the next h days minus the average of the stocks with a valid score
    num_days = len(scores)
    valid = np.isfinite(scores)
    days = np.flatnonzero((np.sum(valid, axis=1) > 100) & (np.arange(num_days) + horizon < num_days))
    score, mask = scores[days], valid[days]
    forward = np.exp(log_prices[days + 1 + horizon] - log_prices[days + 1]) - 1.0
    counts = mask.sum(axis=1, keepdims=True)
    forward = forward - np.sum(np.where(mask, forward, 0.0), axis=1, keepdims=True) / counts

    signal = np.where(mask, -score, 0.0)
    mean = signal.sum(axis=1, keepdims=True) / counts
    deviation = np.where(mask, signal - mean, 0.0)
    standardized = deviation / np.sqrt(np.sum(deviation ** 2, axis=1, keepdims=True) / counts)
    slopes = np.sum(standardized * np.where(mask, forward, 0.0), axis=1) / counts[:, 0]
    buckets = np.digitize(np.where(mask, score, 0.0), Z_EDGES)
    return [forward[mask & (buckets == index)] for index in range(len(Z_LABELS))], slopes

def market_beta(series: Series, market: np.ndarray, fee: float = TRADE_FEE) -> float:
    # Market = equal-weighted average of the universe, as in scripts/alpha_strategies.py
    returns = net_returns(series, fee)
    benchmark = market[np.arange(len(series.growth)) >= series.start]
    return float(np.cov(returns, benchmark)[0, 1] / np.var(benchmark, ddof=1))

STRATEGY_NAMES = {
    "tb": "Contrarian Treynor-Black",
    "pairs": "PCA pairs trading",
    "snap": "Residual snapback",
    "snaph": "Residual snapback (hedged)",
}
# Variants are compared over a common period within each group so the Treynor-Black and pairs results do not move
GROUPS = (("tb", "pairs"), ("snap", "snaph"))

def common_starts(series: dict[str, Series]) -> dict[str, int]:
    starts = {}
    for group in GROUPS:
        start = max(s.start for name, s in series.items() if name.split("|")[0] in group)
        starts.update({key: start for key in group})
    return starts

def report(results: dict[str, Any], daily_returns: DailyReturns) -> str:
    series: dict[str, Series] = results["series"]
    years = daily_returns.dates.astype("datetime64[Y]").astype(int) + 1970
    selection = years < HOLDOUT_START
    holdout = ~selection
    names = STRATEGY_NAMES
    market = daily_returns.returns.mean(axis=1)
    out = []

    out.append("## 0. Residual snapback variants\n")
    out.append("Each variant from its own first trade (April 2002) to December 2023. Market beta is against the "
               "equal-weighted universe, with 5bp fees.\n")
    variants = {
        "Equal-weight benchmark (weekly)": "bench|base|",
        "Default snapback (long only)": "snap|base|",
        "Stricter cutoff (score <= -2.5)": "snap|score threshold|-2.5",
        "Hybrid, 70% unhedged": "snap|hedge mix|70% unhedged",
        "Hybrid, 50% unhedged": "snap|hedge mix|50% unhedged",
        "Hybrid, 30% unhedged": "snap|hedge mix|30% unhedged",
        "Dynamic volatility mix": "snap|hedge mix|dynamic",
        "Fully hedged snapback (long-short)": "snaph|base|",
        "Earlier version: projection hedge of the long picks": "cmp|projection hedge|",
        "Earlier version: dynamic mix, unsmoothed, projection hedge": "cmp|dynamic unsmoothed|",
    }
    rows = []
    for label, name in variants.items():
        s = series[name]
        net = net_returns(s, TRADE_FEE)
        rows.append([label, f"{sharpe(net):.2f}", f"{sharpe(net_returns(s, 0.0)):.2f}", pct(annual_return(net)),
                     pct(np.std(net, ddof=1) * np.sqrt(TRADING_DAYS)), f"{market_beta(s, market):.2f}",
                     pct(max_drawdown(net))])
    for key in ("tb", "pairs"):
        rows.append([names[key], "", "", "", "", f"{market_beta(series[f'{key}|base|'], market):.2f}", ""])
    out.append(table(["Variant", "Sharpe (5bp)", "Sharpe (no fees)", "Return (5bp)", "Volatility (5bp)",
                      "Market beta", "Max drawdown (5bp)"], rows) + "\n")

    out.append("### Residual diagnostics (lagged factor model, every stock and day)\n")
    diagnostics = residual_diagnostics(results["snapback_residuals"])
    out.append(table(["Measure", "Value"], [
        ["Mean residual (bp/day)", f"{diagnostics['mean'] * 1e4:.3f}"],
        ["Newey-West t of daily cross-sectional mean", f"{diagnostics['t']:.2f}"],
        ["Median residual std (daily)", pct(diagnostics["std"])],
        ["60-day mean residual, 10th / 90th percentile across stocks (daily)",
         f"{diagnostics['rolling p10']:.2%} / {diagnostics['rolling p90']:.2%}"],
        ["Median autocorrelation, lag 1 / lag 2",
         f"{diagnostics['autocorrelation 1']:.3f} / {diagnostics['autocorrelation 2']:.3f}"],
        ["Median Var(3-day sum) / (3 Var(1 day))", f"{diagnostics['variance ratio']:.2f}"],
        ["Std of snapback score", f"{diagnostics['score std']:.2f}"],
    ]) + "\n")
    hedged = [series[name] for name in ("snaph|base|", "cmp|projection hedge|")]
    positions = [len(s.growth) - s.start for s in hedged]
    out.append(f"Hedged runs trade from day {hedged[0].start} (long-short) and {hedged[1].start} (projection), "
               f"{positions[0]} and {positions[1]} days.\n")

    out.append("## 1. Regime robustness\n")
    for key, label in names.items():
        base = series[f"{key}|base|"]
        rows = []
        log_pnl = {}
        for year in np.unique(years[base.start:]):
            gross = net_returns(base, 0.0, years == year)
            net = net_returns(base, TRADE_FEE, years == year)
            log_pnl[year] = float(np.sum(np.log1p(gross)))
            rows.append([year, pct(np.prod(1 + gross) - 1), pct(np.prod(1 + net) - 1), f"{sharpe(gross):.2f}", f"{sharpe(net):.2f}"])
        out.append(f"### {label}\n")
        out.append(table(["Year", "Return (no fees)", "Return (5bp)", "Sharpe (no fees)", "Sharpe (5bp)"], rows))

        total = sum(log_pnl.values())
        ranked = sorted(log_pnl, key=lambda year: log_pnl[year], reverse=True)
        gross_all, net_all = net_returns(base, 0.0), net_returns(base, TRADE_FEE)
        first_half = years <= 2012
        summary = [
            ["Years positive (no fees / 5bp)", f"{sum(v > 0 for v in log_pnl.values())} / "
             f"{sum(np.sum(np.log1p(net_returns(base, TRADE_FEE, years == y))) > 0 for y in log_pnl)} of {len(log_pnl)}"],
            [f"Share of total profit from best year ({ranked[0]})", pct(log_pnl[ranked[0]] / total)],
            [f"Share from best two years ({ranked[0]}, {ranked[1]})", pct((log_pnl[ranked[0]] + log_pnl[ranked[1]]) / total)],
            ["Sharpe, all years (no fees / 5bp)", f"{sharpe(gross_all):.2f} / {sharpe(net_all):.2f}"],
            [f"Sharpe excluding {ranked[0]} (no fees / 5bp)",
             f"{sharpe(net_returns(base, 0.0, years != ranked[0])):.2f} / {sharpe(net_returns(base, TRADE_FEE, years != ranked[0])):.2f}"],
            [f"Sharpe excluding {ranked[0]} and {ranked[1]} (no fees / 5bp)",
             f"{sharpe(net_returns(base, 0.0, ~np.isin(years, ranked[:2]))):.2f} / "
             f"{sharpe(net_returns(base, TRADE_FEE, ~np.isin(years, ranked[:2]))):.2f}"],
            ["Sharpe 2002-2012 / 2013-2023 (5bp)",
             f"{sharpe(net_returns(base, TRADE_FEE, first_half)):.2f} / {sharpe(net_returns(base, TRADE_FEE, ~first_half)):.2f}"],
            ["Max drawdown (no fees / 5bp)", f"{pct(max_drawdown(gross_all))} / {pct(max_drawdown(net_all))}"],
        ]
        out.append("\n" + table(["Measure", "Value"], summary) + "\n")

    out.append("## 2. Parameter sensitivity\n")
    sensitivity = {name: s for name, s in series.items() if name.split("|")[0] in names}
    starts = common_starts(sensitivity)
    out.append(f"Variants are compared over a common period starting {daily_returns.dates[starts['tb']]} for "
               f"Treynor-Black and pairs trading and {daily_returns.dates[starts['snap']]} for residual snapback "
               f"(the 1000-day PCA window needs four years of history). Selection = 2002-{HOLDOUT_START - 1}, "
               f"holdout = {HOLDOUT_START}-2023, both with 5bp fees.\n")
    for key, label in names.items():
        common = np.arange(len(years)) >= starts[key]
        rows = []
        for name, s in sensitivity.items():
            strategy, param, value = name.split("|")
            if strategy != key:
                continue
            rows.append([
                "default" if param == "base" else param, value,
                f"{sharpe(net_returns(s, 0.0, common)):.2f}", f"{sharpe(net_returns(s, TRADE_FEE, common)):.2f}",
                pct(annual_return(net_returns(s, TRADE_FEE, common))),
                f"{sharpe(net_returns(s, TRADE_FEE, common & selection)):.2f}",
                f"{sharpe(net_returns(s, TRADE_FEE, holdout)):.2f}",
            ])
        rows.sort(key=lambda row: (row[0] != "default", row[0], row[1]))
        out.append(f"### {label}\n")
        out.append(table(["Parameter", "Value", "Sharpe (no fees)", "Sharpe (5bp)", "Return (5bp)",
                          "Sharpe selection", "Sharpe holdout"], rows) + "\n")

    out.append("## 3. Trading costs\n")
    rows = []
    for key, label in names.items():
        base = series[f"{key}|base|"]
        active = np.arange(len(base.growth)) >= base.start
        gross_mean = float(np.mean(base.growth[active] - 1.0))
        cost_per_bp = float(np.mean(base.growth[active] * base.turnover[active]))
        rows.append([
            label, pct(np.mean(base.turnover[active])), f"{np.mean(base.turnover[active]) * TRADING_DAYS:.0f}x",
            pct(gross_mean * TRADING_DAYS), pct(TRADE_FEE * cost_per_bp * TRADING_DAYS),
            pct(TRADE_FEE * cost_per_bp / gross_mean), f"{gross_mean / cost_per_bp * 1e4:.1f}bp",
        ])
    out.append(table(["Strategy", "Daily turnover", "Annual turnover", "Gross return (mean, annual)",
                      "Fee drag at 5bp (annual)", "Share of edge lost at 5bp", "Break-even fee"], rows) + "\n")
    fees = (0.0, 0.0002, 0.0005, 0.0010, 0.0020)
    rows = [[label] + [f"{sharpe(net_returns(series[f'{key}|base|'], fee)):.2f}" for fee in fees] for key, label in names.items()]
    out.append(table(["Sharpe at fee of"] + [f"{fee * 1e4:g}bp" for fee in fees], rows) + "\n")

    out.append("## 4. Calibration\n")
    log_prices = log_price_paths(daily_returns.returns)
    horizons = (1, 5, 20)

    out.append("### Contrarian Treynor-Black\n")
    out.append("Signal = the strategy's score (minus the capped z-score of alpha / residual variance). Target = each stock's "
               "return over the next h days minus the cross-sectional average. Deciles are formed each day; decile 10 "
               "is what the strategy buys most.\n")
    decile_rows = []
    slope_rows = []
    for h in horizons:
        decile_means, slopes = alpha_calibration(results, log_prices, h)
        decile_rows.append([f"{h}d"] + [f"{m * 1e4:.1f}" for m in decile_means])
        slope_rows.append([h, f"{np.mean(slopes) * 1e4:.2f}bp", f"{newey_west_t(slopes, h):.1f}"])
    out.append(table(["Horizon"] + [f"D{d + 1}" for d in range(10)], decile_rows))
    out.append("\nMean excess return by signal decile, in bp (D1 = most shorted, D10 = most bought).\n")
    out.append(table(["Horizon (days)", "Return per 1 s.d. of signal", "t-stat (Newey-West)"], slope_rows) + "\n")

    out.append("### PCA pairs trading\n")
    out.append("Signal = minus the spread z-score (positive = buy the first stock, short the second). Target = change in "
               "the log price ratio of the pair over the next h days, in the direction of the signal's first leg.\n")
    bucket_rows = [[label] for label in Z_LABELS]
    slope_rows = []
    for h in horizons:
        buckets, slopes = pair_calibration(results, log_prices, h)
        slope_rows.append([h, f"{np.mean(slopes) * 1e4:.2f}bp", f"{newey_west_t(slopes, h):.1f}"])
        for index, selected in enumerate(buckets):
            bucket_rows[index].append(f"{np.mean(selected) * 1e4:.1f} (n={len(selected):,})")
    out.append(table(["Spread z-score"] + [f"Next {h}d (bp)" for h in horizons], bucket_rows))
    out.append("\nThe strategy buys the spread below z = -2 and sells it above z = 2, so returns should fall as z rises.\n")
    out.append(table(["Horizon (days)", "Spread return per 1 s.d. of signal", "t-stat (Newey-West)"], slope_rows) + "\n")

    out.append("### Residual snapback\n")
    out.append("Signal = minus the snapback score (the last 3 days of residuals over the prior 60-day residual volatility, "
               "scaled by sqrt 3), for every stock on every day. Target = each stock's return over the next h days minus "
               "the cross-sectional average.\n")
    scores = snapback_scores(results["snapback_residuals"])
    bucket_rows = [[label] for label in Z_LABELS]
    slope_rows = []
    for h in horizons:
        buckets, slopes = snapback_calibration(scores, log_prices, h)
        slope_rows.append([h, f"{np.mean(slopes) * 1e4:.2f}bp", f"{newey_west_t(slopes, h):.1f}"])
        for index, selected in enumerate(buckets):
            bucket_rows[index].append(f"{np.mean(selected) * 1e4:.1f} (n={len(selected):,})")
    out.append(table(["Snapback score"] + [f"Next {h}d (bp)" for h in horizons], bucket_rows))
    out.append("\nThe strategy buys up to 20 of the lowest scores at or below -2, so returns should fall as the score rises.\n")
    out.append(table(["Horizon (days)", "Excess return per 1 s.d. of signal", "t-stat (Newey-West)"], slope_rows) + "\n")

    out.append("## 5. Holdout\n")
    out.append(f"Choices are made using only 2002-{HOLDOUT_START - 1}, then evaluated once on {HOLDOUT_START}-2023 "
               "(Sharpe with 5bp fees).\n")
    rows = []
    for key, label in names.items():
        common = np.arange(len(years)) >= starts[key]
        base = series[f"{key}|base|"]
        rows.append([label, "Default settings", f"{sharpe(net_returns(base, TRADE_FEE, common & selection)):.2f}",
                     f"{sharpe(net_returns(base, TRADE_FEE, holdout)):.2f}"])
        candidates = {name: s for name, s in sensitivity.items() if name.startswith(f"{key}|")}
        best = max(candidates, key=lambda name: sharpe(net_returns(candidates[name], TRADE_FEE, common & selection)))
        _, param, value = best.split("|")
        rows.append([label, f"Best setting on selection period ({'default' if param == 'base' else f'{param} = {value}'})",
                     f"{sharpe(net_returns(candidates[best], TRADE_FEE, common & selection)):.2f}",
                     f"{sharpe(net_returns(candidates[best], TRADE_FEE, holdout)):.2f}"])
        best_k = max(K_VALUES, key=lambda k: sharpe(net_returns(series[f"k{k}|{key}|oos"], TRADE_FEE, selection)))
        rows.append([label, f"Best K on selection period (K = {best_k})",
                     f"{sharpe(net_returns(series[f'k{best_k}|{key}|oos'], TRADE_FEE, selection)):.2f}",
                     f"{sharpe(net_returns(series[f'k{best_k}|{key}|oos'], TRADE_FEE, holdout)):.2f}"])
    tb = series["tb|base|"]
    rows.append(["Treynor-Black, following alpha (not contrarian)", "Sign check",
                 f"{-sharpe(net_returns(tb, 0.0, selection)):.2f} (no fees)", f"{-sharpe(net_returns(tb, 0.0, holdout)):.2f} (no fees)"])
    out.append(table(["Strategy", "Choice", f"Sharpe 2002-{HOLDOUT_START - 1}", f"Sharpe {HOLDOUT_START}-2023"], rows) + "\n")

    out.append("## 6. Number of factors K\n")
    out.append("Out-of-sample = PCA and alphas re-estimated on trailing windows (no look-ahead). In-sample = PCA and alphas "
               "estimated once on the full 2000-2023 sample, then traded over the same period (look-ahead). "
               "Sharpe ratios; 5bp fees only where stated.\n")
    rows = []
    for k in K_VALUES:
        rows.append([
            k,
            f"{sharpe(net_returns(series[f'k{k}|tb|oos'], 0.0)):.2f}",
            f"{sharpe(net_returns(series[f'k{k}|tb|oos'], TRADE_FEE)):.2f}",
            f"{sharpe(net_returns(series[f'k{k}|tb|is'], 0.0)):.2f}",
            f"{sharpe(net_returns(series[f'k{k}|tb follow|is'], 0.0)):.2f}",
            f"{sharpe(net_returns(series[f'k{k}|pairs|oos'], 0.0)):.2f}",
            f"{sharpe(net_returns(series[f'k{k}|pairs|oos'], TRADE_FEE)):.2f}",
            f"{sharpe(net_returns(series[f'k{k}|pairs|is'], 0.0)):.2f}",
        ])
    out.append(table(["K", "T-B out-of-sample", "T-B out-of-sample (5bp)", "T-B in-sample (contrarian)",
                      "T-B in-sample (following alpha)", "Pairs out-of-sample", "Pairs out-of-sample (5bp)",
                      "Pairs in-sample"], rows) + "\n")
    rows = []
    for k in K_VALUES:
        rows.append([k] + [
            f"{sharpe(net_returns(series[f'k{k}|{key}|{sample}'], fee)):.2f}"
            for key in ("snap", "snaph") for sample, fee in (("oos", 0.0), ("oos", TRADE_FEE), ("is", 0.0))
        ])
    out.append("Residual snapback. In-sample = PCA and regression estimated once on the full sample; residuals are "
               "still computed day by day from those fixed coefficients.\n")
    out.append(table(["K", "Snapback out-of-sample", "Snapback out-of-sample (5bp)", "Snapback in-sample",
                      "Hedged out-of-sample", "Hedged out-of-sample (5bp)", "Hedged in-sample"], rows) + "\n")
    return "\n".join(out)

def load_results() -> dict[str, Any]:
    with open(CACHE_FILE, "rb") as file:
        results = pickle.load(file)
    results["series"] = {
        name: Series(growth=growth, turnover=turnover, start=start)
        for name, (growth, turnover, start) in results["series"].items()
    }
    return results

def run_jobs(jobs: list[str], processes: int) -> None:
    # Results are merged into the existing cache, so jobs can be added without rerunning the others
    results: dict[str, Any] = {"series": {}}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as file:
            results = pickle.load(file)
    with multiprocessing.Pool(processes=processes) as pool:
        outputs = pool.map(run_job, jobs, chunksize=1)
    for output in outputs:
        results["series"].update(output.pop("series"))
        results.update(output)
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "wb") as file:
        pickle.dump(results, file)

def main() -> None:
    # --rerun (or no cache) runs every job, --jobs a,b runs only those and merges them into the cache
    daily_returns = data.load_daily_returns()
    processes = 4
    if "--rerun" in sys.argv or not os.path.exists(CACHE_FILE):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        run_jobs(JOB_ORDER, processes)
    elif "--jobs" in sys.argv:
        run_jobs(sys.argv[sys.argv.index("--jobs") + 1].split(","), processes)

    results = load_results()
    text = report(results, daily_returns)
    with open(REPORT_FILE, "w") as file:
        file.write(text + "\n")
    print(text)

if __name__ == "__main__":
    main()
