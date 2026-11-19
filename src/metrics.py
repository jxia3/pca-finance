import numpy as np

from src.modules.simulator import PortfolioHistory

TRADING_DAYS = 252

def summarize(name: str, history: PortfolioHistory, market_returns: np.ndarray) -> None:
    # Skip the warmup period before the strategy holds any positions
    active = np.flatnonzero(np.any(history.positions != 0.0, axis=1))
    if len(active) == 0:
        print(f"{name:<32} no positions opened")
        return
    start = active[0]
    values = history.total_value[start:]
    returns = values[1:] / values[:-1] - 1.0
    if len(returns) < 2:
        print(f"{name:<32} insufficient active history")
        return
    market = market_returns[start:]

    beta = np.cov(returns, market)[0, 1] / np.var(market, ddof=1)
    annual_return = (values[-1] / values[0]) ** (TRADING_DAYS / len(returns)) - 1.0
    annual_vol = np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS)
    turnover = np.mean(np.sum(np.abs(history.trades[start + 1:]), axis=1))
    print(
        f"{name:<32} return {annual_return:7.2%}  vol {annual_vol:6.2%}  sharpe {sharpe:5.2f}  "
        f"market beta {beta:5.2f}  daily turnover {turnover:6.2%}"
    )