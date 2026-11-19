import numpy as np

from src import data
from src.engine import Engine, ModelEntry
from src.metrics import summarize
from src.models.factor import FactorModel
from src.models.pca import PCAModel
from src.modules.simulator import Simulator
from src.strategies.alpha_quintile import AlphaQuintile
from src.strategies.treynor_black import TreynorBlack
from src.strategies.factor_neutral_alpha import FactorNeutralAlpha
from src.strategies.multi_horizon_alpha import MultiHorizonAlpha

TRADE_FEE = 0.0005

daily_returns = data.load_daily_returns()
pca_model = PCAModel(5, 500, 5)
factor_model = FactorModel("pca", 5, 250, 1)
short_factor_model = FactorModel("pca", 5, 60, 1)
simulators = {
    "Alpha quintile": Simulator(daily_returns, AlphaQuintile("factor"), TRADE_FEE),
    "Alpha quintile (hedged)": Simulator(daily_returns, AlphaQuintile("factor", factor_neutral=True), TRADE_FEE),
    "Treynor-Black (hedged)": Simulator(daily_returns, TreynorBlack("factor"), TRADE_FEE),
    "Contrarian quintile": Simulator(daily_returns, AlphaQuintile("factor", contrarian=True), TRADE_FEE),
    "Contrarian quintile (hedged)": Simulator(
        daily_returns, AlphaQuintile("factor", factor_neutral=True, contrarian=True), TRADE_FEE
    ),
    "Contrarian T-B (hedged)": Simulator(daily_returns, TreynorBlack("factor", contrarian=True), TRADE_FEE),
    "Contrarian quintile 60d (hedged)": Simulator(
        daily_returns, AlphaQuintile("factor_short", factor_neutral=True, contrarian=True), TRADE_FEE
    ),
    "Multi-horizon (hedged)": Simulator(daily_returns, MultiHorizonAlpha("factor_short", "factor"), TRADE_FEE),
    "Constrained alpha": Simulator(daily_returns, FactorNeutralAlpha("factor", turnover_cost=TRADE_FEE), TRADE_FEE),
}
engine = Engine(
    daily_returns=daily_returns,
    models=[
        ModelEntry("pca", pca_model),
        ModelEntry("factor", factor_model),
        ModelEntry("factor_short", short_factor_model),
    ],
    modules=list(simulators.values()),
)
engine.run()

# Equal weighted average of the universe as a market proxy
market_returns = np.mean(daily_returns.returns, axis=1)
for name, simulator in simulators.items():
    summarize(name, simulator.get_history(), market_returns)