import numpy as np

from src import data
from src.engine import Engine, ModelEntry
from src.metrics import summarize
from src.models.pairs import PairsModel
from src.models.pca import PCAModel
from src.modules.simulator import Simulator
from src.strategies.pairs_trading import PairsTrading

TRADE_FEE = 0.0005

daily_returns = data.load_daily_returns()
pca_model = PCAModel(5, 500, 5)
loading_pairs = PairsModel("pca", selection="loadings")
correlation_pairs = PairsModel("pca", selection="correlation")
simulators = {
    "Pairs (PCA loadings)": Simulator(daily_returns, PairsTrading("loading_pairs"), TRADE_FEE),
    "Pairs (correlation)": Simulator(daily_returns, PairsTrading("correlation_pairs"), TRADE_FEE),
}
engine = Engine(
    daily_returns=daily_returns,
    models=[
        ModelEntry("pca", pca_model),
        ModelEntry("loading_pairs", loading_pairs),
        ModelEntry("correlation_pairs", correlation_pairs),
    ],
    modules=list(simulators.values()),
)
engine.run()

# Equal weighted average of the universe as a market proxy
market_returns = np.mean(daily_returns.returns, axis=1)
for name, simulator in simulators.items():
    summarize(name, simulator.get_history(), market_returns)