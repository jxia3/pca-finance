from src import data
from src.engine import Engine, ModelEntry
from src.models.factor import FactorModel
from src.models.pca import PCAModel
from src.modules.simulator import Simulator
from src.strategies.alpha_persistence import AlphaPersistence

daily_returns = data.load_daily_returns()
print("Dates:", daily_returns.dates)
print("Stock IDs:", daily_returns.stock_ids)
print("Returns:", daily_returns.returns)

pca_model = PCAModel(5, 500, 5)
factor_model = FactorModel("pca", 5, 100, 1)
alpha_persistence = AlphaPersistence("pca")
simulator = Simulator(len(daily_returns.stock_ids), alpha_persistence)
engine = Engine(
    daily_returns=daily_returns,
    models=[ModelEntry("pca", pca_model), ModelEntry("factor", factor_model)],
    modules=[simulator],
)
engine.run()