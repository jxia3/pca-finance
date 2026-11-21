"""Self-financing execution with unchanged, non-tradable missing-price marks.

Trade arrays are dollar trades divided by pre-fee NAV. Position arrays are
post-fee NAV weights. Keeping these units distinct preserves frozen holdings.
"""

import numpy as np
from scipy.optimize import brentq

from src.modules.simulator import Simulator


class HoldoutSimulator(Simulator):
    """Separate accounting adapter. Does not change the shared simulator."""

    def update(self, timestamp, returns, model_outputs):
        raw = returns[-1]
        missing = ~np.isfinite(raw)
        marked = np.where(missing, 0.0, raw)
        growth = 1.0 + float(self.positions @ marked)
        if not np.isfinite(growth) or growth <= 0:
            raise RuntimeError("Portfolio is insolvent before trading.")
        before_fees = self.total_value * growth
        drifted = self.positions * (1.0 + marked) / growth
        desired = np.asarray(model_outputs["target"], dtype=float)
        if desired.shape != drifted.shape or not np.isfinite(desired).all():
            raise ValueError("Invalid target weights.")
        if np.abs(desired).sum() > 1 + 1e-8:
            raise ValueError("Active target gross exceeds one.")
        frozen = np.where(missing, drifted, 0.0)
        frozen_gross = float(np.abs(frozen).sum())
        active = np.where(missing, 0.0, desired)

        def holdings(equity_fraction):
            # Frozen dollar holdings are independent of fees on other stocks.
            return frozen + active * max(0.0, equity_fraction - frozen_gross)

        def balance(equity_fraction):
            turnover = np.abs(holdings(equity_fraction) - drifted).sum()
            return equity_fraction + self.trade_fee * turnover - 1.0

        if balance(0.0) >= 0:
            raise RuntimeError("Trading costs exhaust portfolio equity.")
        remaining = brentq(balance, 0.0, 1.0, xtol=1e-14)
        dollars = holdings(remaining)
        trades = dollars - drifted
        fees = before_fees * self.trade_fee * np.abs(trades).sum()
        self.total_value = before_fees * remaining
        self.positions = dollars / remaining
        j = self.history_index
        self.history.total_value[j] = self.total_value
        self.history.positions[j] = self.positions
        self.history.trades[j] = trades
        self.history.fees[j] = fees
        self.history_index += 1
