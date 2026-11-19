import numpy as np

def neutralize(weights: np.ndarray, betas: np.ndarray | None) -> np.ndarray:
    assert weights.ndim == 1
    exposures = np.ones((len(weights), 1), dtype=np.float64)
    if betas is not None:
        assert betas.ndim == 2 and betas.shape[1] == len(weights)
        exposures = np.hstack((exposures, betas.T))

    # Subtract the projection onto the exposures so the dollar and factor exposures are zero
    hedge = np.linalg.lstsq(exposures, weights, rcond=None)[0]
    return weights - exposures @ hedge

def scale_gross(weights: np.ndarray, gross_leverage: float) -> np.ndarray:
    assert weights.ndim == 1 and gross_leverage > 0
    total = float(np.sum(np.abs(weights)))
    if total == 0.0:
        return np.zeros_like(weights)
    return weights * gross_leverage / total