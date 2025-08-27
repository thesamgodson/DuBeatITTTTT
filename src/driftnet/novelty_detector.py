import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

class ForecastDivergenceDetector:
    """
    Detects regime shifts by monitoring for persistent, structured forecast errors.
    This is the "meta-cognitive monitor" of the DriftNet architecture.
    """
    def __init__(self, error_window_size: int = 30, threshold_quantile: float = 0.95):
        """
        Initializes the detector.

        Args:
            error_window_size (int): The rolling window size for calculating error stats.
            threshold_quantile (float): The quantile to use for the adaptive threshold
                                        (e.g., 0.95 means the error must be in the top 5%).
        """
        self.error_window_size = error_window_size
        self.threshold_quantile = threshold_quantile
        self.error_history = []

    def _get_adaptive_threshold(self) -> float:
        """Calculates the current novelty threshold based on error history."""
        if len(self.error_history) < self.error_window_size:
            return np.inf # Not enough data to set a threshold

        return np.quantile(self.error_history, self.threshold_quantile)

    def check_novelty(self, forecast_error: float) -> bool:
        """
        Checks if the current state constitutes a novelty event (a forecast crisis).

        A crisis is defined by:
        1. A forecast error that is unusually high compared to recent history.
        2. The recent forecast errors are autocorrelated, meaning the error is
           systematic, not random noise.

        Args:
            forecast_error (float): The forecast error (e.g., MSE) for the current step.

        Returns:
            bool: True if a novelty event is detected.
        """
        is_novel = False

        # 1. Check if the error exceeds the adaptive threshold
        adaptive_threshold = self._get_adaptive_threshold()
        if forecast_error > adaptive_threshold:

            # 2. Check for autocorrelation in recent errors
            # We need at least a few errors to compute ACF
            if len(self.error_history) > 10:
                # The [1] gets the first lag's autocorrelation
                try:
                    residual_autocorr = acf(self.error_history, nlags=1, fft=False)[1]

                    if residual_autocorr > 0.5:
                        print(f"Novelty detected! Error {forecast_error:.4f} > Threshold {adaptive_threshold:.4f} AND ACF {residual_autocorr:.2f} > 0.5")
                        is_novel = True
                except Exception as e:
                    # acf can sometimes fail with certain patterns of data
                    print(f"Could not compute ACF: {e}")
                    pass

        # Update error history
        self.error_history.append(forecast_error)
        if len(self.error_history) > self.error_window_size:
            self.error_history.pop(0)

        return is_novel
