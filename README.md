# An Empirical Investigation into Model Complexity for Financial Forecasting

This repository documents an extensive research project on a single, challenging task: predicting the next-day direction of the BTC-USD price. It serves as a case study and a collection of reusable modules for time series analysis.

The project's journey evolved through several distinct, increasingly complex architectures. The results provide a clear and valuable narrative about the trade-offs between model complexity and performance in a noisy, efficient market.

---

## 1. Key Findings & Post-Mortem Analysis

The most critical finding of this project is that for this specific task, **increased architectural complexity did not lead to improved predictive performance.** In fact, the simplest model we tested, a well-tuned baseline N-BEATS architecture, remained the most effective.

-   **The Narrative of Diminishing Returns:** Our experiments showed a consistent pattern:
    1.  A **Pure N-BEATS Model** established a strong baseline (Price RMSE of ~777).
    2.  Integrating **Topological Features (SOMs)** failed to improve upon the baseline, achieving a directional accuracy of only **~54%**.
    3.  Pivoting to a novel **HyperGraph-BEATS** model also failed to show an advantage, with directional accuracy struggling at **~52%**.
    4.  A final pivot to a **MarketMind Model**, based on economic microstructure theory, likewise did not succeed, with a final accuracy of **~51%**.

-   **Hypothesis for Failure of Complex Models:** Our analysis suggests that the primary challenge is not the model's capacity to learn, but the nature of the data itself. Financial time series are characterized by a very low signal-to-noise ratio. We hypothesize that the advanced architectures, while theoretically powerful, were ultimately too complex for the data, leading them to overfit to noise rather than capturing the weak underlying signal. The simpler N-BEATS model, with fewer degrees of freedom, provided a form of architectural regularization that proved more effective.

-   **Value of This Repository:** The "failure" to build a highly accurate predictive model is, in itself, a valuable research finding. This repository's primary value is not in a single "winning" model, but in:
    -   **Reusable Components:** The data pipelines, feature engineering functions, and custom layers (like the Hypergraph Convolution) are modular, tested, and can be repurposed for other research. The `src/data_processing` module is particularly robust.
    -   **An Empirical Case Study:** This serves as a real-world example of the challenges of applying complex deep learning architectures to efficient market data.

---

## 2. Guide to a Reusable Component: The Baseline N-BEATS Model

The most successful and robust model built during this project was the `NBEATSWithSOM` architecture from our initial experiments, which can be simplified to a pure N-BEATS model by not providing the SOM features. This model is a strong starting point for any time series forecasting task.

-   **Location:** `src/nbeats/`
-   **Validation:** The baseline's performance was validated in `tests/test_reproduce_baseline.py`.

---

## 3. Experimental Architectures (For Educational & Research Purposes)

The following architectures were implemented and tested. While they did not outperform the baseline in this project, their code serves as a valuable implementation reference for these advanced concepts.

### HyperGraph-BEATS
-   **Concept:** Models higher-order relationships between time steps using a hypergraph, which then gates the N-BEATS stacks.
-   **Code:** `src/hypergraph_builder/` and `src/hypergraph_convolution/`.
-   **Final Performance:** ~52% Directional Accuracy.

### MarketMind Model
-   **Concept:** Models the market as an interaction between different participant types (Informed, Noise, Market-Maker), using an attention mechanism to aggregate their signals based on the market regime.
-   **Code:** `src/market_mind/`.
-   **Final Performance:** ~51% Directional Accuracy.

---

## 4. How to Run Tests

The project is configured with a discoverable test suite. To run all validation tests for the final project state, execute the following command from the root directory:

```bash
python -m unittest discover tests
```
