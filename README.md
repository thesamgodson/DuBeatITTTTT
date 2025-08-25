# Project: Advanced Time Series Forecasting for Financial Markets

This repository documents an extensive research and development project aimed at creating a novel, high-performance time series forecasting model for financial markets, specifically focusing on Bitcoin (BTC-USD) price prediction.

The project evolved through several distinct phases, exploring multiple advanced architectures. This document serves as a record of that journey, the models implemented, their performance, and the key learnings.

## Table of Contents
1. [Initial Project: Topological N-BEATS](#1-initial-project-topological-n-beats)
2. [Project Pivot 1: HyperGraph-BEATS](#2-project-pivot-1-hypergraph-beats)
3. [Project Pivot 2: MarketMind Model](#3-project-pivot-2-marketmind-model)
4. [Final Code Structure](#4-final-code-structure)
5. [How to Run Tests](#5-how-to-run-tests)

---

## 1. Initial Project: Topological N-BEATS

The project began with the goal of integrating Topological Data Analysis (TDA) with the N-BEATS architecture.

### Phase 1: Data Pipeline Foundation
- **Goal:** Establish a rock-solid data processing pipeline.
- **Implementation:** Created a data loader for yfinance data and a simple feature engineer.
- **Outcome:** ✅ Success. A reliable data pipeline was built and validated.

### Phase 2: SOM Implementation
- **Goal:** Implement Self-Organizing Maps (SOMs) to create topological features.
- **Implementation:** A `SimpleSOM` class was created to train SOMs on price patterns.
- **Outcome:** ✅ Success. The SOMs were shown to cluster different patterns effectively, achieving a quantization error of **0.033** (well below the target of 0.5).

### Phase 3: Baseline N-BEATS Model
- **Goal:** Create a pure N-BEATS model to serve as a strong baseline.
- **Implementation:** A standard N-BEATS model with Trend, Seasonality, and Generic stacks was built.
- **Outcome:** ✅ Success. After tuning, the model achieved a price prediction **RMSE of ~777**, surpassing the target of < 800. This was the best-performing model from the initial project.

### Phase 4: SOM Integration (Unsuccessful)
- **Goal:** Integrate the SOM features into the N-BEATS model to improve performance.
- **Implementation:** Multiple strategies were attempted:
    1.  Injecting SOM neuron coordinates into the N-BEATS blocks.
    2.  Using the historical predictive value of neurons as features.
    3.  Using stack-specific projections for the features.
    4.  Applying strong regularization (learnable weights, weight decay).
- **Outcome:** ❌ **Failure.** Despite extensive experimentation, no integration strategy was able to outperform the pure N-BEATS baseline. The final model achieved a directional accuracy of **54.17%**, which was not a significant improvement.

---

## 2. Project Pivot 1: HyperGraph-BEATS

Based on the finding that the initial approach was not sufficiently novel, the project pivoted to a `HyperGraph-BEATS` model.

### Phase 1 (Redux): New Data Pipeline
- **Goal:** Create a new, more robust, and modular data pipeline.
- **Implementation:** A new pipeline was implemented with distinct classes for data loading, pattern extraction, and temporal alignment.
- **Outcome:** ✅ Success. The new pipeline was validated by a comprehensive, self-contained test suite.

### Phase 2: Hypergraph Builder
- **Goal:** Convert time series data into a hypergraph structure.
- **Implementation:** A `HypergraphBuilder` was created. It identifies groups of similar price patterns and represents them as hyperedges using a "clique expansion" algorithm.
- **Outcome:** ✅ Success. The module was validated by unit tests.

### Phase 3: Hypergraph Convolution Layer
- **Goal:** Implement the core message-passing layer for the new model.
- **Implementation:** A custom `CustomHypergraphConv` PyTorch module was built based on the formulation from the "Hypergraph Convolution" paper.
- **Outcome:** ✅ Success. The layer was validated by unit tests confirming its architecture and trainability.

### Phase 4 & 5: Model Integration & Training (Unsuccessful)
- **Goal:** Integrate the components into the final `HyperGraphNBEATS` model and achieve >56% directional accuracy.
- **Implementation:** The final model was built, using the hypergraph layer to create a gating mechanism for the N-BEATS stacks. The model was trained on the task of predicting returns.
- **Outcome:** ❌ **Failure.** After extensive debugging of the complex architecture, the final model achieved a directional accuracy of only **51.60%**, failing to meet the performance target.

---

## 3. Project Pivot 2: MarketMind Model

Based on the failure of the graph-based approach, the project pivoted a final time to a novel architecture based on **Market Microstructure Theory**.

### Phase 1: Microstructure Feature Pipeline
- **Goal:** Engineer features that proxy the behavior of different market participants.
- **Implementation:** A new data pipeline was created to calculate features for liquidity (e.g., `LiquidityProxy`), informed trading (`InformedTradingProxy`), and noise trading (`NoiseTradingProxy`).
- **Outcome:** ✅ Success. The feature pipeline was implemented and validated.

### Phase 2: Participant Modules
- **Goal:** Create separate neural network modules for each participant type.
- **Implementation:** `InformedTraderModule`, `NoiseTraderModule`, and `MarketMakerModule` were created as GRU-based networks.
- **Outcome:** ✅ Success. The modules were validated by unit tests.

### Phase 3: Final Aggregator Model
- **Goal:** Combine the participant signals using a regime-aware attention mechanism.
- **Implementation:** The final `MarketMindModel` was built. It uses a `RegimeDetectionModule` to create a "query" and an attention layer to create a weighted average of the participant module outputs.
- **Outcome:** ✅ Success. The final model architecture was implemented and validated by unit tests confirming its structure and trainability.

### Phase 4: Final Performance Test (Unsuccessful)
- **Goal:** Train the `MarketMindModel` and achieve >65% directional accuracy.
- **Implementation:** A full end-to-end training and evaluation script was created.
- **Outcome:** ❌ **Failure.** The final model achieved a directional accuracy of only **50.93%**, failing to meet the performance target.

---

## 4. Final Code Structure

The final state of the codebase reflects the 'MarketMind' project.

-   `src/data_processing/`: Contains the final data pipeline for creating microstructure features.
-   `src/market_mind/`: Contains the participant modules and the final aggregator model.
-   `tests/`: Contains unit tests for all implemented components.

---

## 5. How to Run Tests

The project is configured with a discoverable test suite. To run all validation tests for the final project state, execute the following command from the root directory:

```bash
python -m unittest discover tests
```
