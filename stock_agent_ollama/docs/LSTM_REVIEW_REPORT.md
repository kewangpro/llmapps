# LSTM Training & Prediction Design Review

**Date:** January 9, 2026
**Module:** `src/tools/lstm`
**Status:** Reviewed & Patched

## 1. Architectural Overview
The system employs a **Walk-Forward Ensemble LSTM** architecture, designed for robustness and stability in financial time-series forecasting.

-   **Ensemble Strategy:** Uses 3 independent models trained on expanding time windows (Walk-Forward Validation). This minimizes overfitting to specific market regimes and provides a more reliable consensus prediction.
-   **Model Architecture:** 
    -   3 stacked LSTM layers (64, 64, 32 units).
    -   **Attention Mechanism** to weigh the importance of different past time steps.
    -   **Batch Normalization** and **Dropout** (0.2-0.3) for regularization.
    -   **Optimizer:** Adam with Gradient Clipping (`clipnorm=1.0`) to prevent exploding gradients.
-   **Feature Engineering:**
    -   **17 Enhanced Features:** OHLCV + Technical Indicators (RSI, MACD, Bollinger Bands, ATR, Momentum).
    -   **Robust Scaling:** Uses a `CompositeScaler` that applies `RobustScaler` (quantile-based) for volatile features and `MinMaxScaler` for others. Includes "aggressive" outlier handling (Winsorization) to prevent data artifacts from skewing the model.

## 2. Implementation Quality & Strengths
-   **Data Leakage Prevention:** The `data_pipeline.py` strictly separates training and validation data *before* fitting scalers. This is a critical best practice that ensures the model's performance metrics are realistic.
-   **Resilience:** The system includes extensive error handling, model loading fallbacks (supporting `.keras`, `.h5`, and `SavedModel` formats), and auto-recovery mechanisms for corrupted files.
-   **Stability:** Features like "gradient clipping" and "learning rate scheduling" ensure the training process is stable even with noisy financial data.

## 3. Recent Fixes (Post-Investigation)
Three critical issues were identified and resolved during the investigation:

1.  **Scaler Inconsistency (Root Cause of "Gap"):**
    -   *Issue:* The prediction service was re-fitting the scaler on new data at runtime, invalidating the model's learned weights.
    -   *Fix:* The system now strictly preserves and uses the scaler parameters fitted during training.
2.  **Sequence Update Bug:**
    -   *Issue:* In the multi-step prediction loop, the 'Open' price for future steps was not being updated, creating inconsistent input sequences.
    -   *Fix:* The sequence updater now correctly rolls the previous 'Close' price to the next day's 'Open'.
3.  **Prediction Volatility:**
    -   *Issue:* Unconstrained recursive predictions could lead to unrealistic crashes or spikes (e.g., -10% in one day).
    -   *Fix:* A **5% daily price change clamp** was implemented to enforce realistic market physics for the projection.

## 4. Design Limitations (The "Flatness" Factor)
Users may observe that predictions "flatten out" or drift towards a mean value over long horizons (e.g., 30 days). This is **expected behavior** for this architecture:
-   **Recursive Prediction:** The model predicts t+1, feeds it back to predict t+2, etc. Errors accumulate at each step.
-   **Mean Reversion:** Without new information (external shocks, news), LSTMs tend to converge to the trend line or mean of their training data.
-   **Verdict:** The model is best used for **short-term (1-7 day) directional signals**. The 30-day projection should be viewed as a "sentiment trajectory," not a precise price target.

## 5. Recommendations for Future Improvement
1.  **Direct Horizon Models:** To improve 30-day accuracy, train separate models specifically to predict the price at t+30 directly (Sequence-to-Point), bypassing the recursive loop.
2.  **News Sentiment Integration:** The current model is purely technical. Feeding the "News Sentiment" score (from `StockAgent`) as an input feature would help the LSTM anticipate trend shifts caused by fundamental events.
3.  **Regular Retraining:** Since the scaler is now static, the model must be retrained regularly (e.g., weekly) to "learn" new price levels, especially when a stock breaks out to all-time highs.

---
**Conclusion:** The implementation is mathematically sound and follows industry best practices. The recent patches have resolved the immediate stability issues, making the tool reliable for its intended purpose.
