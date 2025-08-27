import numpy as np
from sklearn.linear_model import LassoCV

def get_causal_features_lasso(X_train: np.array, y_train: np.array, feature_names: list) -> list:
    """
    Uses Lasso with cross-validation to select features that are most predictive of the target.
    This serves as a proxy for causal feature selection.

    Args:
        X_train (np.array): The training data for the window, shape (samples, seq_len, num_features).
        y_train (np.array): The training targets for the window, shape (samples, forecast_horizon).
        feature_names (list): The list of original feature names.

    Returns:
        list: A list of the names of the selected features.
    """
    # Lasso works on 2D data, so we need to flatten the sequences
    n_samples, seq_len, n_features = X_train.shape
    if n_samples == 0:
        return feature_names # Return all if no data

    X_flattened = X_train.reshape(n_samples, -1)

    # We predict the first step of the forecast horizon
    y_target = y_train[:, 0]

    # Use LassoCV to find the best alpha
    # Using a small number of alphas for speed in this experimental setting
    lasso_cv = LassoCV(cv=3, random_state=42, max_iter=1000, n_alphas=50).fit(X_flattened, y_target)

    print(f"LassoCV selected best alpha: {lasso_cv.alpha_:.5f}")

    # Get the coefficients from the best model
    coefs = lasso_cv.coef_

    # The input is flattened, so we need to map coefficients back to original features
    # We can do this by reshaping the coefficients and summing their importance
    coefs_reshaped = coefs.reshape(seq_len, n_features)

    # Feature importance is the sum of absolute coefficients for that feature over the sequence
    feature_importance = np.sum(np.abs(coefs_reshaped), axis=0)

    # Select features whose importance is greater than a small threshold
    selected_features_mask = feature_importance > 1e-6

    selected_features = [name for name, selected in zip(feature_names, selected_features_mask) if selected]

    # Ensure we always select at least one feature
    if not selected_features:
        most_important_idx = np.argmax(feature_importance)
        selected_features = [feature_names[most_important_idx]]

    print(f"Causal features selected for this window: {selected_features}")

    return selected_features
