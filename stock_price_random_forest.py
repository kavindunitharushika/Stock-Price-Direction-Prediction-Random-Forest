"""Stock Price Direction Prediction Using Random Forest Classification.

Adapted from Appendix B of the group research report (February 2026).
Group: Dilshan A. M. B., Farsath A. T. A., Fernando W. D. N.,
       Handagala H. K. T.

IMPORTANT: Appendix B provides feature engineering, model initialization and
prediction snippets, but not a complete data loading, target construction,
class-balancing or train/test preprocessing pipeline. This module preserves
those snippets as reusable functions; it is not a standalone reproduction of
reported results.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier


FEATURE_COLUMNS = [
    "Open_feat", "High_feat", "Low_feat", "Close_feat", "Volume_feat",
    "Volume_Percentage", "MA10", "MA30", "Daily_Return", "RSI_14",
    "Stoch_%K", "Stoch_%D", "Williams_%R", "MACD", "ROC_10", "OBV",
]


def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_stochastic(df, k_window=14, d_window=3):
    low_min = df["Low"].rolling(window=k_window).min()
    high_max = df["High"].rolling(window=k_window).max()
    stoch_k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=d_window).mean()
    return stoch_k, stoch_d


def calculate_williams_r(df, window=14):
    high_max = df["High"].rolling(window=window).max()
    low_min = df["Low"].rolling(window=window).min()
    return -100 * (high_max - df["Close"]) / (high_max - low_min)


def engineer_features(df):
    """Recreate Appendix B.1; expects chronological OHLCV rows.

    Note: The report uses index positions 3685/3684 for a share-count change.
    Those positions are retained exactly, and require checking against your
    actual dataset before use. Use a default RangeIndex for positional slicing.
    """
    df = df.copy().reset_index(drop=True)
    for source, target in [
        ("Open", "Open_feat"), ("High", "High_feat"),
        ("Low", "Low_feat"), ("Close", "Close_feat"),
        ("Volume", "Volume_feat"),
    ]:
        df[target] = df[source]

    total_volume_after = 16528995520  # Report: ordinary shares 1:10
    total_volume_before = 1652899552
    df["Volume_Percentage"] = 0.0
    df.loc[3685:, "Volume_Percentage"] = (
        df.loc[3685:, "Volume"] / total_volume_after * 100
    )
    df.loc[:3684, "Volume_Percentage"] = (
        df.loc[:3684, "Volume"] / total_volume_before * 100
    )

    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA30"] = df["Close"].rolling(window=30).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["RSI_14"] = calculate_rsi(df["Close"], 14)
    df["Stoch_%K"], df["Stoch_%D"] = calculate_stochastic(df)
    df["Williams_%R"] = calculate_williams_r(df, 14)
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["ROC_10"] = (
        (df["Close"] - df["Close"].shift(10))
        / df["Close"].shift(10) * 100
    )
    df["OBV"] = (
        np.sign(df["Close"].diff()) * df["Volume"]
    ).fillna(0).cumsum()
    return df


def build_model():
    """Appendix B.2 model hyperparameters."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


def train_model(X_train, y_train):
    """X_train/y_train must be prepared separately (not supplied in appendix)."""
    rf_model = build_model()
    rf_model.fit(X_train, y_train)
    return rf_model


def predict_next_day(rf_model, X):
    """Appendix B.3: use last prepared feature row to generate a signal."""
    latest_data = X.iloc[-1:].copy()
    next_prediction = rf_model.predict(latest_data)[0]
    next_proba = rf_model.predict_proba(latest_data)[0]
    confidence = max(next_proba)
    if confidence > 0.60:
        signal = "STRONG"
    elif confidence > 0.55:
        signal = "MODERATE"
    else:
        signal = "WEAK"
    print(f"Direction: {'UP' if next_prediction == 1 else 'DOWN'}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Signal Strength: {signal}")
    return next_prediction, confidence, signal


if __name__ == "__main__":
    print("Appendix B code extracted into reusable functions.")
    print("Provide your own OHLCV DataFrame and prepared training data to run.")
    print("Feature columns:", ", ".join(FEATURE_COLUMNS))
