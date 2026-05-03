import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class LinearRegressionAlgebra:
    def __init__(self):
        self.theta = None
        self.history = []

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta

    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


def main():
    ticker = "^GSPC"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    print("Fetching S&P 500 data...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    df = pd.DataFrame({
        'Close': data['Close'].squeeze(),
        'Open': data['Open'].squeeze(),
        'High': data['High'].squeeze(),
        'Low': data['Low'].squeeze(),
        'Volume': data['Volume'].squeeze()
    })
    df = df.dropna()

    df['Returns'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df = df.dropna()

    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()

    features = ['Open', 'High', 'Low', 'Volume', 'Returns', 'MA5', 'MA20', 'Volatility']
    X = df[features].values
    y = df['Target'].values

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

    model = LinearRegressionAlgebra()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("\n" + "="*50)
    print("LINEAR REGRESSION - LINEAR ALGEBRA IMPLEMENTATION")
    print("="*50)
    print("\nFormula: y = X @ theta")
    print("Normal Equation: theta = (X.T @ X)^-1 @ X.T @ y")
    print("\nModel Parameters (theta):")
    print(f"  Intercept (bias): {model.theta[0]:.4f}")
    for i, feat in enumerate(features):
        print(f"  {feat}: {model.theta[i+1]:.4f}")

    print(f"\n--- Training Set ---")
    print(f"  R2 Score: {model.r2_score(y_train, y_pred_train):.4f}")
    print(f"  MSE: {model.mse(y_train, y_pred_train):.4f}")

    print(f"\n--- Test Set ---")
    print(f"  R2 Score: {model.r2_score(y_test, y_pred_test):.4f}")
    print(f"  MSE: {model.mse(y_test, y_pred_test):.4f}")

    plt.style.use('dark_background')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0a0a0a')

    axes[0].plot(y_test, label='Actual', color='#00ff00', alpha=0.8, linewidth=1.5)
    axes[0].plot(y_pred_test, label='Predicted', color='#ff00ff', alpha=0.8, linewidth=1.5, linestyle='--')
    axes[0].set_xlabel('Time (Days)', color='#00ff00', fontsize=11)
    axes[0].set_ylabel('Price ($)', color='#00ff00', fontsize=11)
    axes[0].set_title('LINEAR ALGEBRA MODEL: Actual vs Predicted', 
                      fontsize=14, fontweight='bold', color='#00ff00', pad=15)
    axes[0].legend(loc='upper left', facecolor='#1a1a1a', edgecolor='#00ff00')
    axes[0].grid(True, alpha=0.2, color='#00ff00')
    axes[0].tick_params(colors='#00ff00')
    for spine in axes[0].spines.values():
        spine.set_edgecolor('#00ff00')
        spine.set_linewidth(1.5)

    errors = y_test - y_pred_test
    axes[1].hist(errors, bins=30, edgecolor='#00ff00', alpha=0.7, color='#003300')
    axes[1].axvline(x=0, color='#ff0000', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prediction Error', color='#00ff00', fontsize=11)
    axes[1].set_ylabel('Frequency', color='#00ff00', fontsize=11)
    axes[1].set_title('ERROR DISTRIBUTION', fontsize=14, fontweight='bold', color='#00ff00', pad=15)
    axes[1].grid(True, alpha=0.2, color='#00ff00')
    axes[1].tick_params(colors='#00ff00')
    for spine in axes[1].spines.values():
        spine.set_edgecolor('#00ff00')
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('linear_model_results.png', dpi=150, facecolor='#0a0a0a')
    plt.show()

    plt.style.use('seaborn-v0_8-whitegrid')

    print("\nPlot saved to linear_model_results.png")


if __name__ == "__main__":
    main()