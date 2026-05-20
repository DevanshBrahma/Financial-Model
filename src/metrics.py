import numpy as np


def compute_metrics(equity_curve):
    returns = equity_curve.pct_change().dropna()

    roi = equity_curve.iloc[-1] - 1
    sharpe = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 0
    max_drawdown = ((equity_curve / equity_curve.cummax()) - 1).min()

    return {"ROI": roi, "Sharpe": sharpe, "Max_Drawdown": max_drawdown}
