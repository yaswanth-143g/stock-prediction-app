"""
quant package — Quantitative Financial Analytics, Risk Testing & Backtesting Engine
====================================================================================
Provides Monte Carlo risk simulation, multi-stock portfolio optimization,
and strategy backtesting algorithms.
"""

from quant.analytics import (
    run_monte_carlo_simulation,
    analyze_portfolio,
    run_strategy_backtest,
    optimize_portfolio_mvo,
)

__all__ = [
    "run_monte_carlo_simulation",
    "analyze_portfolio",
    "run_strategy_backtest",
    "optimize_portfolio_mvo",
]

