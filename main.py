import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gather import get_assets
from mpt import Asset, Portfolio
from typing import List, Tuple

np.random.seed(7)  # For reproducibility

TREASURY_BILL_RATE = 0.05
ASSETS_SYMBOLS = ['ETH', 'ADA', 'MATIC', 'USDT', 'LTC', 'BTC', 'XMR']

# 1) Get the wanted assets
assets = get_assets(ASSETS_SYMBOLS)


def random_portfolios() -> (List, List):
    """ Generate points corresponding to random portfolios. """
    X, y = [], []
    for _ in range(3000):
        portfolio = Portfolio('.', assets)  # Every time a new portfolio is created, it has random weights
        X.append(portfolio.variance)
        y.append(portfolio.expected_log_return)
    return X, y


def efficient_frontier() -> (List, List):
    """ Generate points on the efficient frontier. """
    X, y = [], []
    portfolio = Portfolio('.', assets)
    for rt in np.linspace(0, 200, 1000):    # Vary risk tolerance levels
        try:
            portfolio.optimize_with_risk_tolerance(rt)
            X.append(portfolio.variance)
            y.append(portfolio.expected_log_return)
        except:
            pass     # If the optimization fails, let's just get over it for now
    return X, y


def build_results_dataframe(assets: Tuple[Asset], portfolios: List[Portfolio]) -> pd.DataFrame:
    """ Build a dataframe of optimal portfolios. """
    # Check that we do not have portfolios with different assets
    assert all(p.assets_names == tuple(a.name for a in assets) for p in portfolios)

    df = pd.DataFrame(
        {p.name: p.weights for p in portfolios}
    )
    df.index = [a.name for a in assets]
    return df.T     # For simplicity we created a transposed frame, now we need to transpose back


def plot_portfolio(portfolio: Portfolio, color: str, label: str, alpha: float=1.0) -> None:
    """ Plots a single point representing a portfolio, using Seaborn. """
    assert 0. <= alpha <= 1.
    sns.scatterplot(
        x=[portfolio.variance],
        y=[portfolio.expected_log_return],
        s=160,
        marker='*',
        facecolor=color,
        alpha=alpha,
        label=label,
        zorder=10
    )


# 2) Build portfolios, optimize and plot (variance vs. expected return)
sns.set_style('darkgrid')

# 2a) Random portfolios
X_random, y_random = random_portfolios()
sns.scatterplot(x=X_random, y=y_random, label='Random portfolio')

# 2b) Draw the efficient frontier
X_efficient, y_efficient = efficient_frontier()
sns.lineplot(x=X_efficient, y=y_efficient, color='black', linewidth=2.5, label='Efficient frontier')

# 2c) Compute specific optimal portfolios
portfolios_to_export = []

# Risk tolerance: 0 (riskless portfolio)
riskless_portfolio = Portfolio('riskless', assets)
riskless_portfolio.optimize_with_risk_tolerance(0)
plot_portfolio(riskless_portfolio, 'red', 'riskless', alpha=0.7)
portfolios_to_export.append(riskless_portfolio)

# Risk tolerance: 30
t1_portfolio = Portfolio('rt=30', assets)
t1_portfolio.optimize_with_risk_tolerance(30)
plot_portfolio(t1_portfolio, 'red', 'rt=30', alpha=0.85)
portfolios_to_export.append(t1_portfolio)

# Risk tolerance: 50
t2_portfolio = Portfolio('rt=50', assets)
t2_portfolio.optimize_with_risk_tolerance(50)
plot_portfolio(t2_portfolio, 'red', 'rt=50')
portfolios_to_export.append(t2_portfolio)

# Expected return: 1 (final price 2.7 times the initial one!)
e1_portfolio = Portfolio('er=1', assets)
e1_portfolio.optimize_with_expected_return(1)
plot_portfolio(e1_portfolio, 'green', 'er=1', alpha=0.85)
portfolios_to_export.append(e1_portfolio)

# Expected return: 1.8 (final price 6 times the initial one!)
e2_portfolio = Portfolio('er=1.8', assets)
e2_portfolio.optimize_with_expected_return(1.8)
plot_portfolio(e2_portfolio, 'green', 'er=1.8')
portfolios_to_export.append(e2_portfolio)

# Optimal Sharpe ratio
s_portfolio = Portfolio('optimal S', assets)
s_portfolio.optimize_sharpe_ratio(TREASURY_BILL_RATE)
plot_portfolio(s_portfolio, 'yellow', 'optimal S')
portfolios_to_export.append(s_portfolio)

# 3) Build results dataframe
df = build_results_dataframe(assets,
                             portfolios_to_export
                             )
pd.options.display.float_format = '{:,.2f}%'.format
print(df)

# 4) Plot
plt.xlabel('Portfolio variance')
plt.ylabel('Portfolio expected (logarithmic) return')
plt.legend(loc='lower right')
plt.show()

# Heatmap
sns.heatmap(df, annot=True, fmt='.2%', cmap="Oranges")
plt.show()
