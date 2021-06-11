import pandas as pd
import numpy as np
from typing import Tuple
from functools import lru_cache
from scipy.optimize import minimize

TRADING_DAYS_PER_YEAR = 356


def get_log_returns_over_period(price_history: pd.DataFrame):
    """
    Given the price time series, compute the logarithm of the ration between final and initial price.
    """
    prices = price_history['price'].values
    return np.log(prices[1:] / prices[:-1]).reshape(-1, 1)


def random_portfolio_weights(weights_count):
    """ Random portfolio weights, of length weights_count. """
    weights = np.random.random((weights_count, 1))
    weights /= np.sum(weights)
    return weights.reshape(-1, 1)


class Asset:
    """ A single asset. """
    def __init__(
            self,
            name: str,
            daily_prices: pd.DataFrame
            ):
        self.name = name
        self.daily_log_returns = get_log_returns_over_period(daily_prices)

    @staticmethod
    @lru_cache
    def covariance_matrix(assets: Tuple):
        product_expectation = np.zeros((len(assets), len(assets)))
        for i in range(len(assets)):
            for j in range(len(assets)):
                if i == j:                 # variance
                    product_expectation[i][j] = np.mean(assets[i].daily_log_returns * assets[j].daily_log_returns)
                else:                # covariance
                    product_expectation[i][j] = np.mean(assets[i].daily_log_returns @ assets[j].daily_log_returns.T)

        product_expectation *= (TRADING_DAYS_PER_YEAR - 1) ** 2

        expected_returns = np.array([asset.expected_log_return for asset in assets]).reshape(-1, 1)
        product_of_expectations = expected_returns @ expected_returns.T

        return product_expectation - product_of_expectations

    @property
    def _expected_daily_log_return(self):
        return np.mean(self.daily_log_returns)

    @property
    def expected_log_return(self):
        return TRADING_DAYS_PER_YEAR * self._expected_daily_log_return

    @property
    def expected_return(self):
        return np.exp(self.expected_log_return)

    @property
    def variance(self):
        return np.std(self.expected_log_return)**2

    def __repr__(self):
        return f'<Asset name={self.name}, expected log return={self.expected_log_return}, variance={self.variance}>'


class Portfolio:
    """ Portfolio, containing a combination of assets determined by the weights.
    """
    def __init__(self, name: str, assets: Tuple[Asset]):
        self.name = name
        self.assets = assets
        self.asset_expected_returns = np.array([asset.expected_log_return for asset in assets]).reshape(-1, 1)
        self.covariance_matrix = Asset.covariance_matrix(assets)
        self._weights = random_portfolio_weights(len(assets))

    def optimize_with_risk_tolerance(self, risk_tolerance: float):
        assert risk_tolerance >= 0.
        optim_res = minimize(
            lambda w: self._variance(w) - risk_tolerance * self._expected_log_return(w),
            self._weights,
            constraints=[
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
            ],
            bounds=[(0., 1.) for _ in range(self._weights.size)]
        )

        assert optim_res.success, f'Optimization with RT failed: {optim_res.message}'
        self._weights = optim_res.x.reshape(-1, 1)

    def optimize_with_expected_return(self, expected_log_portfolio_return: float):
        optim_res = minimize(
            lambda w: self._variance(w),
            self._weights,
            constraints=[
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
                {'type': 'eq', 'fun': lambda w: self._expected_log_return(w) - expected_log_portfolio_return},
            ],
            bounds=[(0., 1.) for _ in range(self._weights.size)]
        )

        assert optim_res.success, f'Optimization with ER failed: {optim_res.message}'
        self._weights = optim_res.x.reshape(-1, 1)

    def optimize_sharpe_ratio(self, riskless_return: float):
        # Maximize Sharpe ratio: actually, here we are maximizing the logarithms of the quantities at the numerator in
        # order to fairly consider them. Nevertheless, log(.) is monotonic so the optimization works well although
        # we DO NOT obtain the actual Sharpe ratio as a result.
        optim_res = minimize(
            lambda w: -(self._expected_log_return(w) - np.log(1 + riskless_return / 100)) / np.sqrt(self._variance(w)),
            self._weights,
            constraints=[
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
            ],
            bounds=[(0., 1.) for _ in range(self._weights.size)]
        )

        assert optim_res.success, f'Optimization with Sharpe ratio failed: {optim_res.message}'
        self._weights = optim_res.x.reshape(-1, 1)

    def _expected_log_return(self, w):
        return (self.asset_expected_returns.T @ w.reshape(-1, 1))[0][0]

    def _variance(self, w):
        return (w.reshape(-1, 1).T @ self.covariance_matrix @ w.reshape(-1, 1))[0][0]

    @property
    def weights(self):
        return self._weights.flatten()

    @property
    def assets_names(self):
        return tuple([a.name for a in self.assets])

    @property
    def expected_log_return(self):
        return self._expected_log_return(self._weights)

    @property
    def expected_return(self):
        return np.exp(self.expected_log_return)

    @property
    def variance(self):
        return self._variance(self._weights)

    def __repr__(self):
        return f'<Portfolio assets={[asset.name for asset in self.assets]}, expected log return={self.expected_log_return}, variance={self.variance}>'