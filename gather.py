import time
from datetime import datetime
from typing import List, Dict, Tuple

import pandas as pd
import requests

from mpt import Asset

"""
The complete documentation for the coincap.io REST API can be found at https://docs.coincap.io/.
"""


def request_and_jsonize_calm(url, params=None):
    repeats = 0
    while True:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429:
            repeats += 1
            sleep_secs = 5 * repeats
            time.sleep(sleep_secs)
            print(f'Sleeping {sleep_secs} seconds on url: {url}')
            continue
        if repeats >= 3:
            raise Exception(f'Retried too many times: stuck on url {url}')
        response.raise_for_status()


def get_available_assets(limit: int = 100) -> List:
    """ Get the first most capitalized assets from the coincap.io API. """

    js = request_and_jsonize_calm(f"https://api.coincap.io/v2/assets", params={'limit': str(limit)})

    assets = []
    for asset in js['data']:
        assets.append(
            {
                'id': asset['id'],
                'symbol': asset['symbol'],
                'name': asset['name']
            }
        )
    return assets


def filter_by_symbol(assets: List[Dict], symbols: List) -> List:
    """ Filter the wanted assets, returning a list of assets ordered in the same was as the symbols list.
    If one asset is not available, throw an exception.
    """
    filtered_assets = []
    found = False
    for symbol in symbols:
        for asset in assets:
            if asset['symbol'].lower() == symbol.lower():
                filtered_assets.append(asset)
                found = True
                break
        if not found:
            raise Exception(f'Asset with symbol {symbol} not found in the provided list.')
    return filtered_assets


def get_series(currency_id: str, interval: str) -> pd.DataFrame:
    """ Get the time series for the given currency_id. Timestamps and dates are given in UTC time. """
    url = f"https://api.coincap.io/v2/assets/{currency_id}/history"

    js = request_and_jsonize_calm(url, params={'interval': interval})

    times, prices, dates = [], [], []
    for measurement in js['data']:
        timestamp_seconds = float(measurement['time']) // 1000
        times.append(timestamp_seconds)  # Timestamp is in milliseconds
        prices.append(float(measurement['priceUsd']))
        dates.append(datetime.fromtimestamp(timestamp_seconds))

    df = pd.DataFrame(
        {
            'date': dates,
            'time': times,
            'price': prices
        }
    )

    return df


def get_assets(symbols: List[str], search_limit: int = 100) -> Tuple:
    """ Get the dataframes of the wanted assets, specified as symbols in the symbols list. """
    # 1) Get the available assets up to the specified limit
    assets = get_available_assets(search_limit)

    # 2) Filter out the wanted assets
    try:
        filtered_assets = filter_by_symbol(assets, symbols)
    except Exception as e:
        raise e     # We may be wanting to do something about that

    # 3) For every selected asset, return its dataframe
    to_return = []
    for asset in filtered_assets:
        time.sleep(5)
        series = get_series(asset['id'], 'd1')   # With d1 as interval, we select daily prices
        to_return.append(Asset(asset['symbol'], series))
    return tuple(to_return)
