# Import required libraries
import os
import time
import base64
import hashlib
import hmac
import json
import logging
import urllib.request
from urllib.parse import urlencode
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from dotenv import load_dotenv

# Constants
API_URL = "https://api.kraken.com"
API_VERSION = "0"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
REQUEST_DELAY = 1

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kraken_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KrakenAPIError(Exception):
    pass

class KrakenAuth:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret

    def generate_signature(self, urlpath: str, data: dict) -> str:
        postdata = urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

class MarketData:
    def __init__(self, auth: KrakenAuth):
        self.auth = auth

    def get_ohlc(self, pair: str, interval: int = 1440) -> pd.DataFrame:
        logger.info(f"Fetching OHLC data for pair: {pair}, interval: {interval}")
        try:
            response = requests.get(f"{API_URL}/0/public/OHLC", params={"pair": pair, "interval": interval})
            response.raise_for_status()
            data = response.json()

            if 'result' not in data or pair not in data['result']:
                raise ValueError(f"Invalid response or pair not found in OHLC data: {data}")

            ohlc = pd.DataFrame(data['result'][pair], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
            ])
            ohlc = ohlc.astype(float)
            ohlc['timestamp'] = pd.to_datetime(ohlc['timestamp'], unit='s')

            # Ensure there is enough data for analysis
            if len(ohlc) < 2:
                raise ValueError(f"Not enough OHLC data for pair: {pair}")

            return ohlc.set_index('timestamp')
        except Exception as e:
            logger.error(f"Failed to get OHLC data: {str(e)}")
            raise

    def calculate_atr(self, pair: str, interval: int = 1440, period: int = 14) -> float:
        ohlc = self.get_ohlc(pair, interval)
        high_low = ohlc['high'] - ohlc['low']
        high_close = (ohlc['high'] - ohlc['close'].shift()).abs()
        low_close = (ohlc['low'] - ohlc['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean().iloc[-1]

    def _make_request(self, endpoint: str, data: dict):
        logger.debug(f"Making API request to {endpoint} with data: {data}")
        ...
        logger.debug(f"API response: {response}")

class TradingStrategy:
    def __init__(self, market_data: MarketData):
        self.market_data = market_data

    def analyze(self, pair: str) -> dict:
        raise NotImplementedError

class EMACrossoverStrategy(TradingStrategy):
    def __init__(self, market_data: MarketData):
        super().__init__(market_data)
        # self.fast_period = <CONFIGURE>
        # self.slow_period = <CONFIGURE>
        # self.higher_fast_period = <CONFIGURE>  # 4-hour fast EMA
        # self.higher_slow_period = <CONFIGURE>  # 4-hour slow EMA

    def calculate_higher_ema_crossover(self, pair: str) -> bool:
        """
        Calculate 20×50 EMA crossover for the 4-hour timeframe.
        Returns True if there's a crossover, False otherwise.
        """
        ohlc = self.market_data.get_ohlc(pair, interval=240)  # 240 minutes = 4 hours
        if len(ohlc) < max(self.higher_fast_period, self.higher_slow_period):
            logger.warning(f"Not enough data for 4-hour EMA analysis for pair: {pair}")
            return False

        ohlc['fast_ema'] = ohlc['close'].ewm(span=self.higher_fast_period, adjust=False).mean()
        ohlc['slow_ema'] = ohlc['close'].ewm(span=self.higher_slow_period, adjust=False).mean()

        last_row = ohlc.iloc[-1]
        prev_row = ohlc.iloc[-2]

        # Check for 20×50 crossover conditions
        if prev_row['fast_ema'] <= prev_row['slow_ema'] and last_row['fast_ema'] > last_row['slow_ema']:
            return True  # Bullish crossover
        elif prev_row['fast_ema'] >= prev_row['slow_ema'] and last_row['fast_ema'] < last_row['slow_ema']:
            return True  # Bearish crossover

        return False

    def analyze(self, pair: str) -> dict:
        ohlc = self.market_data.get_ohlc(pair)
        if len(ohlc) < max(self.fast_period, self.slow_period):
            logger.warning(f"Not enough data for EMA analysis for pair: {pair}")
            return {'signal': 'hold', 'confidence': 0}

        ohlc['fast_ema'] = ohlc['close'].ewm(span=self.fast_period, adjust=False).mean()
        ohlc['slow_ema'] = ohlc['close'].ewm(span=self.slow_period, adjust=False).mean()

        last_row = ohlc.iloc[-1]
        prev_row = ohlc.iloc[-2]

        # Check 9×21 crossover (main strategy condition)
        if (prev_row['fast_ema'] <= prev_row['slow_ema'] and last_row['fast_ema'] > last_row['slow_ema']):
            # Confirm with 20×50 EMA on 4-hour timeframe
            if self.calculate_higher_ema_crossover(pair):
                return {'signal': 'buy', 'price': float(last_row['close']), 'confidence': 0.8}
        elif (prev_row['fast_ema'] >= prev_row['slow_ema'] and last_row['fast_ema'] < last_row['slow_ema']):
            # Confirm with 20×50 EMA on 4-hour timeframe
            if self.calculate_higher_ema_crossover(pair):
                return {'signal': 'sell', 'price': float(last_row['close']), 'confidence': 0.8}

        return {'signal': 'hold', 'confidence': 0}

class MACDStrategy(TradingStrategy):
    def __init__(self, market_data: MarketData):
        super().__init__(market_data)
        # self.fast_period = <CONFIGURE>
        # self.slow_period = <CONFIGURE>
        self.signal_period = 9

    def analyze(self, pair: str) -> dict:
        ohlc = self.market_data.get_ohlc(pair)
        fast_ema = ohlc['close'].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = ohlc['close'].ewm(span=self.slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        last_macd = macd_line.iloc[-1]
        last_signal = signal_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]

        if (prev_macd <= prev_signal and last_macd > last_signal):
            return {'signal': 'buy', 'price': float(ohlc['close'].iloc[-1]), 'confidence': 0.7}
        elif (prev_macd >= prev_signal and last_macd < last_signal):
            return {'signal': 'sell', 'price': float(ohlc['close'].iloc[-1]), 'confidence': 0.7}
        return {'signal': 'hold', 'confidence': 0}

class RSIStrategy(TradingStrategy):
    def __init__(self, market_data: MarketData):
        super().__init__(market_data)
        # self.period = <CONFIGURE>
        # self.overbought = <CONFIGURE>
        # self.oversold = <CONFIGURE>

    def analyze(self, pair: str) -> dict:
        ohlc = self.market_data.get_ohlc(pair)
        delta = ohlc['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        last_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        if (prev_rsi <= self.oversold and last_rsi > self.oversold):
            return {'signal': 'buy', 'price': float(ohlc['close'].iloc[-1]), 'confidence': 0.6}
        elif (prev_rsi >= self.overbought and last_rsi < self.overbought):
            return {'signal': 'sell', 'price': float(ohlc['close'].iloc[-1]), 'confidence': 0.6}
        return {'signal': 'hold', 'confidence': 0}

class FibonacciStrategy(TradingStrategy):
    def __init__(self, market_data: MarketData):
        super().__init__(market_data)
        # self.levels = <CONFIGURE>
        # self.lookback = <CONFIGURE>

    def analyze(self, pair: str) -> dict:
        ohlc = self.market_data.get_ohlc(pair, interval=60)
        highs = ohlc['high'].rolling(window=5, center=True).max()
        lows = ohlc['low'].rolling(window=5, center=True).min()
        swing_high = highs.max()
        swing_low = lows.min()
        current_price = float(ohlc['close'].iloc[-1])

        fib_levels = {level: swing_high - (level * (swing_high - swing_low)) for level in self.levels}

        nearest_level = None
        min_dist = float('inf')
        for level, price in fib_levels.items():
            dist = abs(current_price - price)
            if dist < min_dist:
                min_dist = dist
                nearest_level = level

        if current_price >= swing_high * 0.99:
            return {'signal': 'sell', 'price': current_price, 'confidence': 0.6}
        elif current_price <= swing_low * 1.01:
            return {'signal': 'buy', 'price': current_price, 'confidence': 0.6}
        elif nearest_level and min_dist < 0.008 * current_price:
            if nearest_level >= 0.618 and ohlc['close'].iloc[-1] > ohlc['close'].iloc[-2]:
                return {'signal': 'buy', 'price': current_price, 'confidence': 0.6}
            elif nearest_level <= 0.382 and ohlc['close'].iloc[-1] < ohlc['close'].iloc[-2]:
                return {'signal': 'sell', 'price': current_price, 'confidence': 0.6}
        return {'signal': 'hold', 'confidence': 0}

class DailyVWAPStrategy(TradingStrategy):
    def __init__(self, market_data: MarketData):
        super().__init__(market_data)
        # self.std_dev_multiplier = <CONFIGURE>
        # self.confirmation_periods = <CONFIGURE>
        # self.min_volume_factor = <CONFIGURE>
        # self.lower_timeframe_interval = <CONFIGURE>  # 4-hour confirmation timeframe

    def calculate_vwap(self, ohlc_data):
        """Calculate VWAP and standard deviation bands"""
        typical_price = (ohlc_data['high'] + ohlc_data['low'] + ohlc_data['close']) / 3
        cumulative_volume = ohlc_data['volume'].cumsum()
        cumulative_pv = (typical_price * ohlc_data['volume']).cumsum()
        vwap = cumulative_pv / cumulative_volume

        squared_diff = ((typical_price - vwap) ** 2 * ohlc_data['volume']).cumsum()
        std_dev = (squared_diff / cumulative_volume) ** 0.5
        upper_band = vwap + std_dev * self.std_dev_multiplier
        lower_band = vwap - std_dev * self.std_dev_multiplier

        return vwap, upper_band, lower_band

    def analyze_vwap_trend(self, ohlc_data):
        """Determine the VWAP trend direction"""
        vwap, _, _ = self.calculate_vwap(ohlc_data)

        if len(vwap) < self.confirmation_periods:
            return 'neutral'

        recent_vwap = vwap.iloc[-self.confirmation_periods:]  # Use .iloc for slicing
        if all(recent_vwap.iloc[i] > recent_vwap.iloc[i-1] for i in range(1, len(recent_vwap))):
            return 'up'
        elif all(recent_vwap.iloc[i] < recent_vwap.iloc[i-1] for i in range(1, len(recent_vwap))):
            return 'down'
        return 'neutral'

    def analyze_lower_timeframe_vwap(self, pair: str):
        """Check VWAP alignment on 4-hour timeframe"""
        ohlc = self.market_data.get_ohlc(pair, interval=self.lower_timeframe_interval)
        if len(ohlc) < 20:
            return 'neutral'
        return self.analyze_vwap_trend(ohlc)

    def analyze(self, pair: str) -> dict:
        # Daily timeframe analysis (primary)
        daily_ohlc = self.market_data.get_ohlc(pair)
        if len(daily_ohlc) < 20:
            logger.warning(f"Not enough daily data for pair: {pair}")
            return {'signal': 'hold', 'confidence': 0}

        daily_vwap, daily_upper, daily_lower = self.calculate_vwap(daily_ohlc)
        current_price = daily_ohlc['close'].iloc[-1]
        current_volume = daily_ohlc['volume'].iloc[-1]
        avg_volume = daily_ohlc['volume'].rolling(20).mean().iloc[-1]

        # Get trends
        daily_trend = self.analyze_vwap_trend(daily_ohlc)
        four_hour_trend = self.analyze_lower_timeframe_vwap(pair)

        # Bullish entry conditions
        if (current_price > daily_vwap.iloc[-1] and 
            daily_trend == 'up' and
            four_hour_trend in ['up', 'neutral'] and  # Allow neutral on 4H
            current_volume > avg_volume * self.min_volume_factor):

            if any(daily_ohlc['low'].iloc[-5:-1] <= daily_vwap.iloc[-5:-1]):
                return {
                    'signal': 'buy',
                    'price': float(current_price),
                    'confidence': 0.85,
                    'daily_vwap': float(daily_vwap.iloc[-1]),
                    '4h_trend': four_hour_trend,
                    'bands': {
                        'upper': float(daily_upper.iloc[-1]),
                        'lower': float(daily_lower.iloc[-1])
                    }
                }

        # Bearish entry conditions
        elif (current_price < daily_vwap.iloc[-1] and 
              daily_trend == 'down' and
              four_hour_trend in ['down', 'neutral'] and
              current_volume > avg_volume * self.min_volume_factor):

            if any(daily_ohlc['high'].iloc[-5:-1] >= daily_vwap.iloc[-5:-1]):
                return {
                    'signal': 'sell',
                    'price': float(current_price),
                    'confidence': 0.85,
                    'daily_vwap': float(daily_vwap.iloc[-1]),
                    '4h_trend': four_hour_trend,
                    'bands': {
                        'upper': float(daily_upper.iloc[-1]),
                        'lower': float(daily_lower.iloc[-1])
                    }
                }

        return {'signal': 'hold', 'confidence': 0}

class CompositeStrategy(TradingStrategy):
    def __init__(self, market_data: MarketData):
        super().__init__(market_data)
        self.strategies = {
            'vwap': DailyVWAPStrategy(market_data),  # Replace EMA with VWAP
            'macd': MACDStrategy(market_data),
            'rsi': RSIStrategy(market_data),
            'fib': FibonacciStrategy(market_data)
        }
        # self.required_confirmations = <CONFIGURE>

    def analyze(self, pair: str) -> dict:
        logger.info(f"Analyzing trading signals for pair: {pair}")
        signals = []
        results = {}

        for name, strategy in self.strategies.items():
            result = strategy.analyze(pair)
            logger.info(f"Strategy {name} signal: {result['signal']} with confidence {result['confidence']}")
            signals.append(result['signal'])
            results[name] = result

        buy_signals = signals.count('buy')
        sell_signals = signals.count('sell')

        if buy_signals >= self.required_confirmations:
            logger.info(f"Composite signal: buy with {buy_signals} confirmations")
            return {'signal': 'buy', 'price': results['ema']['price'], 'confidence': buy_signals / 5}
        elif sell_signals >= self.required_confirmations:
            logger.info(f"Composite signal: sell with {sell_signals} confirmations")
            return {'signal': 'sell', 'price': results['ema']['price'], 'confidence': sell_signals / 5}
        logger.info("Composite signal: hold")
        return {'signal': 'hold', 'confidence': 0}

class AdaptiveStrategy:
    def __init__(self, market_data: MarketData, base_risk=1.0):
        self.market_data = market_data
        self.market_state = None  # "trending", "ranging", "volatile"
        self.base_risk = base_risk
        self.current_direction = None  # "long" or "short"

    def detect_market_state(self, pair: str):
        """
        Detect the current market state based on ADX and ATR ratio.
        """
        ohlc = self.market_data.get_ohlc(pair, interval=1440)  # Daily candles
        adx = self.calculate_adx(ohlc)
        current_atr = self.market_data.calculate_atr(pair)
        weekly_avg_atr = self.market_data.calculate_atr(pair, interval=10080)  # Weekly ATR

        atr_ratio = current_atr / weekly_avg_atr

        if adx > 25:
            self.market_state = "trending"
        elif atr_ratio > 2.5:
            self.market_state = "volatile"
        else:
            self.market_state = "ranging"

    def calculate_adx(self, ohlc: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate the Average Directional Index (ADX) to determine trend strength.
        """
        high = ohlc['high']
        low = ohlc['low']
        close = ohlc['close']

        plus_dm = high.diff()
        minus_dm = low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()

        return adx.iloc[-1]

    def get_strategy_weights(self):
        """
        Return strategy weights based on the detected market state.
        """
        return {
            'trending': {'vwap': 0.6, 'macd': 0.7, 'rsi': 0.3, 'fib': 0.4},
            'ranging': {'vwap': 0.3, 'macd': 0.4, 'rsi': 0.8, 'fib': 0.6},
            'volatile': {'vwap': 0.1, 'macd': 0.2, 'rsi': 0.1, 'fib': 0.1}
        }[self.market_state]

    def get_risk_factor(self, confidence=0.5):
        """
        Adjust the risk factor based on market state and confidence level.
        """
        if self.market_state == "volatile":
            return self.base_risk * 0.5
        elif confidence > 0.8:
            return self.base_risk * 1.2
        else:
            return self.base_risk

    def check_market_conditions(self, pair: str) -> bool:
        """
        Check market conditions such as liquidity and volatility.
        """
        daily_ohlc = self.market_data.get_ohlc(pair, interval=1440)  # Daily candles
        daily_ema_50 = daily_ohlc['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        daily_trend = daily_ohlc['close'].iloc[-1] > daily_ema_50
        self.current_direction = "long" if daily_trend else "short"

        # Check liquidity conditions
        spread = daily_ohlc['high'].iloc[-1] - daily_ohlc['low'].iloc[-1]
        atr = self.market_data.calculate_atr(pair)
        if spread > 0.1 * atr:
            logger.warning("Low liquidity - skipping trade")
            return False

        return True

    def execute(self, pair: str, confidence: float):
        """
        Execute trades based on the detected market state and direction.
        """
        if not self.check_market_conditions(pair):
            return

        # Get position size based on risk factors
        risk_factor = self.get_risk_factor(confidence)
        position_size = self.base_risk * risk_factor

        # Adjust position size for high volatility
        if self.market_state == "volatile":
            position_size *= 0.3

        # Execute trade based on market state and direction
        weights = self.get_strategy_weights()
        if self.current_direction == "long":
            logger.info(f"Executing long trade on {pair} with weights: {weights}")
            # Call your trading logic here
        else:
            logger.info(f"Executing short trade on {pair} with weights: {weights}")
            # Call your trading logic here

class RiskManager:
    def __init__(self, market_data: MarketData, max_risk_per_trade: float = 0.01):
        self.market_data = market_data
        self.max_risk_per_trade = max_risk_per_trade

    def calculate_position_size(self, pair: str, risk_pct: float) -> float:
        atr = self.market_data.calculate_atr(pair)
        current_price = float(self.market_data.get_ohlc(pair)['close'].iloc[-1])
        return (risk_pct * current_price) / (atr * 2)

class SmartExitStrategy:
    """Advanced exit strategy combining multiple confirmation signals"""

    def __init__(self, market_data: MarketData):
        self.market_data = market_data
        # self.trailing_stop_activation_pct = <CONFIGURE>  # Activate trailing stop after 0.5% profit
        # self.trailing_stop_distance_pct = <CONFIGURE>    # 0.3% trailing distance
        # self.profit_targets = <CONFIGURE>    # Take profit at 0.5%, 1%, and 2%
        # self.emergency_exit_signals = <CONFIGURE>

    def should_exit(self, pair: str, entry_price: float, entry_time: datetime) -> Tuple[bool, Optional[str]]:
        """
        Returns (should_exit, reason)
        Reasons can be: 'profit_target', 'trailing_stop', 'rsi_limit', 
                       'volume_spike', 'time_decay', 'fib_resistance'
        """
        ohlc = self.market_data.get_ohlc(pair, interval=5)  # 5-min candles for exit decisions
        current_price = float(ohlc['close'].iloc[-1])
        price_change_pct = (current_price - entry_price) / entry_price * 100

        # 1. Check profit targets (scaled exits)
        for i, target in enumerate(self.profit_targets):
            if price_change_pct >= target:
                return True, f'profit_target_{i+1}'

        # 2. Trailing stop logic
        if price_change_pct > self.trailing_stop_activation_pct:
            highest_price = ohlc['high'].max()
            if current_price <= highest_price * (1 - self.trailing_stop_distance_pct / 100):
                return True, 'trailing_stop'

        # 3. Emergency exit signals
        # RSI check
        delta = ohlc['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        if rsi.iloc[-1] > self.emergency_exit_signals['rsi_limit']:
            return True, 'rsi_limit'

        # Volume spike check
        avg_volume = ohlc['volume'].rolling(20).mean().iloc[-1]
        if ohlc['volume'].iloc[-1] > avg_volume * self.emergency_exit_signals['volume_spike']:
            return True, 'volume_spike'

        # Time decay check
        if (datetime.now() - entry_time).total_seconds() > self.emergency_exit_signals['time_decay'] * 3600:
            return True, 'time_decay'

        # 4. Fibonacci resistance check
        fib = FibonacciStrategy(self.market_data)
        swing_high, _ = fib.identify_swing_points(ohlc)
        fib_levels = fib.calculate_fib_levels(swing_high, ohlc['low'].min())
        nearest_level = min(fib_levels.values(), key=lambda x: abs(x - current_price))
        if current_price >= nearest_level and current_price < ohlc['high'].iloc[-1]:
            return True, 'fib_resistance'

        return False, None

class TradingBot:
    def __init__(self, api_key: str, api_secret: str, trading_pairs: List[str]):
        self.auth = KrakenAuth(api_key, api_secret)
        self.market_data = MarketData(self.auth)
        self.adaptive_strategy = AdaptiveStrategy(self.market_data)
        self.strategy = CompositeStrategy(self.market_data)
        self.risk_manager = RiskManager(self.market_data)
        self.exit_strategy = SmartExitStrategy(self.market_data)
        self.active_trades = {}  # {pair: (entry_price, entry_time, position_size)}
        self.trading_pairs = trading_pairs  # List of trading pairs to monitor and trade

    def run(self, interval: int = 15, risk_pct: float = 0.01):
        logger.info("Starting trading bot with adaptive strategy...")

        try:
            while True:
                for pair in self.trading_pairs:
                    try:
                        self.adaptive_strategy.detect_market_state(pair)
                        market_state = self.adaptive_strategy.market_state
                        logger.info(f"Market state for {pair}: {market_state}")

                        analysis = self.strategy.analyze(pair)
                        if analysis['signal'] != 'hold' and analysis['confidence'] >= 0.5:
                            self.adaptive_strategy.execute(pair, analysis['confidence'])

                    except Exception as e:
                        logger.error(f"Error processing pair {pair}: {str(e)}")
                        continue  # Skip to the next pair

                time.sleep(interval * 60)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Critical error: {str(e)}")

def get_valid_pairs():
    response = requests.get(f"{API_URL}/0/public/AssetPairs")
    data = response.json()
    valid_pairs = []

    if 'result' in data:
        for pair, details in data['result'].items():
            if 'USD' in details['altname']:  # Filter for USD pairs
                valid_pairs.append(pair)
    return valid_pairs

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    api_key = os.getenv('KRAKEN_API_KEY')
    api_secret = os.getenv('KRAKEN_API_SECRET')

    if not api_key or not api_secret:
        raise ValueError("API keys not set in environment variables")

    # Use the dynamically fetched pairs
    trading_pairs = get_valid_pairs()

    bot = TradingBot(api_key, api_secret, trading_pairs)
    bot.run()

if __name__ == "__main__":
    main()