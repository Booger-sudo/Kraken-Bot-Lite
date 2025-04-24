# Kraken-Bot-Lite

**Multi-Strategy Kraken Trading Bot – Free for Devs, Pre-Configured for $50**

KrakenBot Lite is a modular, Python-based trading bot for the Kraken exchange. It features multiple configurable strategies, adaptive market state detection, and risk-managed trade logic.

This version is **fully open-source**, but requires user configuration before it can be run effectively. If you're a developer, it’s free to explore, extend, and use.

If you'd rather skip the setup and get started right away, a **pre-configured edition** is available for a one-time payment of $50 (details below).

---

## Features

- **Built-In Strategies**: VWAP, EMA, MACD, RSI, Fibonacci, ATR
- **Composite Signal Logic**: Confirms trades with multiple strategies
- **Market State Awareness**: Detects trend, range, or volatility conditions
- **Smart Exits**: Profit targets, trailing stops, time/risk exits
- **Risk Controls**: ATR-based position sizing and liquidity checks
- **Logging**: Monitors trades, signals, and market conditions

---

## Requirements

- Python 3.8+
- [Kraken API credentials](https://support.kraken.com/hc/en-us/articles/360022839451-How-to-generate-an-API-key)
- Dependencies:
  ```bash
  pip install -r requirements.txt

git clone https://github.com/yourusername/krakenbot-lite.git
cd krakenbot-lite

KRAKEN_API_KEY=your_api_key
KRAKEN_API_SECRET=your_api_secret

python krakenbot_lite.py
