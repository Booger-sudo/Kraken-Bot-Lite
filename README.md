# KrakeTrade Lite

**Multi-Strategy Kraken Trading Bot – Free for Devs, Pre-Configured for $50**

KrakeTrade Lite is a modular, Python-based trading bot for the Kraken exchange. It features multiple configurable strategies, adaptive market state detection, and risk-managed trade logic.

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
  ```

---

## Setup (Free Version for Developers)

1. **Clone this repo**  
   ```bash
   git clone https://github.com/yourusername/krakenbot-lite.git
   cd krakenbot-lite
   ```

2. **Create a `.env` file**  
   Add your API keys:
   ```
   KRAKEN_API_KEY=your_api_key
   KRAKEN_API_SECRET=your_api_secret
   ```

3. **Manually configure strategy settings**  
   Inside the bot code, all configurable values (e.g., RSI thresholds, EMA periods, risk limits) are marked with `<CONFIGURE>` comments. Edit them to match your trading style.

4. **Run the bot**  
   ```bash
   python kraken-bot-lite.py
   ```

---

## Want to Skip the Setup?

I offer a **pre-configured version** for users who want to skip manual setup and run the bot immediately.

**Pre-Configured Edition Includes:**
- A ready-to-run private version of this bot
- Pre-filled strategy settings for common trading scenarios
- Pre-structured `.env` and logging setup
- No setup, no tweaking, just launch

**Price:** $50  
**Delivery:** Private GitHub repo invite within 24 hours

**No consultation or custom tuning is included.**  
This is for users who want to get started fast with proven default settings.

**To purchase:**  
- Email: krakenbotlive@proton.me  
- Or [Ko-fi / BuyMeACoffee / Gumroad link]

---

## License

Kraken Bot Lite is open-source under the MIT License, for personal and educational use.  
**Commercial use or resale prohibited without permission.**

---

## Disclaimer

This software is provided *as-is*. No financial advice is given. Use at your own risk. Markets are volatile. Bots can and will lose money without proper setup and testing.
