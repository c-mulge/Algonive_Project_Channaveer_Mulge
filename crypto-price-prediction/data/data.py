import yfinance as yf
btc=yf.download("BTC-USD", start="2018-01-01", end="2025-01-01")
btc.to_csv("BTC-USD.csv")