"""Refresh the bundled price snapshots (run locally, then commit data/): python update_data.py"""
import os, re
import yfinance as yf

with open("secondapp.py", encoding="utf-8") as f:
    symbols = re.findall(r"'(\d{4}\.SR)':", f.read())
os.makedirs("data", exist_ok=True)
for sym in dict.fromkeys(symbols):
    h = yf.Ticker(sym).history(period="5y")[["Open", "High", "Low", "Close", "Volume"]]
    h.index = h.index.tz_localize(None).normalize()
    h.round(4).to_csv(f"data/{sym}.csv", index_label="Date")
    print(sym, len(h), h.index[-1].date())
