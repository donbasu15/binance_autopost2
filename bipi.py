
"""
Binance Square Auto-Poster — Signal-Driven Edition
====================================================
Architecture:
  1. DATA PIPELINE    — Binance klines, CoinGecko, CryptoPanic, Fear&Greed, OI, funding
  2. TA ENGINE        — RSI, MACD, Bollinger Bands, Support/Resistance (computed locally)
  3. SIGNAL SELECTOR  — picks post type based on what data actually shows
  4. GEMINI WRITER    — writes post grounded in real computed numbers
  5. CHART GENERATOR  — price chart + RSI subplot + S/R lines (matplotlib)
  6. PUBLISHER        — uploads chart → posts to Binance Square with image

Requirements:
    pip install google-genai requests python-dotenv numpy pandas matplotlib pillow

.env file:
    GEMINI_API_KEY=your_key
    BINANCE_SQUARE_KEY=your_binance_square_openapi_key
    CRYPTOPANIC_KEY=your_free_key     # optional but recommended
"""

import os, io, time, random, logging, math, json, threading, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import requests
from datetime import datetime, timedelta
from google import genai
from google.genai import types
from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# STATE & LOCKS
# ─────────────────────────────────────────────────────────────────────────────
state_lock = threading.Lock()
bot_state = {
    "status_message": "Initializing...",
    "start_time_utc": None,
    "posts_published": 0,
    "posts_failed": 0,
    "schedule": [],      # [{"time": "ISO", "status": "Pending", "coin": "Pending", "type": "Pending"}]
    "recent_posts": [],  # [{"time": "ISO", "content": "...", "status": "...", "url": "..."}]
    "recent_coins": [],
    "recent_types": [],
    "logs": [],
    "is_running": True,
    "error_message": None,
}

NEWS_STATE_FILE = "/home/don-basumatary/Desktop/BiPass2/.news_state.json"

def load_news_state() -> tuple[datetime | None, list[str]]:
    try:
        if os.path.exists(NEWS_STATE_FILE):
            with open(NEWS_STATE_FILE, "r") as f:
                data = json.load(f)
                val_time = data.get("last_posted_news_time")
                dt = datetime.fromisoformat(val_time) if val_time else None
                titles = data.get("posted_titles", [])
                return dt, titles
    except Exception as e:
        print("Failed to load news state:", e)
    return None, []

def save_news_state(dt: datetime | None, titles: list[str]):
    try:
        with open(NEWS_STATE_FILE, "w") as f:
            json.dump({
                "last_posted_news_time": dt.isoformat() if dt else None,
                "posted_titles": titles
            }, f)
    except Exception as e:
        print("Failed to save news state:", e)

last_posted_news_time, posted_news_titles = load_news_state()
news_lock = threading.Lock()
news_cancel_event = threading.Event()
news_schedule_thread = None

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
from logging.handlers import RotatingFileHandler

class MemoryLogHandler(logging.Handler):
    def __init__(self, log_list, max_logs=50):
        super().__init__()
        self.log_list = log_list
        self.max_logs = max_logs

    def emit(self, record):
        try:
            msg = self.format(record)
            with state_lock:
                self.log_list.append(msg)
                while len(self.log_list) > self.max_logs:
                    self.log_list.pop(0)
        except Exception:
            self.handleError(record)

mem_handler = MemoryLogHandler(bot_state["logs"])
mem_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

# Configure log rotation (max 5MB per file, keeping 3 backups)
rotating_handler = RotatingFileHandler(
    "autoposter.log", 
    maxBytes=5 * 1024 * 1024, 
    backupCount=3,
    encoding="utf-8"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        rotating_handler,
        logging.StreamHandler(),
        mem_handler
    ]
)

# Silence the extremely noisy Flask (werkzeug), urllib3, and HTTP client (httpx) logs
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY")
raw_keys              = os.getenv("GEMINI_API_KEYS", "")
if raw_keys:
    GEMINI_API_KEYS   = [k.strip() for k in raw_keys.split(",") if k.strip()]
else:
    GEMINI_API_KEYS   = []
    if GEMINI_API_KEY:
        GEMINI_API_KEYS.append(GEMINI_API_KEY)
    idx = 2
    while True:
        key = os.getenv(f"GEMINI_API_KEY_{idx}")
        if not key:
            break
        GEMINI_API_KEYS.append(key.strip())
        idx += 1

BINANCE_SQUARE_KEY    = os.getenv("BINANCE_SQUARE_KEY")
CRYPTOPANIC_KEY       = os.getenv("CRYPTOPANIC_KEY", "")

BINANCE_POST_URL      = "https://www.binance.com/bapi/composite/v1/public/pgc/openApi/content/add"
BINANCE_UPLOAD_URL    = "https://www.binance.com/bapi/composite/v2/public/pgc/openApi/image/presignedUrl"
BINANCE_STATUS_URL    = "https://www.binance.com/bapi/composite/v2/public/pgc/openApi/image/imageStatus"
BINANCE_KLINES_URL    = "https://api.binance.com/api/v3/klines"
BINANCE_FUTURES_OI    = "https://fapi.binance.com/fapi/v1/openInterest"
BINANCE_FUNDING_URL   = "https://fapi.binance.com/fapi/v1/fundingRate"
COINGECKO_URL         = "https://api.coingecko.com/api/v3"
FEAR_GREED_URL        = "https://api.alternative.me/fng/"
CRYPTOPANIC_URL       = "https://cryptopanic.com/api/developer/v2/posts/"

POSTS_PER_DAY_MIN     = 10
POSTS_PER_DAY_MAX     = 10
DATA_REFRESH_EVERY    = 5       # refresh global data every N posts
CHART_PROBABILITY     = 0.70    # 70% of posts get a chart image

# Post-type target mix per day (weights, must sum to ~100)
POST_MIX = {
    "oversold_alert":    20,
    "overbought_warn":   20,
    "macd_signal":       20,
    "bollinger_squeeze": 20,
    "volume_alert":      20,
}

INTERVAL_BANDS   = [(45,300),(300,900),(900,2700),(2700,5400),(5400,10800)]
INTERVAL_WEIGHTS = [15, 30, 30, 15, 10]

HASHTAG_POOL = [
    "#crypto","#BinanceSquare","#Write2Earn","#Bitcoin","#Ethereum",
    "#DeFi","#Altcoins","#CryptoTrading","#BullRun","#DYOR",
    "#cryptonews","#Web3","#blockchain","#BTC","#ETH","#BNB","#SOL",
    "#CryptoInvesting","#hodl","#cryptomarket","#TechnicalAnalysis",
    "#CryptoSignals","#AltSeason",
]

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI ROTATOR
# ─────────────────────────────────────────────────────────────────────────────
class GeminiClientRotator:
    def __init__(self, api_keys: list[str]):
        if not api_keys:
            raise ValueError("No Gemini API keys provided. Set GEMINI_API_KEYS or GEMINI_API_KEY.")
        self.api_keys = api_keys
        self.clients = [genai.Client(api_key=k) for k in api_keys]
        self.current_idx = 0
        self.lock = threading.Lock()

    def get_client(self) -> genai.Client:
        with self.lock:
            return self.clients[self.current_idx]

    def rotate(self):
        with self.lock:
            old_idx = self.current_idx
            self.current_idx = (self.current_idx + 1) % len(self.clients)
            log.info(f"🔄 Rotating Gemini client from key index {old_idx} to {self.current_idx}")

    def generate_content_with_retry(self, post_type: str, analysis: dict, global_data: dict,
                                   news: list, recent_coins: list, recent_types: list) -> str:
        last_err = None
        for _ in range(len(self.clients)):
            try:
                client = self.get_client()
                content = generate_post(client, post_type, analysis, global_data, news, recent_coins, recent_types)
                return content
            except Exception as e:
                log.error(f"❌ Gemini error on key index {self.current_idx}: {e}")
                last_err = e
                self.rotate()
        raise last_err

# ─────────────────────────────────────────────────────────────────────────────
# COIN REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
COINS = [
    {"tag":"$BTC",  "cg_id":"bitcoin",             "cp":"BTC",  "sym":"BTCUSDT",  "fsym":"BTCUSDT"},
    {"tag":"$ETH",  "cg_id":"ethereum",            "cp":"ETH",  "sym":"ETHUSDT",  "fsym":"ETHUSDT"},
    {"tag":"$BNB",  "cg_id":"binancecoin",         "cp":"BNB",  "sym":"BNBUSDT",  "fsym":"BNBUSDT"},
    {"tag":"$SOL",  "cg_id":"solana",              "cp":"SOL",  "sym":"SOLUSDT",  "fsym":"SOLUSDT"},
    {"tag":"$XRP",  "cg_id":"ripple",              "cp":"XRP",  "sym":"XRPUSDT",  "fsym":"XRPUSDT"},
    {"tag":"$AVAX", "cg_id":"avalanche-2",         "cp":"AVAX", "sym":"AVAXUSDT", "fsym":"AVAXUSDT"},
    {"tag":"$LINK", "cg_id":"chainlink",           "cp":"LINK", "sym":"LINKUSDT", "fsym":"LINKUSDT"},
    {"tag":"$ARB",  "cg_id":"arbitrum",            "cp":"ARB",  "sym":"ARBUSDT",  "fsym":"ARBUSDT"},
    {"tag":"$DOGE", "cg_id":"dogecoin",            "cp":"DOGE", "sym":"DOGEUSDT", "fsym":"DOGEUSDT"},
    {"tag":"$ADA",  "cg_id":"cardano",             "cp":"ADA",  "sym":"ADAUSDT",  "fsym":"ADAUSDT"},
    {"tag":"$SUI",  "cg_id":"sui",                 "cp":"SUI",  "sym":"SUIUSDT",  "fsym":"SUIUSDT"},
    {"tag":"$PEPE", "cg_id":"pepe",                "cp":"PEPE", "sym":"PEPEUSDT", "fsym":"PEPEUSDT"},
    {"tag":"$NEAR", "cg_id":"near",                "cp":"NEAR", "sym":"NEARUSDT", "fsym":"NEARUSDT"},
    {"tag":"$INJ",  "cg_id":"injective-protocol",  "cp":"INJ",  "sym":"INJUSDT",  "fsym":"INJUSDT"},
    {"tag":"$DOT",  "cg_id":"polkadot",            "cp":"DOT",  "sym":"DOTUSDT",  "fsym":"DOTUSDT"},
    {"tag":"$ATOM", "cg_id":"cosmos",              "cp":"ATOM", "sym":"ATOMUSDT", "fsym":"ATOMUSDT"},
    {"tag":"$OP",   "cg_id":"optimism",            "cp":"OP",   "sym":"OPUSDT",   "fsym":"OPUSDT"},
    {"tag":"$WIF",  "cg_id":"dogwifcoin",          "cp":"WIF",  "sym":"WIFUSDT",  "fsym":"WIFUSDT"},
    {"tag":"$TIA",  "cg_id":"celestia",            "cp":"TIA",  "sym":"TIAUSDT",  "fsym":"TIAUSDT"},
    {"tag":"$JUP",  "cg_id":"jupiter-exchange-solana","cp":"JUP","sym":"JUPUSDT", "fsym":"JUPUSDT"},
]

# ─────────────────────────────────────────────────────────────────────────────
# TA ENGINE
# All indicators computed locally from raw OHLCV data.
# No third-party TA library needed — pure numpy/pandas.
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Compute RSI from close prices. Returns latest RSI value."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def compute_rsi_series(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Full RSI series for charting."""
    rsi = np.full(len(closes), np.nan)
    if len(closes) < period + 1:
        return rsi
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rsi[period] = 100 - (100 / (1 + avg_gain / (avg_loss or 1e-10)))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / (avg_loss or 1e-10)
        rsi[i + 1] = 100 - (100 / (1 + rs))
    return rsi

def compute_ema(values: np.ndarray, period: int) -> np.ndarray:
    ema = np.full(len(values), np.nan)
    k   = 2 / (period + 1)
    start = period - 1
    ema[start] = np.mean(values[:period])
    for i in range(start + 1, len(values)):
        ema[i] = values[i] * k + ema[i-1] * (1 - k)
    return ema

def compute_macd(closes: np.ndarray,
                 fast: int = 12, slow: int = 26, signal: int = 9
                 ) -> dict:
    """Returns macd_line, signal_line, histogram, and crossover direction."""
    ema_fast   = compute_ema(closes, fast)
    ema_slow   = compute_ema(closes, slow)
    macd_line  = ema_fast - ema_slow
    signal_line= compute_ema(macd_line[~np.isnan(macd_line)], signal)
    # Pad signal_line back to full length
    pad = len(macd_line) - len(signal_line)
    signal_full = np.concatenate([np.full(pad, np.nan), signal_line])
    histogram   = macd_line - signal_full

    # Detect crossover in last 3 candles
    recent_hist = histogram[~np.isnan(histogram)][-3:]
    crossover   = "none"
    if len(recent_hist) >= 2:
        if recent_hist[-2] < 0 and recent_hist[-1] >= 0:
            crossover = "bullish"
        elif recent_hist[-2] > 0 and recent_hist[-1] <= 0:
            crossover = "bearish"

    last_macd = macd_line[~np.isnan(macd_line)][-1] if any(~np.isnan(macd_line)) else 0
    last_sig  = signal_full[~np.isnan(signal_full)][-1] if any(~np.isnan(signal_full)) else 0
    last_hist = histogram[~np.isnan(histogram)][-1] if any(~np.isnan(histogram)) else 0

    return {
        "macd":       round(float(last_macd), 6),
        "signal":     round(float(last_sig), 6),
        "histogram":  round(float(last_hist), 6),
        "crossover":  crossover,
        "macd_series":   macd_line,
        "signal_series": signal_full,
        "hist_series":   histogram,
    }

def compute_bollinger(closes: np.ndarray, period: int = 20, std_mult: float = 2.0) -> dict:
    """Returns upper, middle, lower bands and %B position."""
    if len(closes) < period:
        mid = closes[-1]
        return {"upper": mid, "middle": mid, "lower": mid, "pct_b": 0.5, "squeeze": False}
    sma   = np.mean(closes[-period:])
    std   = np.std(closes[-period:], ddof=1)
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    price = closes[-1]
    pct_b = (price - lower) / (upper - lower) if upper != lower else 0.5
    # Squeeze: band width < 2% of price (low volatility)
    band_width = (upper - lower) / sma
    return {
        "upper":   round(upper, 6),
        "middle":  round(sma, 6),
        "lower":   round(lower, 6),
        "pct_b":   round(pct_b, 3),
        "squeeze": band_width < 0.02,
    }

def compute_support_resistance(highs: np.ndarray, lows: np.ndarray,
                                closes: np.ndarray, n: int = 3) -> dict:
    """
    Simple pivot-based S/R.
    Returns nearest support and resistance to current price.
    """
    price    = closes[-1]
    pivots_h = []
    pivots_l = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            pivots_h.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            pivots_l.append(lows[i])

    resistances = sorted([p for p in pivots_h if p > price])[:n]
    supports    = sorted([p for p in pivots_l if p < price], reverse=True)[:n]

    return {
        "resistance": resistances,
        "support":    supports,
        "price":      price,
    }

def compute_volume_trend(volumes: np.ndarray, period: int = 7) -> dict:
    """Compare recent volume to 7-day average."""
    if len(volumes) < period + 1:
        return {"trend": "neutral", "ratio": 1.0}
    avg      = np.mean(volumes[-period-1:-1])
    current  = volumes[-1]
    ratio    = current / avg if avg > 0 else 1.0
    trend    = "spike" if ratio > 2.0 else ("above" if ratio > 1.2 else
               ("below" if ratio < 0.8 else "neutral"))
    return {"trend": trend, "ratio": round(ratio, 2)}

def fmt_price(p) -> str:
    if p is None: return "N/A"
    if p >= 1000:  return f"${p:,.0f}"
    if p >= 1:     return f"${p:,.3f}"
    if p >= 0.001: return f"${p:.5f}"
    return f"${p:.8f}"

def fmt_pct(p) -> str:
    if p is None: return "N/A"
    return f"{'+' if p >= 0 else ''}{p:.2f}%"

def fmt_large(n) -> str:
    if n is None: return "N/A"
    if n >= 1e9: return f"${n/1e9:.2f}B"
    if n >= 1e6: return f"${n/1e6:.1f}M"
    return f"${n:,.0f}"

# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def parse_scraped_date(pub_str: str) -> datetime:
    """
    Parse Javascript, ISO, and standard HTTP/RSS date strings into UTC datetime.
    """
    now = datetime.utcnow()
    if not pub_str:
        return now
    pub_str = pub_str.strip()
    
    # ISO formats (e.g. 2026-06-11T17:15:19.000Z or 2026-06-11T17:15:19Z)
    if "T" in pub_str:
        try:
            t_part = pub_str.split(".")[0].replace("Z", "")
            return datetime.strptime(t_part[:19], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            pass

    # Javascript format (e.g. Thu Jun 11 2026 17:15:19 GMT+0530 (India Standard Time))
    if "GMT" in pub_str:
        try:
            parts = pub_str.split(" GMT")
            dt_part = parts[0].strip()
            dt = datetime.strptime(dt_part, "%a %b %d %Y %H:%M:%S")
            offset_str = parts[1].split("(")[0].strip()
            if len(offset_str) >= 5 and offset_str[0] in ("+", "-"):
                sign = 1 if offset_str[0] == "+" else -1
                hours = int(offset_str[1:3])
                minutes = int(offset_str[3:5])
                dt -= timedelta(hours=sign*hours, minutes=sign*minutes)
            return dt
        except Exception:
            pass
            
    # Standard RSS / HTTP header formats (e.g. Thu, 11 Jun 2026 17:15:19 GMT)
    try:
        dt_clean = pub_str.replace("GMT", "").replace("UTC", "").split("+")[0].split("-")[0].strip()
        return datetime.strptime(dt_clean, "%a, %d %b %Y %H:%M:%S")
    except Exception:
        pass

    return now


class DataPipeline:
    def __init__(self, cryptopanic_key: str = ""):
        self.cp_key  = cryptopanic_key
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "BinanceSquarePoster/2.0"
        })
        self.pushed_news = []
        self.pushed_news_lock = threading.Lock()

    # def _get(self, url: str, params: dict = None, timeout: int = 4) -> dict | list | None:
    #     try:
    #         r = self.session.get(url, params=params, timeout=timeout)
    #         r.raise_for_status()
    #         return r.json()
    #     except Exception as e:
    #         log.warning(f"  ⚠ Fetch failed [{url[:60]}]: {e}")
    #         return None
    def _get(self, url, params=None, timeout=4):
        try:
           r = self.session.get(url, params=params, timeout=timeout)
           log.info(f"URL={url} STATUS={r.status_code}")
           r.raise_for_status()
           return r.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                log.warning(f"⚠️ Rate limited (429) on: {url}")
            else:
                log.warning(f"⚠️ Fetch failed: {url} | Status: {e.response.status_code if e.response is not None else 'unknown'}")
            return None
        except Exception as e:
            log.warning(f"⚠️ Fetch error: {url} | {e}")
            return None

    # ── Binance klines (OHLCV) — free, no auth ──
    def fetch_klines(self, symbol: str, interval: str = "4h",
                     limit: int = 100) -> pd.DataFrame | None:
        data = self._get(BINANCE_KLINES_URL, params={
            "symbol": symbol, "interval": interval, "limit": limit
        })
        if not data:
            return None
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","trades","tbbav","tbqav","ignore"
        ])
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        return df

    # ── CoinGecko: market data (free) ──
    def fetch_market_data(self) -> dict:
        ids  = ",".join(c["cg_id"] for c in COINS)
        data = self._get(f"{COINGECKO_URL}/coins/markets", params={
            "vs_currency": "usd", "ids": ids,
            "order": "market_cap_desc", "per_page": 50,
            "price_change_percentage": "1h,24h,7d",
        })
        if not data: return {}
        result = {}
        for c in data:
            result[c["id"]] = {
                "price":     c.get("current_price"),
                "ch_1h":     c.get("price_change_percentage_1h_in_currency"),
                "ch_24h":    c.get("price_change_percentage_24h"),
                "ch_7d":     c.get("price_change_percentage_7d_in_currency"),
                "volume":    c.get("total_volume"),
                "mcap":      c.get("market_cap"),
                "high_24h":  c.get("high_24h"),
                "low_24h":   c.get("low_24h"),
                "symbol":    c.get("symbol","").upper(),
            }
        log.info(f"  📊 Market data: {len(result)} coins")
        return result

    # ── Fear & Greed (alternative.me — free, no key) ──
    def fetch_fear_greed(self) -> dict:
        data = self._get(FEAR_GREED_URL, params={"limit": 3})
        if not data or "data" not in data:
            return {"value": "N/A", "label": "Unknown"}
        latest = data["data"][0]
        return {"value": int(latest.get("value", 50)),
                "label": latest.get("value_classification", "Unknown")}

    # ── Binance futures: open interest ──
    def fetch_open_interest(self, fsym: str) -> dict | None:
        data = self._get(BINANCE_FUTURES_OI, params={"symbol": fsym})
        if not data: return None
        return {"oi": float(data.get("openInterest", 0)),
                "time": data.get("time")}

    # ── Binance futures: funding rate ──
    def fetch_funding_rate(self, fsym: str) -> float | None:
        data = self._get(BINANCE_FUNDING_URL,
                         params={"symbol": fsym, "limit": 1})
        if not data or not isinstance(data, list) or len(data) == 0:
            return None
        return float(data[0].get("fundingRate", 0))

    def fetch_news(self, currency: str = None) -> list[dict]:
        results = []
        now = datetime.utcnow()
        
        # Append only pushed news items from the local cache
        with self.pushed_news_lock:
            valid_pushed = []
            for item in self.pushed_news:
                pub_dt = item.get("pub_dt")
                if pub_dt:
                    age_min = int((now - pub_dt).total_seconds() / 60)
                    if age_min < 1440:  # Keep news from last 24 hours
                        if currency and currency.upper() not in item["title"].upper():
                            continue
                        results.append({
                            "title": item["title"],
                            "age_min": age_min,
                            "time": pub_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                        })
                        valid_pushed.append(item)
            self.pushed_news = valid_pushed
        
        # Sort by age_min so that the latest news is always at the top
        results = sorted(results, key=lambda x: x["age_min"])
        
        if results:
            log.info(f"📰 Successfully fetched {len(results)} news items from pushed queue. Latest articles:")
            for item in results[:3]:
                log.info(f"   • [{item['time']} / {item['age_min']}m ago] {item['title']}")
        else:
            log.warning("⚠️ No pushed news items currently queued.")
            
        return results[:15]

    # ── Trending coins (CoinGecko) ──
    def fetch_trending(self) -> list[str]:
        data = self._get(f"{COINGECKO_URL}/search/trending")
        if not data: return []
        return [c["item"]["symbol"].upper() for c in data.get("coins",[])[:7]]

    # ── Full coin analysis ──
    def analyse_coin(self, coin: dict, interval: str = "4h") -> dict | None:
        """
        Fetch klines + compute all TA indicators for one coin.
        Returns a rich analysis dict, or None if data unavailable.
        """
        df = self.fetch_klines(coin["sym"], interval=interval, limit=120)
        if df is None or len(df) < 30:
            return None

        closes  = df["close"].values
        highs   = df["high"].values
        lows    = df["low"].values
        volumes = df["volume"].values
        times   = df["open_time"].values

        rsi      = compute_rsi(closes)
        rsi_ser  = compute_rsi_series(closes)
        macd     = compute_macd(closes)
        boll     = compute_bollinger(closes)
        sr       = compute_support_resistance(highs, lows, closes)
        vol_trnd = compute_volume_trend(volumes)

        # Determine primary signal
        signal = _select_signal(rsi, macd, boll, vol_trnd)

        return {
            "coin":       coin,
            "interval":   interval,
            "df":         df,
            "closes":     closes,
            "highs":      highs,
            "lows":       lows,
            "volumes":    volumes,
            "times":      times,
            "rsi":        rsi,
            "rsi_series": rsi_ser,
            "macd":       macd,
            "bollinger":  boll,
            "sr":         sr,
            "vol_trend":  vol_trnd,
            "signal":     signal,
        }


def _select_signal(rsi: float, macd: dict, boll: dict, vol: dict) -> str:
    """
    Data-driven signal selection.
    Priority: strongest signal wins.
    """
    if rsi < 28:
        return "oversold_alert"
    if rsi > 72:
        return "overbought_warn"
    if macd["crossover"] == "bullish":
        return "macd_signal_bull"
    if macd["crossover"] == "bearish":
        return "macd_signal_bear"
    if boll["squeeze"]:
        return "bollinger_squeeze"
    if vol["trend"] == "spike":
        return "volume_alert"
    if boll["pct_b"] > 0.9:
        return "overbought_warn"
    if boll["pct_b"] < 0.1:
        return "oversold_alert"
    return "market_vibe"


# ─────────────────────────────────────────────────────────────────────────────
# CHART GENERATOR
# Price chart (top) + RSI subplot (bottom)
# Support/Resistance lines drawn automatically
# ─────────────────────────────────────────────────────────────────────────────

class ChartGenerator:

    THEMES = {
        "dark": {
            "bg": "#0d1117", "panel": "#0d1117", "text": "#e6edf3",
            "grid": "#21262d", "tick": "#8b949e",
            "green": "#3fb950", "red": "#f85149",
            "blue": "#58a6ff", "orange": "#d29922",
            "sr_support": "#3fb95055", "sr_resist": "#f8514955",
        },
        "midnight": {
            "bg": "#070d1a", "panel": "#0a1428", "text": "#ccd6f6",
            "grid": "#112240", "tick": "#8892b0",
            "green": "#64ffda", "red": "#ff5370",
            "blue": "#82aaff", "orange": "#ffcb6b",
            "sr_support": "#64ffda44", "sr_resist": "#ff537044",
        },
        "slate": {
            "bg": "#1e293b", "panel": "#0f172a", "text": "#e2e8f0",
            "grid": "#334155", "tick": "#94a3b8",
            "green": "#4ade80", "red": "#f87171",
            "blue": "#60a5fa", "orange": "#fb923c",
            "sr_support": "#4ade8044", "sr_resist": "#f8717144",
        },
    }

    def generate(self, analysis: dict) -> bytes | None:
        try:
            return self._draw(analysis)
        except Exception as e:
            log.warning(f"  Chart generation failed: {e}")
            return None

    def _draw(self, a: dict) -> bytes:
        theme = self.THEMES[random.choice(list(self.THEMES.keys()))]
        df    = a["df"]
        coin  = a["coin"]
        rsi_s = a["rsi_series"]
        macd  = a["macd"]
        sr    = a["sr"]
        boll  = a["bollinger"]

        times  = df["open_time"].tolist()
        closes = df["close"].values
        price  = closes[-1]
        ch24   = float(((closes[-1] - closes[-24]) / closes[-24]) * 100) if len(closes) >= 24 else 0
        is_up  = ch24 >= 0
        color  = theme["green"] if is_up else theme["red"]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(11, 6.5),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.04},
            facecolor=theme["bg"]
        )

        # ── TOP: price + Bollinger + S/R ──
        ax1.set_facecolor(theme["panel"])
        ax1.plot(times, closes, color=color, linewidth=1.8, zorder=5)

        # Bollinger Bands
        sma = pd.Series(closes).rolling(20).mean().values
        std = pd.Series(closes).rolling(20).std().values
        upper = sma + 2 * std
        lower = sma - 2 * std
        ax1.plot(times, sma,   color=theme["blue"],   linewidth=0.8, alpha=0.7, linestyle="--")
        ax1.plot(times, upper, color=theme["orange"],  linewidth=0.7, alpha=0.5)
        ax1.plot(times, lower, color=theme["orange"],  linewidth=0.7, alpha=0.5)
        ax1.fill_between(times, upper, lower, alpha=0.04, color=theme["orange"])

        # Fill under price line
        ax1.fill_between(times, closes, closes.min(), alpha=0.08, color=color)

        # Support lines
        for s in sr["support"][:2]:
            ax1.axhline(s, color=theme["green"], linewidth=0.9,
                        linestyle=":", alpha=0.75)
            ax1.text(times[-1], s, f" S {fmt_price(s)}", color=theme["green"],
                     fontsize=6.5, va="center", alpha=0.9)

        # Resistance lines
        for r in sr["resistance"][:2]:
            ax1.axhline(r, color=theme["red"], linewidth=0.9,
                        linestyle=":", alpha=0.75)
            ax1.text(times[-1], r, f" R {fmt_price(r)}", color=theme["red"],
                     fontsize=6.5, va="center", alpha=0.9)

        # Current price marker
        ax1.axhline(price, color=color, linewidth=0.6, alpha=0.4)

        # Price label
        sign = "▲" if is_up else "▼"
        ax1.set_title(
            f"{coin['tag']}  •  {fmt_price(price)}  {sign} {fmt_pct(ch24)}  "
            f"│  RSI {a['rsi']}  │  {a['interval'].upper()} Chart",
            color=theme["text"], fontsize=11.5, fontweight="bold",
            pad=10, loc="left"
        )

        # Bollinger legend
        patches = [
            mpatches.Patch(color=theme["blue"],   alpha=0.7, label="BB Mid"),
            mpatches.Patch(color=theme["orange"], alpha=0.5, label="BB ±2σ"),
            mpatches.Patch(color=theme["green"],  alpha=0.7, label="Support"),
            mpatches.Patch(color=theme["red"],    alpha=0.7, label="Resistance"),
        ]
        ax1.legend(handles=patches, loc="upper left", fontsize=7,
                   facecolor=theme["panel"], edgecolor=theme["grid"],
                   labelcolor=theme["tick"])

        _style_axis(ax1, theme)

        # ── BOTTOM: RSI ──
        ax2.set_facecolor(theme["panel"])
        valid_mask = ~np.isnan(rsi_s)
        rsi_times  = [times[i] for i in range(len(times)) if valid_mask[i]]
        rsi_vals   = rsi_s[valid_mask]

        ax2.plot(rsi_times, rsi_vals, color=theme["blue"], linewidth=1.3)
        ax2.axhline(70, color=theme["red"],   linewidth=0.7, linestyle="--", alpha=0.6)
        ax2.axhline(30, color=theme["green"], linewidth=0.7, linestyle="--", alpha=0.6)
        ax2.axhline(50, color=theme["tick"],  linewidth=0.5, linestyle=":",  alpha=0.4)

        # Shade overbought/oversold zones
        ax2.fill_between(rsi_times, 70, 100,
                         where=[v >= 70 for v in rsi_vals],
                         alpha=0.15, color=theme["red"])
        ax2.fill_between(rsi_times, 0, 30,
                         where=[v <= 30 for v in rsi_vals],
                         alpha=0.15, color=theme["green"])

        current_rsi = a["rsi"]
        rsi_color   = theme["red"] if current_rsi > 70 else (
                       theme["green"] if current_rsi < 30 else theme["blue"])
        ax2.set_ylabel(f"RSI {current_rsi:.0f}", color=rsi_color,
                       fontsize=8, fontweight="bold")
        ax2.set_ylim(0, 100)
        ax2.text(rsi_times[0], 72, "OB", color=theme["red"],   fontsize=6, alpha=0.7)
        ax2.text(rsi_times[0], 22, "OS", color=theme["green"], fontsize=6, alpha=0.7)

        _style_axis(ax2, theme)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=25, ha="right",
                 fontsize=7, color=theme["tick"])

        # Watermark
        fig.text(0.99, 0.01, "Binance Square", fontsize=7,
                 color=theme["tick"], alpha=0.3, ha="right")

        ax1.tick_params(labelbottom=False)
        plt.tight_layout(pad=0.5)

        buf = io.BytesIO()
        plt.savefig(buf, format="PNG", dpi=140,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()


def _style_axis(ax, theme: dict):
    ax.set_facecolor(theme["panel"])
    ax.grid(True, color=theme["grid"], linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=theme["tick"], labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(theme["grid"])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: fmt_price(x)))


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE UPLOADER
# 2-step: get presigned URL → PUT image bytes
# ─────────────────────────────────────────────────────────────────────────────

class ImageUploader:
    def __init__(self, api_key: str):
        self.key = api_key

    def _headers(self) -> dict:
        return {"X-Square-OpenAPI-Key": self.key,
                "Content-Type": "application/json",
                "clienttype": "web"}

    def upload(self, image_bytes: bytes) -> str | None:
        # Step 1: get presigned URL & file ticket
        try:
            r = requests.post(BINANCE_UPLOAD_URL,
                              headers=self._headers(),
                              json={"imageName": "chart.png"},
                              timeout=15)
            data = r.json()
            if data.get("code") != "000000":
                log.warning(f"  Upload presigned URL failed: {data.get('message')}")
                return None
            res_data = data.get("data", {})
            put_url = res_data.get("presignedUrl")
            file_ticket = res_data.get("fileTicket")
        except Exception as e:
            log.warning(f"  Upload token error: {e}")
            return None

        if not put_url or not file_ticket:
            log.warning(f"  Missing presigned URL or file ticket: {res_data}")
            return None

        # Step 2: PUT image to presigned URL
        try:
            put_r = requests.put(put_url, data=image_bytes,
                                 headers={"Content-Type": "image/png"},
                                 timeout=30)
            if put_r.status_code not in (200, 204):
                log.warning(f"  Image PUT failed: {put_r.status_code}")
                return None
        except Exception as e:
            log.warning(f"  Image PUT error: {e}")
            return None

        # Step 3: Poll imageStatus until public CDN URL is generated
        cdn_url = None
        try:
            for attempt in range(1, 11):
                time.sleep(1.5)
                r_st = requests.post(BINANCE_STATUS_URL,
                                     headers=self._headers(),
                                     json={"fileTicket": file_ticket},
                                     timeout=10)
                data_st = r_st.json().get("data", {})
                if data_st and data_st.get("imageUrl"):
                    cdn_url = data_st.get("imageUrl")
                    break
                if data_st and data_st.get("status") == 2:
                    log.warning(f"  Image processing failed on backend: {data_st.get('failedReason')}")
                    return None
        except Exception as e:
            log.warning(f"  Image status polling error: {e}")
            return None

        if not cdn_url:
            log.warning("  Timeout waiting for public image URL from Binance status endpoint")
            return None

        log.info(f"  🖼  Image uploaded → {cdn_url[:55]}...")
        return cdn_url


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI WRITER
# System prompt enforces persona + data-grounded writing
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a sharp, data-driven crypto KOL posting on Binance Square.
Your edge: you post real computed indicators, not opinions.
You sound like a confident professional trader — concise, specific, no fluff.

HARD RULES:
1. 50–130 words maximum. Short posts only.
2. Always use the EXACT numbers given to you — price, RSI, MACD, support, resistance.
3. Cashtags like $BTC, $ETH must appear in the body (at least 1).
4. Hashtags go on the LAST LINE only, 2–5 tags.
5. ZERO external URLs or website links anywhere.
6. ZERO social handles (no @, no Telegram, no Discord).
7. No "I think" or "In my opinion" openers.
8. No guaranteed return language ("100% sure", "definitely").
9. Data source names never appear (not "CoinGecko says" — just state the fact).
10. Vary sentence structure — mix 3-word punchy lines with longer analytical ones.
11. 1–3 emojis max, only where they add punch, never mid-sentence.
12. Output ONLY the post text. Zero preamble."""


POST_INSTRUCTIONS = {

"oversold_alert": """
The data shows RSI is in OVERSOLD territory. Write a post that:
- Leads with the RSI number (don't say "RSI is" — work it in naturally)
- States current price
- Notes the nearest support level from S/R data
- Adds a brief historical note ("last time RSI was this low...")
- Closes with a cautious but interested tone (not a buy order)
""",

"overbought_warn": """
The data shows RSI is in OVERBOUGHT territory. Write a post that:
- Uses the RSI reading as the hook
- States current price and the 24h gain
- References the nearest resistance level
- Warns about overextension without being dramatic
- Tone: experienced trader, not panicking but cautious
""",

"macd_signal_bull": """
MACD just crossed bullish. Write a post that:
- Opens with the signal ("MACD just flipped bullish on the 4H")
- States the coin, current price
- Cross-checks: mention RSI to show it's not overbought
- Notes the next resistance to watch
- Tone: measured optimism, not euphoria
""",

"macd_signal_bear": """
MACD just crossed bearish. Write a post that:
- Opens with the signal
- States coin, current price, and how much it's already dropped
- Notes the support level to watch
- Warns: bear crossovers in downtrends are serious
- Tone: clear-eyed, not doom-posting
""",

"bollinger_squeeze": """
Bollinger Bands are squeezing (extremely low volatility). Write a post that:
- Explains a squeeze = explosion coming (direction unknown)
- States the coin and current price
- Notes the upper and lower band levels as breakout targets
- Creates urgency without predicting direction
- Tone: analytical, like a chess player seeing the board
""",

"volume_alert": """
Volume just spiked significantly above average. Write a post that:
- Opens with the volume stat (e.g., "2.4x normal volume just printed on...")
- States coin and price
- Asks: is this accumulation or distribution? Let readers decide
- References the RSI and price action briefly
- Tone: curious analyst, invites engagement
""",

"breakout_news": """
A real news headline just broke. Write a post that:
- Reacts to the news topic (paraphrase — no source names, no URLs)
- States how the coin is reacting (price % change)
- Gives ONE key level to watch
- Brief take on whether this is a 2-hour move or a 2-week move
- Tone: fast, sharp, first-mover energy
""",

"whale_oi_watch": """
Open Interest and/or funding rate data is notable. Write a post that:
- Opens with the OI stat or funding rate
- Explains what it means in plain language
- Cross-references the current price
- Notes whether longs or shorts are at risk
- Tone: informed insider who reads the order book
""",

"greed_warning": """
Fear & Greed is in extreme territory. Write a post that:
- Opens with the actual index number
- Gives brief historical context (what happened last time it was this high/low)
- Does NOT say sell or buy — says "manage your risk"
- Mentions 1-2 coins relevant to the market condition
- Tone: wise veteran, not an alarmist
""",

"price_target": """
Write a forward-looking price analysis post that:
- States the current price
- Gives 2-3 realistic target levels based on S/R data provided
- Uses a price ladder format (e.g., 3200 → 4100 → 5500)
- Anchors targets to the resistance levels in the data
- Ends with DYOR or a risk disclaimer
""",

"market_vibe": """
Write a brief market vibe/sentiment post that:
- Captures the current mood (Fear & Greed, overall 24h performance)
- Mentions 2-3 coins and their actual 24h change
- Gives ONE simple take on where the market is
- Could be funny, dry, sharp — pick a tone
- No forced optimism or pessimism — just read the room
""",

"educational": """
Write a short educational post that:
- Explains ONE concept (RSI, MACD, Bollinger, support/resistance, funding rate)
- Uses the current coin's ACTUAL DATA as the example
- Is simple enough for a beginner but not condescending
- Ends with a practical takeaway
- Tone: teacher, not lecturer
""",
}


def build_prompt(post_type: str, analysis: dict, global_data: dict,
                 news: list, recent_coins: list, recent_types: list) -> str:
    coin = analysis["coin"]
    m    = global_data["market"].get(coin["cg_id"], {})
    fg   = global_data["fg"]
    sr   = analysis["sr"]
    rsi  = analysis["rsi"]
    macd = analysis["macd"]
    boll = analysis["bollinger"]
    vol  = analysis["vol_trend"]

    # Build data block
    data_block = f"""
=== LIVE COMPUTED DATA ({global_data['fetched_at']}) ===
Coin: {coin['tag']}
Price: {fmt_price(m.get('price') or analysis['closes'][-1])}
1h Change: {fmt_pct(m.get('ch_1h'))}
24h Change: {fmt_pct(m.get('ch_24h'))}
7d Change: {fmt_pct(m.get('ch_7d'))}
24h Volume: {fmt_large(m.get('volume'))}
Market Cap: {fmt_large(m.get('mcap'))}
24h High: {fmt_price(m.get('high_24h'))} | 24h Low: {fmt_price(m.get('low_24h'))}

--- COMPUTED INDICATORS (4H chart) ---
RSI(14): {rsi} {'🔴 OVERBOUGHT' if rsi > 70 else ('🟢 OVERSOLD' if rsi < 30 else '⚪ NEUTRAL')}
MACD Line: {macd['macd']} | Signal: {macd['signal']} | Histogram: {macd['histogram']}
MACD Crossover: {macd['crossover'].upper()}
Bollinger Upper: {fmt_price(boll['upper'])} | Mid: {fmt_price(boll['middle'])} | Lower: {fmt_price(boll['lower'])}
Bollinger %B: {boll['pct_b']} (>0.8 overbought, <0.2 oversold)
Bollinger Squeeze: {'YES — breakout imminent' if boll['squeeze'] else 'No'}
Volume Trend: {vol['trend'].upper()} ({vol['ratio']}x avg)

--- SUPPORT & RESISTANCE ---
Resistance levels: {[fmt_price(r) for r in sr['resistance'][:3]] or 'None detected'}
Support levels: {[fmt_price(s) for s in sr['support'][:3]] or 'None detected'}

--- MARKET SENTIMENT ---
Fear & Greed: {fg['value']} ({fg['label']})
Trending coins: {', '.join(global_data.get('trending', [])[:5]) or 'N/A'}
"""

    # Add news if relevant
    fresh_news = [n for n in news if n["age_min"] < 120][:4]
    if fresh_news:
        data_block += "\n--- RECENT NEWS (use for breakout_news type only) ---\n"
        for n in fresh_news:
            data_block += f"  [{n['age_min']}min ago] {n['title']}\n"

    # Add OI/funding if available
    if "oi" in global_data:
        oi_data = global_data["oi"].get(coin["sym"], {})
        if oi_data:
            data_block += f"\nOpen Interest: {fmt_large(oi_data.get('oi'))}\n"
    if "funding" in global_data:
        fr = global_data["funding"].get(coin["sym"])
        if fr is not None:
            data_block += f"Funding Rate: {fr:.4%} ({'longs paying' if fr > 0 else 'shorts paying'})\n"

    data_block += "=" * 55

    tags = " ".join(random.sample(HASHTAG_POOL, random.randint(3, 5)))

    instruction = POST_INSTRUCTIONS.get(post_type, POST_INSTRUCTIONS["market_vibe"])

    return f"""{data_block}

POST TYPE: {post_type}
INSTRUCTION: {instruction}

RECENT COINS (avoid repeating): {', '.join(recent_coins[-5:]) or 'none'}
RECENT POST TYPES (avoid repeating): {', '.join(recent_types[-3:]) or 'none'}

End post with hashtags: {tags}

Write the post now. Output ONLY the post text."""


def generate_post(client: genai.Client, post_type: str, analysis: dict,
                  global_data: dict, news: list,
                  recent_coins: list, recent_types: list) -> str:
    prompt = build_prompt(post_type, analysis, global_data, news,
                          recent_coins, recent_types)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.95,
            top_p=0.95,
            max_output_tokens=3000,
        )
    )
    return resp.text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLISHER
# ─────────────────────────────────────────────────────────────────────────────

def publish(content: str, image_urls: list = None) -> dict:
    headers = {
        "X-Square-OpenAPI-Key": BINANCE_SQUARE_KEY,
        "Content-Type": "application/json",
        "clienttype": "web",
    }
    payload = {"bodyTextOnly": content, "contentType": 1}
    if image_urls:
        payload["imageList"] = image_urls[:4]
    resp = requests.post(BINANCE_POST_URL, headers=headers,
                         json=payload, timeout=15)
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL SELECTOR — picks which coin + post type for next post
# Uses weighted mixing so day's output matches POST_MIX targets
# ─────────────────────────────────────────────────────────────────────────────

class SignalSelector:
    def __init__(self, pipeline: DataPipeline):
        self.pipeline    = pipeline
        self._type_count = {k: 0 for k in POST_MIX}

    def select(self, global_data: dict, news: list,
               recent_coins: list, recent_types: list) -> tuple[dict, str] | None:
        """
        Returns (analysis, post_type) or None.
        Strategy:
          1. Select a target post_type from POST_MIX using weighted random choices based on current deficits.
          2. Scan the coin pool on-demand, caching analyses, to find a coin satisfying that type's criteria.
          3. Fall back to any coin with an active TA signal if target is not met.
          4. Absolute fallback to educational, price_target, or market_vibe on a random coin.
        """
        # 1. Helper to get current counts mapped to POST_MIX keys (e.g. aggregate macd_signal_bull/bear)
        def get_current_counts():
            counts = {}
            for k in POST_MIX:
                if k == "macd_signal":
                    counts[k] = self._type_count.get("macd_signal_bull", 0) + \
                                self._type_count.get("macd_signal_bear", 0)
                else:
                    counts[k] = self._type_count.get(k, 0)
            return counts

        # 2. Calculate remaining deficits for each post type
        counts = get_current_counts()
        deficits = {k: max(0, POST_MIX[k] - counts[k]) for k in POST_MIX}
        
        # 3. Choose a target post type weighted by remaining deficits
        eligible_types = [k for k, v in deficits.items() if v > 0]
        if not eligible_types:
            eligible_types = list(POST_MIX.keys())
            weights = [POST_MIX[k] for k in eligible_types]
        else:
            weights = [deficits[k] for k in eligible_types]

        target_type = random.choices(eligible_types, weights=weights, k=1)[0]
        log.info(f"🎲 Selected target post type: {target_type} (deficit weight: {deficits.get(target_type, 0)})")

        # 4. Prepare coin pool (shuffle and exclude recent coins to keep variety)
        pool = [c for c in COINS if c["tag"] not in recent_coins[-3:]]
        random.shuffle(pool)

        # On-demand analysis cache to prevent redundant API calls
        analyses = {}
        def get_analysis(coin):
            tag = coin["tag"]
            if tag not in analyses:
                analyses[tag] = self.pipeline.analyse_coin(coin)
            return analyses[tag]

        # Helper to find first coin satisfying a specific TA condition
        def find_coin_for_signal(condition_fn):
            for coin in pool:
                a = get_analysis(coin)
                if a and condition_fn(a):
                    return a
            return None

        # 5. Try to satisfy the chosen target_type
        a_selected = None
        final_type = target_type

        if target_type == "oversold_alert":
            a_selected = find_coin_for_signal(lambda x: x["rsi"] < 30 or x["bollinger"]["pct_b"] < 0.1)

        elif target_type == "overbought_warn":
            a_selected = find_coin_for_signal(lambda x: x["rsi"] > 70 or x["bollinger"]["pct_b"] > 0.9)

        elif target_type == "macd_signal":
            a_selected = find_coin_for_signal(lambda x: x["macd"]["crossover"] in ("bullish", "bearish"))
            if a_selected:
                final_type = "macd_signal_bull" if a_selected["macd"]["crossover"] == "bullish" else "macd_signal_bear"

        elif target_type == "bollinger_squeeze":
            a_selected = find_coin_for_signal(lambda x: x["bollinger"]["squeeze"])

        elif target_type == "volume_alert":
            a_selected = find_coin_for_signal(lambda x: x["vol_trend"]["trend"] == "spike")

        elif target_type == "breakout_news":
            breaking = [n for n in news if n["age_min"] < 120]
            if breaking:
                # Try to find a coin mentioned in news
                for n in breaking:
                    for coin in pool:
                        if coin["tag"].lower() in n["title"].lower() or coin["cp"].lower() in n["title"].lower():
                            a = get_analysis(coin)
                            if a:
                                a_selected = a
                                break
                    if a_selected:
                        break
                # Fallback to BTC/ETH if no news mention matching
                if not a_selected:
                    for tag in ("$BTC", "$ETH"):
                        match_coin = next((c for c in pool if c["tag"] == tag), None)
                        if match_coin:
                            a = get_analysis(match_coin)
                            if a:
                                a_selected = a
                                break

        elif target_type == "whale_oi_watch":
            for coin in pool:
                has_oi = "oi" in global_data and coin["sym"] in global_data["oi"]
                has_funding = "funding" in global_data and coin["sym"] in global_data["funding"]
                if has_oi or has_funding:
                    a = get_analysis(coin)
                    if a:
                        a_selected = a
                        break

        elif target_type == "greed_warning":
            for tag in ("$BTC", "$ETH", "$BNB", "$SOL"):
                match_coin = next((c for c in pool if c["tag"] == tag), None)
                if match_coin:
                    a = get_analysis(match_coin)
                    if a:
                        a_selected = a
                        break

        elif target_type in ("price_target", "market_vibe", "educational"):
            # Can run on any analyzed coin
            for coin in pool:
                a = get_analysis(coin)
                if a:
                    a_selected = a
                    break

        # 6. Fallback: if we couldn't satisfy target_type, find ANY coin with an active TA signal
        if not a_selected:
            log.info(f"⚠️ Could not satisfy target type '{target_type}'. Scanning for any active signal...")
            for coin in pool:
                a = get_analysis(coin)
                if a:
                    sig = a["signal"]
                    mapped_type = {
                        "oversold_alert":    "oversold_alert",
                        "overbought_warn":   "overbought_warn",
                        "macd_signal_bull":  "macd_signal_bull",
                        "macd_signal_bear":  "macd_signal_bear",
                        "bollinger_squeeze": "bollinger_squeeze",
                        "volume_alert":      "volume_alert",
                    }.get(sig)
                    if mapped_type:
                        a_selected = a
                        final_type = mapped_type
                        log.info(f"✨ Fallback to active TA signal '{final_type}' on {coin['tag']}")
                        break

        # 7. Absolute fallback: pick any analyzed coin and choose a general type
        if not a_selected:
            for coin in pool:
                a = get_analysis(coin)
                if a:
                    a_selected = a
                    final_type = random.choice(["market_vibe", "price_target", "educational"])
                    log.info(f"✨ Absolute fallback to '{final_type}' on {coin['tag']}")
                    break

        if a_selected:
            self._type_count[final_type] = self._type_count.get(final_type, 0) + 1
            return a_selected, final_type

        return None


# ─────────────────────────────────────────────────────────────────────────────
# INTERVAL + SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def pick_interval() -> int:
    if random.random() < 0.10:
        return random.randint(15, 55)      # micro-burst
    band   = random.choices(INTERVAL_BANDS, weights=INTERVAL_WEIGHTS, k=1)[0]
    base   = random.randint(band[0], band[1])
    jitter = int(base * random.uniform(-0.25, 0.25))
    return max(15, base + jitter)

def fmt_sec(s: int) -> str:
    if s < 60:   return f"{s}s"
    if s < 3600: return f"{s//60}m {s%60}s"
    return f"{s//3600}h {(s%3600)//60}m"

def build_schedule(n: int) -> list[datetime]:
    now   = datetime.utcnow()
    start = now.replace(hour=6, minute=0, second=0, microsecond=0)
    end   = now.replace(hour=23, minute=0, second=0, microsecond=0)
    if now > start:
        start = now + timedelta(seconds=30)
    total = int((end - start).total_seconds())
    if total <= 0:
        return [now + timedelta(seconds=i*90) for i in range(n)]
    return sorted([start + timedelta(seconds=random.randint(0, total)) for _ in range(n)])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SESSION
# ─────────────────────────────────────────────────────────────────────────────

# Global managers
global_pipeline = None
global_rotator = None
global_charts = None
global_uploader = None
global_selector = None

def init_globals():
    global global_pipeline, global_rotator, global_charts, global_uploader, global_selector
    
    if not GEMINI_API_KEYS:
        raise ValueError("Set GEMINI_API_KEYS or GEMINI_API_KEY in .env")
    if not BINANCE_SQUARE_KEY:
        raise ValueError("Set BINANCE_SQUARE_KEY in .env")
        
    global_pipeline = DataPipeline(cryptopanic_key=CRYPTOPANIC_KEY)
    global_rotator = GeminiClientRotator(GEMINI_API_KEYS)
    global_charts = ChartGenerator()
    global_uploader = ImageUploader(api_key=BINANCE_SQUARE_KEY)
    global_selector = SignalSelector(pipeline=global_pipeline)

def trigger_single_post():
    global bot_state, global_pipeline, global_rotator, global_charts, global_uploader, global_selector
    
    if not global_pipeline:
        init_globals()
        
    log.info("⚡ Triggering manual post-now cycle...")
    
    # 1. Fetch fresh data
    global_data = _refresh_global(global_pipeline)
    news = global_pipeline.fetch_news()
    
    # 2. Select signal + coin
    result = global_selector.select(global_data, news, bot_state["recent_coins"], bot_state["recent_types"])
    if not result:
        raise ValueError("No trading signal or fallback candidate found.")
        
    analysis, post_type = result
    coin = analysis["coin"]
    log.info(f"🎯 Manual Signal: [{post_type}] on {coin['tag']}")
    
    # 3. Generate text using rotator
    coin_news = global_pipeline.fetch_news(currency=coin["cp"])
    all_news = news + [n for n in coin_news if n not in news]
    
    content = global_rotator.generate_content_with_retry(
        post_type, analysis, global_data, all_news,
        bot_state["recent_coins"], bot_state["recent_types"]
    )
    
    words = len(content.split())
    if words < 10:
        raise ValueError(f"Generated content was too short: {words} words.")
    if words > 200:
        content = " ".join(content.split()[:180]) + "..."
        
    # 4. Generate and upload chart
    image_urls = []
    chart_bytes = global_charts.generate(analysis)
    if chart_bytes:
        cdn = global_uploader.upload(chart_bytes)
        if cdn:
            image_urls.append(cdn)
            
    # 5. Publish
    result_pub = publish(content, image_urls or None)
    code = result_pub.get("code", "")
    
    if code == "000000":
        post_id = result_pub.get("data", {}).get("id", "unknown")
        url = f"https://www.binance.com/square/post/{post_id}"
        log.info(f"  ✅ Manual Post Success: {url}")
        
        with state_lock:
            bot_state["posts_published"] += 1
            bot_state["recent_posts"].append({
                "time": datetime.utcnow().isoformat() + "Z",
                "content": content,
                "status": "Success",
                "url": url
            })
            if len(bot_state["recent_posts"]) > 20:
                bot_state["recent_posts"].pop(0)
                
            bot_state["recent_coins"].append(coin["tag"])
            bot_state["recent_types"].append(post_type)
            if len(bot_state["recent_coins"]) > 12: bot_state["recent_coins"].pop(0)
            if len(bot_state["recent_types"]) > 8:  bot_state["recent_types"].pop(0)
            
        return url
    else:
        msg = result_pub.get("message", "no message")
        log.warning(f"  ⚠ Manual Post Rejected: {code} | {msg}")
        with state_lock:
            bot_state["posts_failed"] += 1
            bot_state["recent_posts"].append({
                "time": datetime.utcnow().isoformat() + "Z",
                "content": content,
                "status": f"Rejected ({code})",
                "url": ""
            })
            if len(bot_state["recent_posts"]) > 20:
                bot_state["recent_posts"].pop(0)
        raise ValueError(f"Binance rejected post: {code} - {msg}")

def run_daily_session():
    global bot_state, global_pipeline, global_rotator, global_charts, global_uploader, global_selector
    
    if not global_pipeline:
        init_globals()
        
    log.info("🚀 Starting daily session — fetching initial market data...")
    with state_lock:
        bot_state["status_message"] = "Fetching initial market data..."
        
    global_data = _refresh_global(global_pipeline)
    news = global_pipeline.fetch_news()
    
    n_posts = random.randint(POSTS_PER_DAY_MIN, POSTS_PER_DAY_MAX)
    schedule = build_schedule(n_posts)
    log.info(f"📅 {n_posts} posts planned | "
             f"{schedule[0].strftime('%H:%M')} → {schedule[-1].strftime('%H:%M')} UTC")
             
    with state_lock:
        bot_state["schedule"] = [
            {
                "time": t.isoformat() + "Z",
                "status": "Pending",
                "coin": "Pending",
                "type": "Pending"
            }
            for t in schedule
        ]
        bot_state["status_message"] = f"Scheduled {n_posts} posts for today."
        
    for idx, sched_time in enumerate(schedule):
        # Refresh global data every N posts
        if idx > 0 and idx % DATA_REFRESH_EVERY == 0:
            log.info("🔄 Refreshing market data + news...")
            with state_lock:
                bot_state["status_message"] = "Refreshing market data & news..."
            global_data = _refresh_global(global_pipeline)
            news = global_pipeline.fetch_news()
            
        # Wait for scheduled time in small increments
        while True:
            with state_lock:
                if not bot_state["is_running"]:
                    log.info("Scheduler received shutdown signal. Exiting session.")
                    return
            
            now = datetime.utcnow()
            wait = (sched_time - now).total_seconds()
            if wait <= 0:
                break
                
            with state_lock:
                bot_state["status_message"] = f"Waiting {fmt_sec(int(wait))} for post {idx+1}/{n_posts} at {sched_time.strftime('%H:%M')} UTC..."
            time.sleep(min(1, wait))
            
        # Select signal + coin
        with state_lock:
            bot_state["status_message"] = f"Selecting signal for post {idx+1}/{n_posts}..."
            bot_state["schedule"][idx]["status"] = "Selecting signal"
            
        result = global_selector.select(global_data, news, bot_state["recent_coins"], bot_state["recent_types"])
        if not result:
            log.warning("  No signal found — skipping slot.")
            with state_lock:
                bot_state["schedule"][idx]["status"] = "Skipped"
                bot_state["schedule"][idx]["coin"] = "None"
                bot_state["schedule"][idx]["type"] = "None"
            continue
            
        analysis, post_type = result
        coin = analysis["coin"]
        log.info(f"🎯 Signal: [{post_type}] on {coin['tag']} (RSI={analysis['rsi']}, MACD={analysis['macd']['crossover']})")
        
        with state_lock:
            bot_state["schedule"][idx]["coin"] = coin["tag"]
            bot_state["schedule"][idx]["type"] = post_type
            bot_state["schedule"][idx]["status"] = "Generating content..."
            bot_state["status_message"] = f"Generating content for {coin['tag']} ({post_type})..."
            
        # Generate text
        content = ""
        try:
            coin_news = global_pipeline.fetch_news(currency=coin["cp"])
            all_news = news + [n for n in coin_news if n not in news]
            
            content = global_rotator.generate_content_with_retry(
                post_type, analysis, global_data, all_news,
                bot_state["recent_coins"], bot_state["recent_types"]
            )
            
            words = len(content.split())
            if words < 10:
                log.warning(f"  Post too short ({words}w), skipping.")
                with state_lock:
                    bot_state["schedule"][idx]["status"] = "Failed (Too short)"
                continue
            if words > 200:
                content = " ".join(content.split()[:180]) + "..."
        except Exception as e:
            log.error(f"  Content generation failed: {e}")
            with state_lock:
                bot_state["schedule"][idx]["status"] = "Failed (Gen error)"
                bot_state["posts_failed"] += 1
            continue
            
        # Generate and upload chart
        image_urls = []
        if random.random() < CHART_PROBABILITY:
            log.info(f"  📈 Generating chart for {coin['tag']}...")
            with state_lock:
                bot_state["schedule"][idx]["status"] = "Generating chart..."
            chart_bytes = global_charts.generate(analysis)
            if chart_bytes:
                cdn = global_uploader.upload(chart_bytes)
                if cdn:
                    image_urls.append(cdn)
                    
        # Publish
        try:
            log.info(f"📤 Publishing {'+ chart' if image_urls else '(text only)'}...")
            with state_lock:
                bot_state["schedule"][idx]["status"] = "Publishing..."
                
            result_pub = publish(content, image_urls or None)
            code = result_pub.get("code", "")
            
            if code == "000000":
                post_id = result_pub.get("data", {}).get("id", "unknown")
                url = f"https://www.binance.com/square/post/{post_id}"
                log.info(f"  ✅ #{idx+1} [{post_type}] {coin['tag']} → {url}")
                
                with state_lock:
                    bot_state["posts_published"] += 1
                    bot_state["schedule"][idx]["status"] = "Published"
                    bot_state["recent_posts"].append({
                        "time": datetime.utcnow().isoformat() + "Z",
                        "content": content,
                        "status": "Success",
                        "url": url
                    })
                    if len(bot_state["recent_posts"]) > 20:
                        bot_state["recent_posts"].pop(0)
                        
                    bot_state["recent_coins"].append(coin["tag"])
                    bot_state["recent_types"].append(post_type)
                    if len(bot_state["recent_coins"]) > 12: bot_state["recent_coins"].pop(0)
                    if len(bot_state["recent_types"]) > 8:  bot_state["recent_types"].pop(0)
            else:
                msg = result_pub.get("message", "no message")
                log.warning(f"  ⚠ Rejected: {code} | {msg}")
                with state_lock:
                    bot_state["posts_failed"] += 1
                    bot_state["schedule"][idx]["status"] = f"Rejected: {code}"
                    bot_state["recent_posts"].append({
                        "time": datetime.utcnow().isoformat() + "Z",
                        "content": content,
                        "status": f"Rejected ({code})",
                        "url": ""
                    })
                    if len(bot_state["recent_posts"]) > 20:
                        bot_state["recent_posts"].pop(0)
                        
                if code in ("10001", "20001"):
                    log.error("  Fatal: Invalid API key.")
                    with state_lock:
                        bot_state["error_message"] = "Invalid Binance API key"
                    return
                elif code == "40003":
                    log.warning("  Daily post limit reached.")
                    with state_lock:
                        bot_state["error_message"] = "Daily post limit reached"
                    break
                elif code == "30001":
                    log.error("  Account banned.")
                    with state_lock:
                        bot_state["error_message"] = "Account banned"
                    return
        except Exception as e:
            log.error(f"  Publishing failed: {e}")
            with state_lock:
                bot_state["posts_failed"] += 1
                bot_state["schedule"][idx]["status"] = "Failed (Pub error)"
                bot_state["recent_posts"].append({
                    "time": datetime.utcnow().isoformat() + "Z",
                    "content": content,
                    "status": "Failed",
                    "url": ""
                })
                if len(bot_state["recent_posts"]) > 20:
                    bot_state["recent_posts"].pop(0)
                    
    log.info(f"\n🏁 Session complete — {bot_state['posts_published']} posts published.")
    _print_summary(global_selector)


def _refresh_global(pipeline: DataPipeline) -> dict:
    time.sleep(0.5)
    market   = pipeline.fetch_market_data()
    fg       = pipeline.fetch_fear_greed()
    trending = pipeline.fetch_trending()

    # Fetch OI and funding for top coins
    oi_data      = {}
    funding_data = {}
    for coin in COINS[:8]:   # top 8 only to avoid rate limits
        try:
            oi = pipeline.fetch_open_interest(coin["fsym"])
            if oi:
                oi_data[coin["sym"]] = oi
            fr = pipeline.fetch_funding_rate(coin["fsym"])
            if fr is not None:
                funding_data[coin["sym"]] = fr
            time.sleep(0.1)
        except:
            pass

    log.info(f"  🌐 Global data: {len(market)} coins | F&G={fg['value']} ({fg['label']}) "
             f"| Trending: {trending[:3]}")
    return {
        "market":     market,
        "fg":         fg,
        "trending":   trending,
        "oi":         oi_data,
        "funding":    funding_data,
        "fetched_at": datetime.utcnow().strftime("%H:%M UTC"),
    }


def _print_summary(selector: SignalSelector):
    log.info("\n📊 Post Type Distribution Today:")
    for k, v in sorted(selector._type_count.items(), key=lambda x: -x[1]):
        if v > 0:
            bar = "█" * v
            log.info(f"   {k:<22} {bar} ({v})")


# ─────────────────────────────────────────────────────────────────────────────
# FLASK WEB APP & EMBEDDED DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

scheduler_started = False
scheduler_lock = threading.Lock()

@app.before_request
def start_scheduler_on_first_request():
    global scheduler_started
    if not scheduler_started:
        with scheduler_lock:
            if not scheduler_started:
                try:
                    start_background_thread()
                    scheduler_started = True
                    log.info("Background scheduler thread started via before_request in worker process.")
                except Exception as e:
                    log.error(f"Failed to start background thread: {e}")

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Binance Square Auto-Poster Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #070b19;
            --card-bg: rgba(13, 20, 38, 0.45);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-primary: #6366f1;
            --accent-secondary: #a855f7;
            --success: #10b981;
            --danger: #f43f5e;
            --warning: #f59e0b;
            --info: #3b82f6;
            --glow: rgba(99, 102, 241, 0.15);
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(168, 85, 247, 0.05) 0%, transparent 40%);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 2rem;
            line-height: 1.5;
        }

        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.15);
            box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.08);
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--card-border);
        }
        
        .logo-section h1 {
            font-size: 2.25rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
            letter-spacing: -0.5px;
        }
        
        .logo-section p {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }
        
        .status-badge-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            padding: 0.5rem 1rem;
            border-radius: 99px;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--success);
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--success);
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px var(--success);
        }
        
        .status-dot.pulsing {
            animation: pulse 1.5s infinite alternate;
        }
        
        @keyframes pulse {
            0% { transform: scale(0.9); opacity: 0.6; box-shadow: 0 0 4px var(--success); }
            100% { transform: scale(1.2); opacity: 1; box-shadow: 0 0 12px var(--success); }
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            padding: 1.5rem;
            position: relative;
            overflow: hidden;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--accent-primary);
        }
        
        .metric-card.success::before { background: var(--success); }
        .metric-card.danger::before { background: var(--danger); }
        .metric-card.warning::before { background: var(--warning); }
        
        .metric-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.2;
        }
        
        .metric-subtitle {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
        }

        .panels-grid {
            display: grid;
            grid-template-columns: 1.4fr 1fr;
            gap: 2rem;
        }
        
        @media (max-width: 1024px) {
            .panels-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .panel {
            padding: 1.75rem;
            display: flex;
            flex-direction: column;
            height: 600px;
        }
        
        .panel-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 0.75rem;
        }
        
        .timeline-container {
            overflow-y: auto;
            flex-grow: 1;
            padding-right: 0.5rem;
        }
        
        .timeline-item {
            display: grid;
            grid-template-columns: 100px 1fr auto;
            align-items: center;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid rgba(255, 255, 255, 0.03);
            margin-bottom: 0.75rem;
            font-size: 0.9rem;
            transition: all 0.2s ease;
        }
        
        .timeline-item:hover {
            background: rgba(255, 255, 255, 0.03);
            border-color: rgba(255, 255, 255, 0.06);
        }
        
        .timeline-time {
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
        }
        
        .timeline-coin-type {
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .badge {
            font-size: 0.75rem;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-weight: 600;
        }
        
        .badge.coin {
            background: rgba(99, 102, 241, 0.15);
            color: #818cf8;
            border: 1px solid rgba(99, 102, 241, 0.2);
        }
        
        .badge.type {
            background: rgba(168, 85, 247, 0.15);
            color: #c084fc;
            border: 1px solid rgba(168, 85, 247, 0.2);
        }
        
        .timeline-status {
            font-weight: 600;
        }
        
        .status-txt-pending { color: var(--warning); }
        .status-txt-published { color: var(--success); }
        .status-txt-failed { color: var(--danger); }
        .status-txt-processing { color: var(--info); }
        .status-txt-skipped { color: var(--text-secondary); }

        .console-logs {
            background: #02040a;
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            overflow-y: auto;
            flex-grow: 1;
            white-space: pre-wrap;
            color: #d0d7de;
        }
        
        .log-line {
            margin-bottom: 0.35rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.01);
            padding-bottom: 0.15rem;
        }
        
        .log-info { color: #8b949e; }
        .log-warning { color: var(--warning); }
        .log-error { color: var(--danger); }
        .log-success { color: var(--success); }
        
        .posts-container {
            overflow-y: auto;
            flex-grow: 1;
            padding-right: 0.5rem;
        }
        
        .post-card {
            padding: 1rem;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.015);
            border: 1px solid rgba(255, 255, 255, 0.04);
            margin-bottom: 1rem;
            font-size: 0.88rem;
        }
        
        .post-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            color: var(--text-secondary);
            font-size: 0.8rem;
        }
        
        .post-content {
            margin-bottom: 0.75rem;
            color: var(--text-primary);
            line-height: 1.4;
        }
        
        .post-link {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            color: var(--accent-primary);
            text-decoration: none;
            font-weight: 600;
            transition: color 0.2s ease;
        }
        
        .post-link:hover {
            color: var(--accent-secondary);
            text-decoration: underline;
        }
        
        .btn-post-now {
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            border: none;
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
        }
        
        .btn-post-now:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4);
        }
        
        .btn-post-now:active:not(:disabled) {
            transform: translateY(0);
        }
        
        .btn-post-now:disabled {
            background: #21262d;
            color: var(--text-secondary);
            cursor: not-allowed;
            box-shadow: none;
        }
        
        .spinner {
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 1s infinite linear;
            display: none;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .tab-buttons {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            background: rgba(255, 255, 255, 0.02);
            padding: 0.25rem;
            border-radius: 8px;
            border: 1px solid var(--card-border);
        }
        
        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            padding: 0.4rem 1rem;
            border-radius: 6px;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 0.85rem;
        }
        
        .tab-btn.active {
            background: var(--card-bg);
            color: var(--text-primary);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: var(--card-border);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.15);
        }
        
        .tab-content {
            display: none;
            flex-direction: column;
            height: calc(100% - 40px);
        }
        .tab-content.active {
            display: flex;
        }
        
        .system-status-msg {
            font-size: 0.9rem;
            color: var(--text-secondary);
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--card-border);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .banner-alert {
            display: none;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        .banner-alert.success {
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }
        .banner-alert.danger {
            background: rgba(244, 63, 94, 0.1);
            color: var(--danger);
            border: 1px solid rgba(244, 63, 94, 0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-section">
                <h1>Binance Square Auto-Poster</h1>
                <p>AI Signal-Driven Market Analysis & Automated Publishing</p>
            </div>
            <div class="status-badge-header" id="status-header">
                <span class="status-dot pulsing" id="status-dot"></span>
                <span id="status-text">Active Scheduler</span>
            </div>
        </header>

        <div class="banner-alert" id="toast-banner"></div>

        <div class="system-status-msg" id="status-msg-container">
            <span style="color: var(--accent-primary);">⚡</span>
            <span id="bot-status-message">Loading system state...</span>
        </div>

        <div class="metrics-grid">
            <div class="glass-card metric-card">
                <div class="metric-title">Uptime</div>
                <div class="metric-value" id="bot-uptime">--:--:--</div>
                <div class="metric-subtitle" id="bot-start-time">Started: --</div>
            </div>
            <div class="glass-card metric-card success">
                <div class="metric-title">Progress (Published)</div>
                <div class="metric-value" id="bot-published">0</div>
                <div class="metric-subtitle" id="bot-progress-percent">0% of scheduled targets</div>
            </div>
            <div class="glass-card metric-card danger">
                <div class="metric-title">Failed Posts</div>
                <div class="metric-value" id="bot-failed">0</div>
                <div class="metric-subtitle">Failed api calls or validation</div>
            </div>
            <div class="glass-card metric-card warning">
                <div class="metric-title">Countdown to Next Post</div>
                <div class="metric-value" id="bot-countdown">--:--</div>
                <div class="metric-subtitle" id="bot-next-post-time">Next run: None</div>
            </div>
        </div>

        <div class="panels-grid">
            <div class="glass-card panel">
                <div class="panel-title">
                    <span>Activity Dashboard</span>
                    <button class="btn-post-now" id="btn-post-now" onclick="triggerPostNow()">
                        <span class="spinner" id="btn-spinner"></span>
                        <span id="btn-text">Post Now</span>
                    </button>
                </div>
                
                <div class="tab-buttons">
                    <button class="tab-btn active" onclick="switchTab('left-panel', 'tab-schedule', this)">Schedule Timeline</button>
                    <button class="tab-btn" onclick="switchTab('left-panel', 'tab-posts', this)">Recent Published Posts</button>
                    <button class="tab-btn" onclick="switchTab('left-panel', 'tab-news-queue', this)">Pushed News Queue</button>
                </div>

                <div class="tab-content active" id="tab-schedule">
                    <div class="timeline-container" id="timeline-list">
                        <div style="color: var(--text-secondary); text-align: center; padding: 2rem;">No items scheduled.</div>
                    </div>
                </div>

                <div class="tab-content" id="tab-posts">
                    <div class="posts-container" id="posts-list">
                        <div style="color: var(--text-secondary); text-align: center; padding: 2rem;">No posts published yet.</div>
                    </div>
                </div>

                <div class="tab-content" id="tab-news-queue">
                    <div class="posts-container" id="news-queue-list">
                        <div style="color: var(--text-secondary); text-align: center; padding: 2rem;">No news items currently queued.</div>
                    </div>
                </div>
            </div>

            <div class="glass-card panel">
                <div class="panel-title">
                    <span>Live Console Logs</span>
                </div>
                <div class="console-logs" id="console-logs-window">
                    Initializing console logs...
                </div>
            </div>
        </div>
    </div>

    <script>
        let botStartTime = null;
        let nextPendingPostTime = null;
        let statusPollInterval = null;
        let countdownInterval = null;
        
        function switchTab(panelId, tabId, btnEl) {
            const buttons = btnEl.parentNode.getElementsByClassName('tab-btn');
            for (let btn of buttons) {
                btn.classList.remove('active');
            }
            btnEl.classList.add('active');
            
            document.getElementById('tab-schedule').classList.remove('active');
            document.getElementById('tab-posts').classList.remove('active');
            document.getElementById('tab-news-queue').classList.remove('active');
            
            document.getElementById(tabId).classList.add('active');
        }
        
        function formatUptime(diffSeconds) {
            if (isNaN(diffSeconds) || diffSeconds < 0) return "--:--:--";
            const h = Math.floor(diffSeconds / 3600);
            const m = Math.floor((diffSeconds % 3600) / 60);
            const s = Math.floor(diffSeconds % 60);
            return [
                h.toString().padStart(2, '0'),
                m.toString().padStart(2, '0'),
                s.toString().padStart(2, '0')
            ].join(':');
        }
        
        function updateUptime() {
            if (!botStartTime) return;
            const now = new Date();
            const diffMs = now - botStartTime;
            const diffSec = Math.floor(diffMs / 1000);
            document.getElementById('bot-uptime').innerText = formatUptime(diffSec);
        }
        
        function updateCountdown() {
            if (!nextPendingPostTime) {
                document.getElementById('bot-countdown').innerText = "--:--";
                document.getElementById('bot-next-post-time').innerText = "Next run: None";
                return;
            }
            const now = new Date();
            const diffSec = Math.floor((nextPendingPostTime - now) / 1000);
            
            if (diffSec <= 0) {
                document.getElementById('bot-countdown').innerText = "00:00";
                return;
            }
            
            const m = Math.floor(diffSec / 60);
            const s = diffSec % 60;
            
            if (m >= 60) {
                const h = Math.floor(m / 60);
                const remM = m % 60;
                document.getElementById('bot-countdown').innerText = 
                    `${h.toString().padStart(2, '0')}:${remM.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
            } else {
                document.getElementById('bot-countdown').innerText = 
                    `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
            }
        }

        function showToast(message, isSuccess) {
            const banner = document.getElementById('toast-banner');
            banner.innerText = message;
            banner.className = 'banner-alert ' + (isSuccess ? 'success' : 'danger');
            banner.style.display = 'block';
            setTimeout(() => {
                banner.style.display = 'none';
            }, 6000);
        }

        function triggerPostNow() {
            const btn = document.getElementById('btn-post-now');
            const spinner = document.getElementById('btn-spinner');
            const btnText = document.getElementById('btn-text');
            
            btn.disabled = true;
            spinner.style.display = 'inline-block';
            btnText.innerText = 'Posting...';
            
            fetch('/api/post-now', {
                method: 'POST'
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    showToast(`Success! Post published to Binance Square: ${data.url}`, true);
                    pollStatus();
                } else {
                    showToast(`Failed: ${data.message || 'Unknown error'}`, false);
                }
            })
            .catch(err => {
                showToast(`Network Error: ${err.message || err}`, false);
            })
            .finally(() => {
                btn.disabled = false;
                spinner.style.display = 'none';
                btnText.innerText = 'Post Now';
            });
        }
        
        function pollStatus() {
            fetch('/api/status')
            .then(res => res.json())
            .then(state => {
                document.getElementById('bot-status-message').innerText = state.status_message || 'Active';
                
                if (state.start_time_utc) {
                    botStartTime = new Date(state.start_time_utc);
                    document.getElementById('bot-start-time').innerText = `Started: ${botStartTime.toLocaleTimeString()} (${botStartTime.toLocaleDateString()})`;
                }
                
                const pub = state.posts_published || 0;
                const fail = state.posts_failed || 0;
                const totalSched = state.schedule ? state.schedule.length : 0;
                
                document.getElementById('bot-published').innerText = pub;
                document.getElementById('bot-failed').innerText = fail;
                
                const pct = totalSched > 0 ? Math.round((pub / totalSched) * 100) : 0;
                document.getElementById('bot-progress-percent').innerText = `${pct}% of ${totalSched} scheduled targets`;
                
                const statusHeader = document.getElementById('status-header');
                const statusDot = document.getElementById('status-dot');
                const statusText = document.getElementById('status-text');
                
                if (state.error_message) {
                    statusHeader.style.borderColor = 'rgba(244, 63, 94, 0.2)';
                    statusHeader.style.background = 'rgba(244, 63, 94, 0.1)';
                    statusHeader.style.color = 'var(--danger)';
                    statusDot.className = 'status-dot';
                    statusDot.style.backgroundColor = 'var(--danger)';
                    statusDot.style.boxShadow = 'none';
                    statusText.innerText = 'Error Status';
                    document.getElementById('bot-status-message').innerHTML = `<strong style="color: var(--danger)">Error:</strong> ${state.error_message}`;
                } else if (!state.is_running) {
                    statusHeader.style.borderColor = 'rgba(245, 158, 11, 0.2)';
                    statusHeader.style.background = 'rgba(245, 158, 11, 0.1)';
                    statusHeader.style.color = 'var(--warning)';
                    statusDot.className = 'status-dot';
                    statusDot.style.backgroundColor = 'var(--warning)';
                    statusDot.style.boxShadow = 'none';
                    statusText.innerText = 'Suspended';
                } else {
                    statusHeader.style.borderColor = 'rgba(16, 185, 129, 0.2)';
                    statusHeader.style.background = 'rgba(16, 185, 129, 0.1)';
                    statusHeader.style.color = 'var(--success)';
                    statusDot.className = 'status-dot pulsing';
                    statusDot.style.backgroundColor = 'var(--success)';
                    statusDot.style.boxShadow = '0 0 8px var(--success)';
                    statusText.innerText = 'Active Scheduler';
                }
                
                const timelineContainer = document.getElementById('timeline-list');
                if (state.schedule && state.schedule.length > 0) {
                    let html = '';
                    nextPendingPostTime = null;
                    
                    state.schedule.forEach(item => {
                        const itemTime = new Date(item.time);
                        const displayTime = itemTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                        
                        let statusClass = 'status-txt-pending';
                        if (item.status === 'Published') statusClass = 'status-txt-published';
                        else if (item.status.includes('Failed') || item.status.includes('Rejected')) statusClass = 'status-txt-failed';
                        else if (item.status.includes('Generating') || item.status.includes('Publishing') || item.status.includes('Selecting')) statusClass = 'status-txt-processing';
                        else if (item.status === 'Skipped') statusClass = 'status-txt-skipped';
                        
                        if (item.status === 'Pending' && !nextPendingPostTime) {
                            nextPendingPostTime = itemTime;
                            document.getElementById('bot-next-post-time').innerText = `Next run: ${displayTime}`;
                        }
                        
                        html += `
                            <div class="timeline-item">
                                <div class="timeline-time">${displayTime}</div>
                                <div class="timeline-coin-type">
                                    <span class="badge coin">${item.coin}</span>
                                    <span class="badge type">${item.type}</span>
                                </div>
                                <div class="timeline-status ${statusClass}">${item.status}</div>
                            </div>
                        `;
                    });
                    timelineContainer.innerHTML = html;
                } else {
                    timelineContainer.innerHTML = '<div style="color: var(--text-secondary); text-align: center; padding: 2rem;">No items scheduled.</div>';
                    nextPendingPostTime = null;
                }
                
                const postsContainer = document.getElementById('posts-list');
                if (state.recent_posts && state.recent_posts.length > 0) {
                    let html = '';
                    const reversedPosts = [...state.recent_posts].reverse();
                    reversedPosts.forEach(post => {
                        const postTime = new Date(post.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                        const linkHtml = post.url ? `<a href="${post.url}" target="_blank" class="post-link">View on Binance Square ↗</a>` : `<span style="color: var(--danger); font-weight:600;">Publish Error (${post.status})</span>`;
                        
                        html += `
                            <div class="post-card">
                                <div class="post-header">
                                    <span>Time: ${postTime}</span>
                                    <span>Status: ${post.status}</span>
                                </div>
                                <div class="post-content">${post.content}</div>
                                <div>${linkHtml}</div>
                            </div>
                        `;
                    });
                    postsContainer.innerHTML = html;
                } else {
                    postsContainer.innerHTML = '<div style="color: var(--text-secondary); text-align: center; padding: 2rem;">No posts published yet.</div>';
                }
                
                const newsContainer = document.getElementById('news-queue-list');
                if (state.pushed_news_queue && state.pushed_news_queue.length > 0) {
                    let html = '';
                    state.pushed_news_queue.forEach(item => {
                        const itemTime = item.published ? new Date(item.published).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : 'Pending';
                        html += `
                            <div class="post-card">
                                <div class="post-header">
                                    <span>Source: ${item.source}</span>
                                    <span>Time: ${itemTime}</span>
                                </div>
                                <div class="post-content" style="font-weight: 600;">${item.title}</div>
                            </div>
                        `;
                    });
                    newsContainer.innerHTML = html;
                } else {
                    newsContainer.innerHTML = '<div style="color: var(--text-secondary); text-align: center; padding: 2rem;">No news items currently queued.</div>';
                }
                
                const logsContainer = document.getElementById('console-logs-window');
                if (state.logs && state.logs.length > 0) {
                    let logHtml = '';
                    const wasScrolledToBottom = logsContainer.scrollHeight - logsContainer.clientHeight <= logsContainer.scrollTop + 20;
                    
                    state.logs.forEach(log => {
                        let styleClass = 'log-info';
                        if (log.includes('[WARNING]') || log.includes('⚠')) styleClass = 'log-warning';
                        else if (log.includes('[ERROR]') || log.includes('❌')) styleClass = 'log-error';
                        else if (log.includes('✅')) styleClass = 'log-success';
                        
                        const escapedLog = log.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                        logHtml += `<div class="log-line ${styleClass}">${escapedLog}</div>`;
                    });
                    logsContainer.innerHTML = logHtml;
                    
                    if (wasScrolledToBottom) {
                        logsContainer.scrollTop = logsContainer.scrollHeight;
                    }
                } else {
                    logsContainer.innerHTML = 'Initializing logs window...';
                }
            })
            .catch(err => {
                console.error("Polled error:", err);
                document.getElementById('bot-status-message').innerHTML = '<strong style="color: var(--danger)">Connection Error:</strong> Disconnected from bot server.';
            });
        }
        
        pollStatus();
        statusPollInterval = setInterval(pollStatus, 3000);
        
        countdownInterval = setInterval(() => {
            updateUptime();
            updateCountdown();
        }, 1000);
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return DASHBOARD_HTML

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

@app.route("/api/status", methods=["GET"])
def api_status():
    with state_lock:
        state = dict(bot_state)
    news_list = []
    if global_pipeline:
        with global_pipeline.pushed_news_lock:
            for item in global_pipeline.pushed_news:
                pub_dt = item.get("pub_dt")
                news_list.append({
                    "title": item.get("title", ""),
                    "source": item.get("source", "Unknown"),
                    "published": pub_dt.isoformat() if pub_dt else None
                })
    state["pushed_news_queue"] = news_list
    return jsonify(state)

@app.route("/api/post-now", methods=["POST"])
def post_now():
    try:
        url = trigger_single_post()
        return jsonify({
            "status": "success",
            "url": url
        })
    except Exception as e:
        log.exception("Error in manual post-now endpoint")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


def clean_source_name(src: str) -> str:
    if not src:
        return "Unknown"
    src = src.strip()
    # Handle social media URLs
    if src.startswith("http://") or src.startswith("https://"):
        parts = src.replace("https://", "").replace("http://", "").split("/")
        if len(parts) > 1 and ("twitter.com" in parts[0] or "x.com" in parts[0]):
            return parts[1].split("?")[0].strip()
            
    # Clean standard domain suffixes and prefixes
    src = src.replace("https://", "").replace("http://", "")
    src = src.split("/")[0]
    for suffix in (".com", ".org", ".net", ".co", ".io", ".xyz", ".info"):
        if src.lower().endswith(suffix):
            src = src[:-len(suffix)]
    src = src.replace("@", "")
    return src.strip()

NEWS_SYSTEM_PROMPT = """You are an AI assistant that formats cryptocurrency news posts and searches the web for a relevant, publicly accessible image URL to accompany the news.
Your task is to:
1. Generate the post body text. The post body text MUST consist of the news title followed by the news source. You are allowed to make minor corrections to the title to fix grammatical mistakes or writing/spelling errors if necessary, but keep the overall content and meaning the same.
2. You MUST add relevant coin tags (e.g., $BTC, $ETH, $SOL) at the end of the post if there are no coin tags or coin names mentioned in the title/content. If the title does not mention a coin tag, identify the main coin associated with the news and add its tag (e.g. $BTC).
3. You MUST add exactly 1 to 2 relevant hashtags (e.g., #Bitcoin, #Crypto) at the end of the post.
4. Search the web for a direct, hotlinkable image URL (ending in .png, .jpg, .jpeg, or .webp) that is highly relevant to the news article.
5. The image URL must be a real, direct URL. Do NOT return html pages, and do NOT return local/relative URLs. If no high-quality, direct image URL can be found, return empty string for 'image_url'.

Output your response ONLY as a JSON block wrapped in ```json ... ```:
{
  "content": "The news title (with minor grammar corrections if needed)\\n\\nSource: CleanSourceName\\n\\n#hashtags $COINS",
  "image_url": "The direct image URL"
}"""

def generate_news_post(client: genai.Client, title: str, source: str) -> tuple[str, str]:
    prompt = f"News Title: {title}\nNews Source: {source}\n\nFormat this news post and find a relevant image URL using Google Search."
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=NEWS_SYSTEM_PROMPT,
                tools=[{"google_search": {}}],
                temperature=0.2,
            )
        )
        text = resp.text.strip()
        # Parse JSON from model response
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            res = json.loads(match.group(1).strip())
        else:
            match_braces = re.search(r"(\{.*\})", text, re.DOTALL)
            if match_braces:
                res = json.loads(match_braces.group(1).strip())
            else:
                raise ValueError("No JSON block found in model response")
        return res["content"], res.get("image_url", "")
    except Exception as e:
        log.warning(f"Failed to generate news post via Gemini: {e}. Falling back to manual formatting.")
        content = f"{title}\n\nSource: {source}"
        return content, ""

def news_scheduler_worker(items: list, start_time: datetime):
    global last_posted_news_time
    n_items = len(items)
    if n_items == 0:
        return
        
    log.info(f"📅 Starting news scheduler thread for {n_items} items over the next 25 minutes.")
    
    total_seconds = 25 * 60
    if n_items == 1:
        interval_sec = 0
    else:
        interval_sec = total_seconds / (n_items - 1)
        
    for i, item in enumerate(items):
        target_time = start_time + timedelta(seconds=i * interval_sec)
        
        while True:
            if news_cancel_event.is_set():
                log.info("🚫 News scheduler thread cancelled.")
                return
                
            now = datetime.utcnow()
            wait = (target_time - now).total_seconds()
            if wait <= 0:
                break
            time.sleep(min(1, wait))
            
        log.info(f"📰 News Scheduler: Posting item {i+1}/{n_items}: '{item['title']}'")
        try:
            title = item.get("title", "")
            if item.get("final_content"):
                post_content = item["final_content"]
                image_url = item.get("image_url", "")
                log.info("  ✨ Using pre-generated news content and image URL from intake evaluation.")
            else:
                raw_source = item.get("source", "Unknown")
                clean_src = clean_source_name(raw_source)
                post_content, image_url = generate_news_post(global_rotator.get_client(), title, clean_src)
            
            image_urls = []
            image_uploaded = False
            
            if image_url:
                log.info(f"  📥 Attempting to download Gemini recommended image from: {image_url}")
                try:
                    r = requests.get(image_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                    if r.status_code == 200:
                        cdn = global_uploader.upload(r.content)
                        if cdn:
                            image_urls.append(cdn)
                            image_uploaded = True
                            log.info(f"  ✅ Successfully uploaded Gemini recommended image: {cdn}")
                except Exception as e:
                    log.warning(f"  ⚠️ Failed to download/upload Gemini image: {e}")
            
            # Fallback to local TA chart generation if Gemini image wasn't successfully uploaded
            if not image_uploaded:
                mentioned_coin = None
                for coin in COINS:
                    if (coin["tag"].lower() in title.lower() or 
                        coin["cp"].lower() in title.lower() or 
                        coin["cg_id"].lower() in title.lower()):
                        mentioned_coin = coin
                        break
                
                if not mentioned_coin:
                    mentioned_coin = next((c for c in COINS if c["tag"] == "$BTC"), None)
                    
                if mentioned_coin:
                    log.info(f"  📈 Fallback: Generating TA chart for {mentioned_coin['tag']}...")
                    analysis = global_pipeline.analyse_coin(mentioned_coin)
                    if analysis:
                        chart_bytes = global_charts.generate(analysis)
                        if chart_bytes:
                            cdn = global_uploader.upload(chart_bytes)
                            if cdn:
                                image_urls.append(cdn)
                                
            log.info(f"📤 Publishing news post: {post_content[:60]}... {'with image' if image_urls else 'text-only'}")
            result_pub = publish(post_content, image_urls or None)
            code = result_pub.get("code", "")
            
            if code == "000000":
                post_id = result_pub.get("data", {}).get("id", "unknown")
                url = f"https://www.binance.com/square/post/{post_id}"
                log.info(f"  ✅ News Post Success → {url}")
                
                pub_dt = item.get("pub_dt")
                if pub_dt:
                    last_posted_news_time = pub_dt
                
                # Deduplication: record the cleaned title to prevent posting it again
                clean_title = item.get("title", "").strip().lower()
                if clean_title and clean_title not in posted_news_titles:
                    posted_news_titles.append(clean_title)
                    if len(posted_news_titles) > 500:
                        posted_news_titles.pop(0)
                save_news_state(last_posted_news_time, posted_news_titles)
                    
                with state_lock:
                    bot_state["posts_published"] += 1
                    bot_state["recent_posts"].append({
                        "time": datetime.utcnow().isoformat() + "Z",
                        "content": post_content,
                        "status": "Success",
                        "url": url
                    })
                    if len(bot_state["recent_posts"]) > 20:
                        bot_state["recent_posts"].pop(0)
            else:
                msg = result_pub.get("message", "Unknown error")
                log.error(f"  ❌ News Publish Failed: {msg} (code={code})")
                
        except Exception as e:
            log.exception(f"Error in news scheduler posting item: {e}")


def evaluate_and_clean_news_batch(news_items: list[dict]) -> list[dict]:
    if not news_items:
        return []
        
    global global_rotator
    if not global_rotator:
        log.warning("Gemini rotator not initialized. Skipping evaluation.")
        for item in news_items:
            item["importance"] = 1
            item["final_content"] = ""
            item["image_url"] = ""
        return news_items
        
    # Prepare input list
    input_data = []
    for idx, item in enumerate(news_items):
        input_data.append({
            "id": idx,
            "title": item.get("title", ""),
            "source": item.get("source", "")
        })
        
    prompt = f"""You are a cryptocurrency news analyst and content writer.
Your tasks are:
1. For each news article, assess its market importance for crypto traders on a scale of 1 to 30 (30 = critical market-moving event, 1 = low-value noise/unimportant).
   Determine importance based on:
   - Coin Name / Tag: Does it have a coin name mentioned? (This is the highest priority). You must identify the coin associated with the news. If a news item does not mention any cryptocurrency/coin and you cannot identify a relevant coin to tag it with, reject the news item by setting its importance to 0.
   - Actionability: Will someone decide to buy or sell the coin based on this news?
   - Price Fluctuation: Will the price of the coin fluctuate because of this news? The more the price is expected to change/fluctuate, the higher the importance score.
2. Identify the top 4 most important news items (importance > 0). If there is a tie, select the ones that are most market-moving.
3. For these top 4 news items only:
   - Search the web using the Google Search tool for a direct, hotlinkable image URL (ending in .png, .jpg, .jpeg, or .webp) that is highly relevant to the news article.
   - Correct any spelling, grammatical, or writing errors in the title.
   - Generate the final post body text ('final_content') consisting of the news title followed by the news source.
   - You MUST add relevant coin tags (e.g. $BTC, $ETH, $SOL) at the end of the content if they are not already in the title/content. If the title does not mention a coin tag, identify the main coin associated with the news and add its tag (e.g. $BTC).
   - You MUST add exactly 1 to 2 relevant hashtags (e.g. #Bitcoin, #Crypto) at the end of the content.
   - Format: "News Title\\n\\nSource: CleanSourceName\\n\\n#hashtags $COINS" (Clean the source name by removing suffixes like .com, @ prefix, etc.).

Return the processed articles in JSON format as a list of objects.

Input news items:
{json.dumps(input_data, indent=2)}

Output JSON block wrapped in ```json ... ```:
[
  {{
    "id": 0,
    "title": "Cleaned corrected title",
    "importance": 25,
    "final_content": "Cleaned corrected title\\n\\nSource: CleanSourceName\\n\\n#hashtags $COINS",
    "image_url": "https://example.com/image.jpg"
  }},
  ...
]"""

    # Call Gemini
    last_err = None
    for _ in range(len(global_rotator.clients)):
        try:
            client = global_rotator.get_client()
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    temperature=0.2,
                )
            )
            text = response.text
            
            # Extract JSON block
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            else:
                text = text.strip()
                
            results = json.loads(text)
            
            # Map back to original items
            evaluated_items = []
            for res in results:
                original_idx = res.get("id")
                if original_idx is not None and 0 <= original_idx < len(news_items):
                    orig = news_items[original_idx]
                    cleaned_item = dict(orig)
                    cleaned_item["title"] = res.get("title", orig.get("title"))
                    cleaned_item["importance"] = int(res.get("importance", 1))
                    cleaned_item["final_content"] = res.get("final_content", "")
                    cleaned_item["image_url"] = res.get("image_url", "")
                    evaluated_items.append(cleaned_item)
            return evaluated_items
            
        except Exception as e:
            log.error(f"❌ Gemini news batch evaluation failed on key index {global_rotator.current_idx}: {e}")
            last_err = e
            global_rotator.rotate()
            
    log.warning(f"⚠️ Failed to evaluate news batch via Gemini: {last_err}. Using original items with default importance.")
    for item in news_items:
        item["importance"] = 1
        item["final_content"] = ""
        item["image_url"] = ""
    return news_items


@app.route("/api/news", methods=["POST"])
def receive_news():
    global last_posted_news_time, news_schedule_thread, news_cancel_event
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid JSON format"}), 400
        
    if global_pipeline is None:
        return jsonify({"status": "error", "message": "Data pipeline is not initialized yet"}), 503
        
    items = data if isinstance(data, list) else [data]
    
    # 1. Filter only those news that came after the last news that was posted and not already published
    unprocessed_new_items = []
    for item in items:
        title = item.get("title")
        if not title:
            continue
            
        clean_t = title.strip().lower()
        if clean_t in posted_news_titles:
            continue
            
        published = item.get("published")
        pub_dt = parse_scraped_date(published)
        
        # Check against last_posted_news_time
        if last_posted_news_time is not None:
            if pub_dt <= last_posted_news_time:
                continue
        # Temporarily store pub_dt
        item_with_dt = dict(item)
        item_with_dt["_pub_dt"] = pub_dt
        unprocessed_new_items.append(item_with_dt)
        
    if not unprocessed_new_items:
        log.info(f"Received {len(items)} news items, but all of them are duplicates or older than the last posted news.")
        return jsonify({
            "status": "success",
            "message": "No new news items to schedule (all duplicates or older items)."
        })
        
    # Evaluate importance and fix minor writing/grammar mistakes using Gemini
    log.info(f"Analyzing and cleaning {len(unprocessed_new_items)} new news items via Gemini...")
    evaluated_items = evaluate_and_clean_news_batch(unprocessed_new_items)
    
    # Associate parsed pub_dt back to evaluated items
    for item, orig in zip(evaluated_items, unprocessed_new_items):
        item["pub_dt"] = orig["_pub_dt"]
        
    # Filter out items where importance is 0 (rejected by Gemini)
    evaluated_items = [item for item in evaluated_items if item.get("importance", 0) > 0]
    
    if not evaluated_items:
        log.info("All new news items were rejected because they did not mention or have relevance to any specific coin/cryptocurrency.")
        return jsonify({
            "status": "success",
            "message": "All news items were rejected (no coin mentions or relevance)."
        })
        
    # Sort items by importance descending (highest first) and then by pub_dt descending (most recent first)
    evaluated_items.sort(key=lambda x: (x.get("importance", 1), x["pub_dt"]), reverse=True)
    
    # 2. Count news posts published in the last 24 hours to enforce the rolling limit of 80
    now = datetime.utcnow()
    last_24h_posts = 0
    with state_lock:
        for post in bot_state.get("recent_posts", []):
            try:
                post_dt = datetime.fromisoformat(post["time"].replace("Z", ""))
                if (now - post_dt).total_seconds() < 86400:
                    last_24h_posts += 1
            except Exception as e:
                pass
                
    remaining_budget = max(0, 80 - last_24h_posts)
    if remaining_budget == 0:
        log.warning("⚠️ Daily limit of 80 news posts reached. Skipping scheduling for this batch.")
        return jsonify({
            "status": "success",
            "message": "Daily limit of 80 news posts reached. Pushed news skipped."
        })
        
    # Select the most important & recent ones, capping at min(remaining_budget, 2)
    max_to_schedule = min(remaining_budget, 2)
    selected_items = evaluated_items[:max_to_schedule]
    
    # Log the selection
    log.info(f"Selected {len(selected_items)} most important/recent news items to schedule (out of {len(evaluated_items)} candidates). remaining_budget={remaining_budget}")
    for idx, item in enumerate(selected_items):
        log.info(f"  [{idx+1}] Importance: {item.get('importance', 1)} | Title: {item.get('title')}")
        
    # 3. Before storing, clear the previous news and keep only the selected ones
    with global_pipeline.pushed_news_lock:
        global_pipeline.pushed_news.clear()
        for item in selected_items:
            global_pipeline.pushed_news.append(item)
            
    # 4. Schedule the selected news for next 25 mins
    with news_lock:
        # Cancel current running news scheduler thread if any
        if news_schedule_thread and news_schedule_thread.is_alive():
            news_cancel_event.set()
            news_schedule_thread.join(timeout=2)
        news_cancel_event.clear()
        
        # Prepare copies without internal prefix keys for scheduler worker
        worker_items = []
        for item in selected_items:
            item_copy = dict(item)
            if "_pub_dt" in item_copy:
                del item_copy["_pub_dt"]
            worker_items.append(item_copy)
            
        news_schedule_thread = threading.Thread(
            target=news_scheduler_worker,
            args=(worker_items, datetime.utcnow()),
            daemon=True
        )
        news_schedule_thread.start()
        
    return jsonify({
        "status": "success",
        "message": f"Successfully evaluated {len(evaluated_items)} items. Scheduled the top {len(selected_items)} most important/recent news items for the next 25 minutes."
    })


# ─────────────────────────────────────────────────────────────────────────────
# DAEMON BACKGROUND THREAD MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
worker_started = False
worker_lock = threading.Lock()

def background_worker():
    global bot_state
    log.info("Background worker thread started.")
    with state_lock:
        bot_state["start_time_utc"] = datetime.utcnow().isoformat() + "Z"
        
    while True:
        try:
            with state_lock:
                bot_state["is_running"] = True
                bot_state["error_message"] = None
            run_daily_session()
        except Exception as e:
            log.exception(f"Fatal error in background worker: {e}")
            with state_lock:
                bot_state["error_message"] = f"Fatal error: {str(e)}"
                bot_state["is_running"] = False
                
        log.info("Daily session ended. Sleeping 8 hours before starting a new session...")
        # Sleep in 1s increments so that if is_running becomes False we can exit
        for _ in range(8 * 3600):
            with state_lock:
                if not bot_state["is_running"]:
                    break
            time.sleep(1)

def start_background_thread():
    global worker_started
    with worker_lock:
        if not worker_started:
            t = threading.Thread(target=background_worker, daemon=True)
            t.start()
            worker_started = True
            log.info("Background thread spawned.")


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-START FOR WSGI / GUNICORN
# ─────────────────────────────────────────────────────────────────────────────
import sys
if "--cli" not in sys.argv:
    try:
        init_globals()
        # Defer start_background_thread() to app.before_request to ensure it runs inside the Gunicorn worker process.
    except Exception as e:
        log.warning(f"Could not auto-initialize globals on import: {e}. "
                    "Make sure environment variables are set before starting the server.")

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--cli" in sys.argv:
        # Run in standard CLI mode (single daily session loop)
        try:
            init_globals()
            run_daily_session()
        except KeyboardInterrupt:
            log.info("Shutdown requested by user. Exiting.")
        except Exception as e:
            log.exception(f"Fatal CLI execution error: {e}")
    else:
        # Run Flask web server
        try:
            port = int(os.getenv("PORT", 5000))
            log.info(f"Starting Flask server on port {port}...")
            app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
        except KeyboardInterrupt:
            log.info("Web server shutdown requested.")
        except Exception as e:
            log.exception(f"Fatal server execution error: {e}")
