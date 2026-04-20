# -*- coding: utf-8 -*-
"""
Sarada Trading System v3.1
Motore: Sarada v2.0 (ChatGPT) + timeframe multipli (D/W/M) + cloud GitHub Actions
Sviluppato da Jennifer, governante personale del signore.
"""

import os, io, json, time, warnings, base64, sys, importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import requests, numpy as np, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
#  AMBIENTE: locale vs GitHub Actions
# =============================================================================

IS_CLOUD = os.environ.get("GITHUB_ACTIONS") == "true" or "--single-run" in sys.argv

if IS_CLOUD:
    OUTPUT_DIR = Path("./docs")
    DATA_DIR   = Path("./data")
else:
    OUTPUT_DIR = Path(os.environ.get("SARADA_OUTPUT_DIR", r"C:\Sarada_Trading"))
    DATA_DIR   = OUTPUT_DIR / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

INTERVAL_MINUTES        = int(os.environ.get("SARADA_INTERVAL_MINUTES", "15"))
LOOKBACK_DAYS           = int(os.environ.get("SARADA_LOOKBACK_DAYS", "3000"))
MACRO_REFRESH_DAYS      = 7

DASHBOARD_FILE          = OUTPUT_DIR / "dashboard.html"
EXCEL_FILE              = OUTPUT_DIR / "sarada_dati.xlsx"
PRICE_HISTORY_FILE      = DATA_DIR  / "price_history.pkl"
SNAPSHOT_HISTORY_FILE   = DATA_DIR  / "snapshot_history.pkl"
MACRO_CACHE_FILE        = DATA_DIR  / "macro_cache.json"
HEATMAP_30              = DATA_DIR  / "heatmap_30d.png"
HEATMAP_60              = DATA_DIR  / "heatmap_60d.png"
HEATMAP_90              = DATA_DIR  / "heatmap_90d.png"

LAYER2_FILE             = Path(__file__).with_name("Sarada trading system operatività.py")
LAYER2_ALT_FILE         = Path(__file__).with_name("Sarada_trading_system_operativita.py")

FRED_API_KEY = os.environ.get("FRED_API_KEY", "").strip()

# =============================================================================
#  ASSET UNIVERSE
# =============================================================================

ASSETS = {
    # ── INDICI ────────────────────────────────────────────────────────────────
    "SP500":    {"ticker":"^GSPC",    "cat":"Indici",     "name":"S&P 500"},
    "NASDAQ":   {"ticker":"^IXIC",    "cat":"Indici",     "name":"Nasdaq Composite"},
    "DOW":      {"ticker":"^DJI",     "cat":"Indici",     "name":"Dow Jones"},
    "RUSSELL":  {"ticker":"^RUT",     "cat":"Indici",     "name":"Russell 2000"},
    "NIKKEI":   {"ticker":"^N225",    "cat":"Indici",     "name":"Nikkei 225"},
    "FTSE100":  {"ticker":"^FTSE",    "cat":"Indici",     "name":"FTSE 100"},
    "DAX":      {"ticker":"^GDAXI",   "cat":"Indici",     "name":"DAX"},
    "CAC40":    {"ticker":"^FCHI",    "cat":"Indici",     "name":"CAC 40"},
    "FTSEMIB":  {"ticker":"FTSEMIB.MI","cat":"Indici",   "name":"FTSE MIB"},
    "MSCIWORLD":{"ticker":"VT",       "cat":"Indici",     "name":"MSCI World ETF"},
    "EMERGING": {"ticker":"EEM",      "cat":"Indici",     "name":"MSCI Emerging ETF"},
    "ASIA_EX_JP":{"ticker":"AAXJ",   "cat":"Indici",     "name":"MSCI Asia ex Japan"},

    # ── ETF SETTORIALI USA ────────────────────────────────────────────────────
    "XLK":      {"ticker":"XLK",      "cat":"Settoriali", "name":"Technology"},
    "XLV":      {"ticker":"XLV",      "cat":"Settoriali", "name":"Healthcare"},
    "XLF":      {"ticker":"XLF",      "cat":"Settoriali", "name":"Financials"},
    "KBE":      {"ticker":"KBE",      "cat":"Settoriali", "name":"Banks"},
    "KIE":      {"ticker":"KIE",      "cat":"Settoriali", "name":"Insurance"},
    "ITA":      {"ticker":"ITA",      "cat":"Settoriali", "name":"Aerospace & Defense"},
    "XLE":      {"ticker":"XLE",      "cat":"Settoriali", "name":"Energy"},
    "XLU":      {"ticker":"XLU",      "cat":"Settoriali", "name":"Utilities"},
    "XLI":      {"ticker":"XLI",      "cat":"Settoriali", "name":"Industrials"},
    "XLY":      {"ticker":"XLY",      "cat":"Settoriali", "name":"Consumer Discr."},
    "XLP":      {"ticker":"XLP",      "cat":"Settoriali", "name":"Consumer Staples"},
    "XLB":      {"ticker":"XLB",      "cat":"Settoriali", "name":"Materials"},
    "XLRE":     {"ticker":"XLRE",     "cat":"Settoriali", "name":"Real Estate"},
    "XLC":      {"ticker":"XLC",      "cat":"Settoriali", "name":"Communication Svcs"},
    "SOXX":     {"ticker":"SOXX",     "cat":"Settoriali", "name":"Semiconductors"},
    "IBB":      {"ticker":"IBB",      "cat":"Settoriali", "name":"Biotech"},
    "ICLN":     {"ticker":"ICLN",     "cat":"Settoriali", "name":"Clean Energy"},
    "XME":      {"ticker":"XME",      "cat":"Settoriali", "name":"Metals & Mining"},
    "MOO":      {"ticker":"MOO",      "cat":"Settoriali", "name":"Agribusiness"},

    # ── SINGLE STOCKS USA ─────────────────────────────────────────────────────
    # Tech & Semis
    "AAPL":     {"ticker":"AAPL",     "cat":"Azioni USA", "name":"Apple"},
    "MSFT":     {"ticker":"MSFT",     "cat":"Azioni USA", "name":"Microsoft"},
    "NVDA":     {"ticker":"NVDA",     "cat":"Azioni USA", "name":"Nvidia"},
    "META":     {"ticker":"META",     "cat":"Azioni USA", "name":"Meta Platforms"},
    "GOOGL":    {"ticker":"GOOGL",    "cat":"Azioni USA", "name":"Alphabet"},
    "AMZN":     {"ticker":"AMZN",     "cat":"Azioni USA", "name":"Amazon"},
    "TSLA":     {"ticker":"TSLA",     "cat":"Azioni USA", "name":"Tesla"},
    "AMD":      {"ticker":"AMD",      "cat":"Azioni USA", "name":"AMD"},
    "AVGO":     {"ticker":"AVGO",     "cat":"Azioni USA", "name":"Broadcom"},
    "TSM":      {"ticker":"TSM",      "cat":"Azioni USA", "name":"TSMC"},
    # Finance
    "JPM":      {"ticker":"JPM",      "cat":"Azioni USA", "name":"JPMorgan Chase"},
    "GS":       {"ticker":"GS",       "cat":"Azioni USA", "name":"Goldman Sachs"},
    "BRK":      {"ticker":"BRK-B",    "cat":"Azioni USA", "name":"Berkshire Hathaway"},
    "BAC":      {"ticker":"BAC",      "cat":"Azioni USA", "name":"Bank of America"},
    "V":        {"ticker":"V",        "cat":"Azioni USA", "name":"Visa"},
    # Healthcare & Pharma
    "LLY":      {"ticker":"LLY",      "cat":"Azioni USA", "name":"Eli Lilly"},
    "JNJ":      {"ticker":"JNJ",      "cat":"Azioni USA", "name":"Johnson & Johnson"},
    "UNH":      {"ticker":"UNH",      "cat":"Azioni USA", "name":"UnitedHealth"},
    "PFE":      {"ticker":"PFE",      "cat":"Azioni USA", "name":"Pfizer"},
    # Energy
    "XOM":      {"ticker":"XOM",      "cat":"Azioni USA", "name":"ExxonMobil"},
    "CVX":      {"ticker":"CVX",      "cat":"Azioni USA", "name":"Chevron"},
    # Industrials / Defense
    "CAT":      {"ticker":"CAT",      "cat":"Azioni USA", "name":"Caterpillar"},
    "RTX":      {"ticker":"RTX",      "cat":"Azioni USA", "name":"RTX Corp"},
    "LMT":      {"ticker":"LMT",      "cat":"Azioni USA", "name":"Lockheed Martin"},
    # Consumer
    "MCD":      {"ticker":"MCD",      "cat":"Azioni USA", "name":"McDonald's"},
    "NKE":      {"ticker":"NKE",      "cat":"Azioni USA", "name":"Nike"},
    "SBUX":     {"ticker":"SBUX",     "cat":"Azioni USA", "name":"Starbucks"},

    # ── AZIONI ITALIANE ───────────────────────────────────────────────────────
    # Finanza / Banche
    "UCG":      {"ticker":"UCG.MI",   "cat":"Azioni IT", "name":"UniCredit"},
    "ISP":      {"ticker":"ISP.MI",   "cat":"Azioni IT", "name":"Intesa Sanpaolo"},
    "MB":       {"ticker":"MB.MI",    "cat":"Azioni IT", "name":"Mediobanca"},
    "G":        {"ticker":"G.MI",     "cat":"Azioni IT", "name":"Generali"},
    "NEXI":     {"ticker":"NEXI.MI",  "cat":"Azioni IT", "name":"Nexi"},
    "PST":      {"ticker":"PST.MI",   "cat":"Azioni IT", "name":"Poste Italiane"},
    # Energia / Utilities
    "ENEL":     {"ticker":"ENEL.MI",  "cat":"Azioni IT", "name":"Enel"},
    "TRN":      {"ticker":"TRN.MI",   "cat":"Azioni IT", "name":"Terna"},
    "SRG":      {"ticker":"SRG.MI",   "cat":"Azioni IT", "name":"Snam"},
    "IG_IT":    {"ticker":"IG.MI",    "cat":"Azioni IT", "name":"Italgas"},
    "ENI":      {"ticker":"ENI.MI",   "cat":"Azioni IT", "name":"Eni"},
    "SAIP":     {"ticker":"SAIP.MI",  "cat":"Azioni IT", "name":"Saipem"},
    # Industria / Difesa
    "PRY":      {"ticker":"PRY.MI",   "cat":"Azioni IT", "name":"Prysmian"},
    "LDO":      {"ticker":"LDO.MI",   "cat":"Azioni IT", "name":"Leonardo"},
    "IVG":      {"ticker":"IVG.MI",   "cat":"Azioni IT", "name":"Iveco Group"},
    "FNC":      {"ticker":"FNC.MI",   "cat":"Azioni IT", "name":"Fincantieri"},
    # Tech / TLC
    "STLAM":    {"ticker":"STLAM.MI", "cat":"Azioni IT", "name":"Stellantis"},
    "STM":      {"ticker":"STMPA.PA", "cat":"Azioni IT", "name":"STMicroelectronics"},
    "TIT":      {"ticker":"TIT.MI",   "cat":"Azioni IT", "name":"Telecom Italia"},
    "INW":      {"ticker":"INW.MI",   "cat":"Azioni IT", "name":"INWIT"},
    "REY":      {"ticker":"REY.MI",   "cat":"Azioni IT", "name":"Reply"},
    # Lusso / Consumer
    "MONC":     {"ticker":"MONC.MI",  "cat":"Azioni IT", "name":"Moncler"},
    "BC":       {"ticker":"BC.MI",    "cat":"Azioni IT", "name":"Brunello Cucinelli"},
    "CPR":      {"ticker":"CPR.MI",   "cat":"Azioni IT", "name":"Campari"},
    "RACE":     {"ticker":"RACE.MI",  "cat":"Azioni IT", "name":"Ferrari"},
    # Pharma / Biotech
    "REC":      {"ticker":"REC.MI",   "cat":"Azioni IT", "name":"Recordati"},
    "DIA":      {"ticker":"DIA.MI",   "cat":"Azioni IT", "name":"DiaSorin"},
    # Commodity / Industriali
    "TEN":      {"ticker":"TEN.MI",   "cat":"Azioni IT", "name":"Tenaris"},

    # ── COMMODITIES ──────────────────────────────────────────────────────────
    "WTI":      {"ticker":"CL=F",     "cat":"Commodities","name":"WTI Crude"},
    "BRENT":    {"ticker":"BZ=F",     "cat":"Commodities","name":"Brent Crude"},
    "GAS":      {"ticker":"NG=F",     "cat":"Commodities","name":"Natural Gas"},
    "GOLD":     {"ticker":"GC=F",     "cat":"Commodities","name":"Gold"},
    "SILVER":   {"ticker":"SI=F",     "cat":"Commodities","name":"Silver"},
    "PLATINUM": {"ticker":"PL=F",     "cat":"Commodities","name":"Platinum"},
    "PALLADIUM":{"ticker":"PA=F",     "cat":"Commodities","name":"Palladium"},
    "COPPER":   {"ticker":"HG=F",     "cat":"Commodities","name":"Copper"},
    "CORN":     {"ticker":"ZC=F",     "cat":"Commodities","name":"Corn"},
    "WHEAT":    {"ticker":"ZW=F",     "cat":"Commodities","name":"Wheat"},
    "SOYBEAN":  {"ticker":"ZS=F",     "cat":"Commodities","name":"Soybean"},
    "COFFEE":   {"ticker":"KC=F",     "cat":"Commodities","name":"Coffee"},
    "SUGAR":    {"ticker":"SB=F",     "cat":"Commodities","name":"Sugar"},
    "COTTON":   {"ticker":"CT=F",     "cat":"Commodities","name":"Cotton"},

    # ── CRYPTO ───────────────────────────────────────────────────────────────
    "BTC":      {"ticker":"BTC-USD",  "cat":"Crypto",     "name":"Bitcoin"},
    "ETH":      {"ticker":"ETH-USD",  "cat":"Crypto",     "name":"Ethereum"},
    "SOL":      {"ticker":"SOL-USD",  "cat":"Crypto",     "name":"Solana"},
    "ADA":      {"ticker":"ADA-USD",  "cat":"Crypto",     "name":"Cardano"},
    "DOT":      {"ticker":"DOT-USD",  "cat":"Crypto",     "name":"Polkadot"},
    "XRP":      {"ticker":"XRP-USD",  "cat":"Crypto",     "name":"XRP"},
    "AVAX":     {"ticker":"AVAX-USD", "cat":"Crypto",     "name":"Avalanche"},
    "LINK":     {"ticker":"LINK-USD", "cat":"Crypto",     "name":"Chainlink"},

    # ── FX / MACRO ────────────────────────────────────────────────────────────
    "DXY":      {"ticker":"DX-Y.NYB", "cat":"FX/Macro",   "name":"US Dollar Index"},
    "EURUSD":   {"ticker":"EURUSD=X", "cat":"FX/Macro",   "name":"EUR/USD"},
    "GBPUSD":   {"ticker":"GBPUSD=X", "cat":"FX/Macro",   "name":"GBP/USD"},
    "USDJPY":   {"ticker":"JPY=X",    "cat":"FX/Macro",   "name":"USD/JPY"},

    # ── BOND ─────────────────────────────────────────────────────────────────
    "US2Y":     {"ticker":"SHY",      "cat":"Bond",       "name":"US Treasury 1-3Y ETF"},
    "US10Y":    {"ticker":"IEF",      "cat":"Bond",       "name":"US Treasury 7-10Y ETF"},
    "US30Y":    {"ticker":"TLT",      "cat":"Bond",       "name":"US Treasury 20Y+ ETF"},
    "TIPS":     {"ticker":"TIP",      "cat":"Bond",       "name":"US TIPS ETF"},
    "HY":       {"ticker":"HYG",      "cat":"Bond",       "name":"High Yield Corp ETF"},
    "IG":       {"ticker":"LQD",      "cat":"Bond",       "name":"Investment Grade Corp ETF"},
}

# =============================================================================
#  UTILITY
# =============================================================================

def now_str(): return datetime.now().strftime("%d/%m/%Y %H:%M:%S")
def safe_float(x, default=np.nan):
    try:
        if x is None or x == "" or (isinstance(x, str) and x.strip() == "."): return default
        return float(x)
    except: return default
def clamp(x, lo, hi): return max(lo, min(hi, x))
def pct_fmt(x): return "—" if pd.isna(x) else f"{x:+.2f}%"
def num_fmt(x):
    if pd.isna(x): return "—"
    if abs(x) >= 1000: return f"{x:,.0f}"
    if abs(x) >= 10:   return f"{x:.2f}"
    return f"{x:.4f}"

def normalize_prob_dict(d):
    total = sum(max(v,0.0) for v in d.values())
    if total <= 0:
        n = len(d) if d else 1
        return {k: round(100.0/n,1) for k in d}
    out = {k: round(max(v,0.0)/total*100,1) for k,v in d.items()}
    drift = round(100.0-sum(out.values()),1)
    if out: first=next(iter(out)); out[first]=round(out[first]+drift,1)
    return out

# =============================================================================
#  INDICATORI TECNICI
# =============================================================================

def sma(s,n): return s.rolling(n).mean()
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    delta=s.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    ag=gain.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    al=loss.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    rs=ag/al.replace(0,np.nan)
    return 100-(100/(1+rs))
def macd(s,fast=12,slow=26,signal=9):
    ml=ema(s,fast)-ema(s,slow); sl=ema(ml,signal); hist=ml-sl
    return ml,sl,hist
def bollinger(s,n=20,k=2):
    m=sma(s,n); std=s.rolling(n).std(); upper=m+k*std; lower=m-k*std
    bw=(upper-lower)/m.replace(0,np.nan); pb=(s-lower)/(upper-lower).replace(0,np.nan)
    return upper,lower,bw,pb
def atr(high,low,close,n=14):
    pc=close.shift(1)
    tr=pd.concat([high-low,(high-pc).abs(),(low-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()
def adx(high,low,close,n=14):
    up=high.diff(); down=-low.diff()
    plus_dm=pd.Series(np.where((up>down)&(up>0),up,0.0),index=high.index)
    minus_dm=pd.Series(np.where((down>up)&(down>0),down,0.0),index=high.index)
    tr=pd.concat([high-low,(high-close.shift(1)).abs(),(low-close.shift(1)).abs()],axis=1).max(axis=1)
    atr_n=tr.rolling(n).mean().replace(0,np.nan)
    plus_di=100*(plus_dm.rolling(n).mean()/atr_n)
    minus_di=100*(minus_dm.rolling(n).mean()/atr_n)
    dx=100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,np.nan)
    return dx.rolling(n).mean(),plus_di,minus_di
def roc(s,n=10): return (s/s.shift(n)-1)*100
def stochastic(high,low,close,n=14,d=3):
    lowest=low.rolling(n).min(); highest=high.rolling(n).max()
    k=((close-lowest)/(highest-lowest).replace(0,np.nan))*100
    return k,k.rolling(d).mean()
def williams_r(high,low,close,n=14):
    hh=high.rolling(n).max(); ll=low.rolling(n).min()
    return -100*(hh-close)/(hh-ll).replace(0,np.nan)
def cci(high,low,close,n=20):
    tp=(high+low+close)/3; ma=tp.rolling(n).mean()
    md=(tp-ma).abs().rolling(n).mean()
    return (tp-ma)/(0.015*md.replace(0,np.nan))

def donchian(high, low, n=20):
    return high.rolling(n).max(), low.rolling(n).min()

def linear_reg_slope(series, n=20):
    s = series.dropna().tail(n)
    if len(s) < max(8, n//2): return np.nan
    y = s.values.astype(float)
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    denom = np.nanmean(np.abs(y)) if np.nanmean(np.abs(y)) else 1.0
    return (slope / denom) * 100

def zscore(series, n=20):
    m = series.rolling(n).mean()
    sd = series.rolling(n).std().replace(0, np.nan)
    return (series - m) / sd

def support_resistance_levels(high, low, lookbacks=(20,50,100)):
    out={}
    for lb in lookbacks:
        if len(high) >= lb:
            out[f"Res_{lb}"] = float(high.rolling(lb).max().iloc[-1])
            out[f"Sup_{lb}"] = float(low.rolling(lb).min().iloc[-1])
        else:
            out[f"Res_{lb}"] = np.nan
            out[f"Sup_{lb}"] = np.nan
    return out

def ichimoku_base_conversion(high, low):
    conv = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2
    return conv, base

def fibonacci_levels(close, lookback=90):
    s = close.dropna().tail(lookback)
    if len(s) < 20: return {}
    hi, lo = s.max(), s.min(); diff = hi - lo
    return {"Fib 23.6":hi-diff*0.236,"Fib 38.2":hi-diff*0.382,"Fib 50.0":hi-diff*0.5,"Fib 61.8":hi-diff*0.618,"Fib 78.6":hi-diff*0.786}


def detect_candles(df):
    if len(df) < 5: return []
    out=[]
    o1,h1,l1,c1 = df["Open"].iloc[-1],df["High"].iloc[-1],df["Low"].iloc[-1],df["Close"].iloc[-1]
    o2,c2 = df["Open"].iloc[-2],df["Close"].iloc[-2]
    o3,c3 = df["Open"].iloc[-3],df["Close"].iloc[-3]
    body = abs(c1-o1); rng = max(h1-l1,1e-9); upper = h1-max(o1,c1); lower=min(o1,c1)-l1

    if body/rng < 0.12: out.append("Doji")
    if lower > 2*body and upper < body: out.append("Hammer")
    if upper > 2*body and lower < body: out.append("Shooting Star")
    if c1 > o1 and c2 < o2 and c1 >= o2 and o1 <= c2: out.append("Bullish Engulfing")
    if c1 < o1 and c2 > o2 and c1 <= o2 and o1 >= c2: out.append("Bearish Engulfing")
    if c2 < o2 and c1 > o1 and c1 > (o2+c2)/2 and o1 < c2: out.append("Piercing Pattern")
    if c2 > o2 and c1 < o1 and c1 < (o2+c2)/2 and o1 > c2: out.append("Dark Cloud Cover")
    if c3 < o3 and abs(c2-o2)/max(df["High"].iloc[-2]-df["Low"].iloc[-2],1e-9) < 0.2 and c1 > o1 and c1 > (o3+c3)/2:
        out.append("Morning Star")
    if c3 > o3 and abs(c2-o2)/max(df["High"].iloc[-2]-df["Low"].iloc[-2],1e-9) < 0.2 and c1 < o1 and c1 < (o3+c3)/2:
        out.append("Evening Star")
    if c2 < o2 and c1 > o1 and o1 > c2 and c1 < o2: out.append("Bullish Harami")
    if c2 > o2 and c1 < o1 and o1 < c2 and c1 > o2: out.append("Bearish Harami")
    return list(dict.fromkeys(out))



def detect_patterns(close,high,low):
    out=[]
    if len(close)<30: return out
    last=close.iloc[-1]
    w20h=high.rolling(20).max(); w20l=low.rolling(20).min()
    w50h=high.rolling(50).max() if len(close) >= 50 else pd.Series(dtype=float)
    w50l=low.rolling(50).min() if len(close) >= 50 else pd.Series(dtype=float)
    if len(w20h.dropna())>1 and last>=w20h.iloc[-2]: out.append("Breakout 20d")
    if len(w20l.dropna())>1 and last<=w20l.iloc[-2]: out.append("Breakdown 20d")
    if isinstance(w50h, pd.Series) and len(w50h.dropna())>1 and last>=w50h.iloc[-2]: out.append("Breakout 50d")
    if isinstance(w50l, pd.Series) and len(w50l.dropna())>1 and last<=w50l.iloc[-2]: out.append("Breakdown 50d")
    recent=close.iloc[-40:]
    if len(recent)>=40:
        p1,p2=recent.iloc[:20].max(),recent.iloc[20:].max()
        t1,t2=recent.iloc[:20].min(),recent.iloc[20:].min()
        if abs(p1-p2)/max(abs(p1),1e-9)<0.02: out.append("Double Top")
        if abs(t1-t2)/max(abs(t1),1e-9)<0.02: out.append("Double Bottom")
    if len(high)>=3:
        if high.iloc[-1] < high.iloc[-2] and low.iloc[-1] > low.iloc[-2]:
            out.append("Inside Bar")
        if high.iloc[-1] > high.iloc[-2] and low.iloc[-1] < low.iloc[-2]:
            out.append("Outside Bar")
    if len(close)>=15:
        hh=high.tail(15).max(); ll=low.tail(15).min()
        if hh > ll:
            pos=(last-ll)/(hh-ll)
            if 0.42 <= pos <= 0.58 and linear_reg_slope(close, 15) > 0.08:
                out.append("Bull Flag / Consolidation")
            if 0.42 <= pos <= 0.58 and linear_reg_slope(close, 15) < -0.08:
                out.append("Bear Flag / Consolidation")
    bbz = zscore(close, 20)
    if len(bbz.dropna()) and abs(float(bbz.iloc[-1])) < 0.3 and len(close) >= 20:
        rng=(high.tail(10).max()-low.tail(10).min())/max(last,1e-9)*100
        if rng < 4.5:
            out.append("Volatility Squeeze")
    return list(dict.fromkeys(out))


# =============================================================================
#  TIMEFRAME RESAMPLING
#  Dato lo storico daily, ricava dati weekly e monthly
# =============================================================================

def resample_ohlcv(df, rule):
    """Ricampiona OHLCV daily in weekly ('W') o monthly ('ME')."""
    agg = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    cols = [c for c in agg if c in df.columns]
    agg_filtered = {k:v for k,v in agg.items() if k in cols}
    return df.resample(rule).agg(agg_filtered).dropna(subset=["Close"])

# =============================================================================
#  SCORE ENGINE — calcola score per un DataFrame OHLCV (qualsiasi TF)
# =============================================================================


def compute_score_for_df(df, macro, regime, corr60, code):
    """
    Calcola score, confidence e segnale su un DataFrame OHLCV.
    Usato per daily, weekly e monthly.
    """
    if len(df) < 6:
        return None

    close  = df["Close"].dropna()
    high   = df["High"].dropna()  if "High"   in df.columns else close
    low    = df["Low"].dropna()   if "Low"    in df.columns else close
    volume = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float)

    n = len(df)

    rsi_period  = min(14, max(3, n//3))
    ema_s       = min(20,  max(3, n//3))
    ema_m       = min(50,  max(5, n//2))
    ema_l       = min(200, max(8, n-2))
    bb_period   = min(20,  max(5, n//3))
    atr_period  = min(14,  max(3, n//4))
    adx_period  = min(14,  max(3, n//4))
    roc_period  = min(20,  max(2, n//4))
    stoch_period= min(14,  max(3, n//4))

    last     = float(close.iloc[-1])
    day_ret  = (close.iloc[-1]/close.iloc[-2]-1)*100  if len(close)>=2  else np.nan
    week_ret = (close.iloc[-1]/close.iloc[-6]-1)*100  if len(close)>=6  else np.nan
    month_ret= (close.iloc[-1]/close.iloc[-22]-1)*100 if len(close)>=22 else np.nan
    qtr_ret  = (close.iloc[-1]/close.iloc[-63]-1)*100 if len(close)>=63 else np.nan

    sma20v = safe_float(sma(close,min(20,max(3,n//3))).iloc[-1]) if len(close)>=3 else np.nan
    sma50v = safe_float(sma(close,min(50,max(5,n//2))).iloc[-1]) if len(close)>=5 else np.nan
    sma100v= safe_float(sma(close,min(100,ema_l)).iloc[-1]) if len(close)>=min(100,ema_l) else np.nan
    sma200v= safe_float(sma(close,ema_l).iloc[-1]) if len(close)>=ema_l else np.nan

    ema20v  = safe_float(ema(close,ema_s).iloc[-1])
    ema50v  = safe_float(ema(close,ema_m).iloc[-1])  if len(close)>=ema_m  else np.nan
    ema100v = safe_float(ema(close,min(100,ema_l)).iloc[-1]) if len(close)>=min(100,ema_l) else np.nan
    ema200v = safe_float(ema(close,ema_l).iloc[-1]) if len(close)>=ema_l else np.nan

    rsi14v  = safe_float(rsi(close,rsi_period).iloc[-1])
    ml,ms,mh= macd(close,fast=min(12,max(3,n//4)),slow=min(26,max(6,n//2)),signal=min(9,max(3,n//5)))
    _,_,bw,pb = bollinger(close,bb_period,2)
    atr14   = safe_float(atr(high,low,close,atr_period).iloc[-1]) if len(high)>=atr_period else np.nan
    adx_l,plus_di,minus_di = adx(high,low,close,adx_period)
    adx14   = safe_float(adx_l.iloc[-1])
    roc20v  = safe_float(roc(close,roc_period).iloc[-1])
    stk,std_= stochastic(high,low,close,stoch_period)
    wr14    = safe_float(williams_r(high,low,close,min(14,max(3,n-1))).iloc[-1])
    cci20v  = safe_float(cci(high,low,close,min(20,max(5,n-1))).iloc[-1])

    don_hi20, don_lo20 = donchian(high, low, min(20, max(5, n//3)))
    don_hi55, don_lo55 = donchian(high, low, min(55, max(10, n-2)))
    conv_ichi, base_ichi = ichimoku_base_conversion(high, low)
    slope20 = linear_reg_slope(close, min(20, max(8, n//3)))
    slope50 = linear_reg_slope(close, min(50, max(10, n//2)))
    z20 = safe_float(zscore(close, min(20, max(5, n//3))).iloc[-1]) if len(close) >= max(5, n//3) else np.nan

    range20_hi,range20_lo = high.rolling(min(20,max(5,n//3))).max().iloc[-1], low.rolling(min(20,max(5,n//3))).min().iloc[-1]
    range50_hi,range50_lo = high.rolling(min(50,max(10,n//2))).max().iloc[-1], low.rolling(min(50,max(10,n//2))).min().iloc[-1]
    range100_hi,range100_lo = high.rolling(min(100,max(15,n-1))).max().iloc[-1], low.rolling(min(100,max(15,n-1))).min().iloc[-1]
    pos20 = (last-range20_lo)/(range20_hi-range20_lo) if (range20_hi-range20_lo) else np.nan
    pos50 = (last-range50_lo)/(range50_hi-range50_lo) if (range50_hi-range50_lo) else np.nan
    pos100 = (last-range100_lo)/(range100_hi-range100_lo) if (range100_hi-range100_lo) else np.nan

    if len(high) >= 252:
        high52 = high.rolling(252).max().iloc[-1]
        low52  = low.rolling(252).min().iloc[-1]
    else:
        high52 = high.max() if len(high) else np.nan
        low52  = low.min() if len(low) else np.nan
    dist_52h = (last/high52 - 1)*100 if not pd.isna(high52) and high52 else np.nan
    dist_52l = (last/low52 - 1)*100 if not pd.isna(low52) and low52 else np.nan

    sr = support_resistance_levels(high, low, (20,50,100))
    fibs = fibonacci_levels(close, min(90, max(20, n)))
    fib_dist = {k:(last/v-1)*100 if v else np.nan for k,v in fibs.items()}
    nearest_fib = min(fib_dist.items(), key=lambda kv: abs(kv[1]))[0] if fib_dist else ""

    volume_ratio = volume.iloc[-1]/volume.rolling(min(20,max(5,n//3))).mean().iloc[-1] if len(volume)>=max(5,n//3) and volume.rolling(min(20,max(5,n//3))).mean().iloc[-1] else np.nan

    candles  = detect_candles(df)
    patterns = detect_patterns(close,high,low)

    bullish_stack = (not np.isnan(ema20v) and not np.isnan(ema50v) and not np.isnan(ema200v)) and last>ema20v>ema50v>ema200v
    bearish_stack = (not np.isnan(ema20v) and not np.isnan(ema50v) and not np.isnan(ema200v)) and last<ema20v<ema50v<ema200v
    sma_bullish = (not np.isnan(sma20v) and not np.isnan(sma50v) and not np.isnan(sma200v)) and last>sma20v>sma50v>sma200v
    sma_bearish = (not np.isnan(sma20v) and not np.isnan(sma50v) and not np.isnan(sma200v)) and last<sma20v<sma50v<sma200v

    trend_label = "Trend Bullish" if bullish_stack else "Trend Bearish" if bearish_stack else \
                  "Bullish" if (not np.isnan(ema50v) and not np.isnan(ema200v) and last>ema50v>ema200v) else \
                  "Bearish" if (not np.isnan(ema50v) and not np.isnan(ema200v) and last<ema50v<ema200v) else "Range"

    market_regime = regime.get("market_regime","Transition")
    cat = ASSETS[code]["cat"]
    quadrant = macro.get("quadrant","")

    trend_score=momentum_score=meanrev_score=vol_score=pat_score=macro_bias=corr_sc=0.0
    if bullish_stack: trend_score+=18
    elif bearish_stack: trend_score-=18
    if sma_bullish: trend_score += 6
    elif sma_bearish: trend_score -= 6
    trend_score += 6 if (not np.isnan(ema50v) and last>ema50v) else -6
    trend_score += 6 if (not np.isnan(ema200v) and last>ema200v) else -6
    if not np.isnan(slope20): trend_score += clamp(slope20*12, -6, 6)
    if not np.isnan(slope50): trend_score += clamp(slope50*10, -6, 6)
    if not np.isnan(adx14) and adx14>22:
        trend_score += 6 if trend_label.startswith("Trend Bullish") else (-6 if trend_label.startswith("Trend Bearish") else 0)
    if len(conv_ichi.dropna()) and len(base_ichi.dropna()):
        cval = safe_float(conv_ichi.iloc[-1]); bval = safe_float(base_ichi.iloc[-1])
        if not np.isnan(cval) and not np.isnan(bval):
            if last > cval > bval: trend_score += 4
            elif last < cval < bval: trend_score -= 4

    mh_val = safe_float(mh.iloc[-1])
    if not np.isnan(mh_val): momentum_score += clamp(mh_val*10,-8,8)
    if not np.isnan(roc20v): momentum_score += clamp(roc20v/2,-8,8)
    roc10v = safe_float(roc(close, min(10,max(2,n//5))).iloc[-1]) if len(close)>=max(2,n//5) else np.nan
    if not np.isnan(roc10v): momentum_score += clamp(roc10v/2.5, -6, 6)
    if not np.isnan(month_ret): momentum_score += clamp(month_ret/2.5,-8,8)
    if not np.isnan(qtr_ret):   momentum_score += clamp(qtr_ret/4,-8,8)
    sk = safe_float(stk.iloc[-1]); sdv = safe_float(std_.iloc[-1])
    if not np.isnan(sk) and not np.isnan(sdv):
        if sk > sdv and sk < 80: momentum_score += 2
        elif sk < sdv and sk > 20: momentum_score -= 2

    if not np.isnan(rsi14v):
        if rsi14v<30: meanrev_score+=10
        elif rsi14v<40: meanrev_score+=5
        elif rsi14v>70: meanrev_score-=10
        elif rsi14v>60: meanrev_score-=4
    pb_val=safe_float(pb.iloc[-1])
    if not np.isnan(pb_val):
        if pb_val<0.1: meanrev_score+=8
        elif pb_val>0.9: meanrev_score-=8
    if not np.isnan(z20):
        if z20 < -1.5: meanrev_score += 4
        elif z20 > 1.5: meanrev_score -= 4

    if not np.isnan(atr14) and last:
        atr_pct=atr14/last*100; vol_score += clamp(-(atr_pct-3),-6,6)
    if not np.isnan(pos20):
        if pos20 > 0.95: vol_score += 4
        elif pos20 < 0.05: vol_score -= 2
    if len(bw.dropna()):
        bbw = safe_float(bw.iloc[-1])
        if not np.isnan(bbw) and bbw < 0.08: vol_score += 2

    if "Breakout 20d" in patterns: pat_score+=8
    if "Breakdown 20d" in patterns: pat_score-=8
    if "Breakout 50d" in patterns: pat_score+=6
    if "Breakdown 50d" in patterns: pat_score-=6
    if "Bullish Engulfing" in candles or "Hammer" in candles or "Piercing Pattern" in candles or "Morning Star" in candles or "Bullish Harami" in candles: pat_score+=4
    if "Bearish Engulfing" in candles or "Shooting Star" in candles or "Dark Cloud Cover" in candles or "Evening Star" in candles or "Bearish Harami" in candles: pat_score-=4
    if "Double Bottom" in patterns: pat_score+=3
    if "Double Top" in patterns: pat_score-=3
    if "Bull Flag / Consolidation" in patterns: pat_score += 3
    if "Bear Flag / Consolidation" in patterns: pat_score -= 3

    infl_delta = safe_float(macro.get("inflation_delta", np.nan))
    growth_delta = safe_float(macro.get("growth_delta", np.nan))
    real_yield_proxy = safe_float(macro.get("real_yield_proxy", np.nan))
    gold_sp_ratio_3m = safe_float(macro.get("gold_sp_ratio_3m", np.nan))
    tips_3m = safe_float(macro.get("tips_3m", np.nan))
    quadrant_strength = safe_float(macro.get("quadrant_strength", np.nan))

    if quadrant=="Goldilocks / Reflazione":
        if cat in ["Indici","Settoriali","Crypto"]: macro_bias+=5
        if code in ["NASDAQ","RUSSELL","EMERGING","XLK","SOXX","XLY","XLI","BTC","ETH","SOL","ADA","XLRE","HY"]: macro_bias+=4
        if cat=="Bond" and code not in ["US2Y","HY"]: macro_bias-=2
        if not pd.isna(real_yield_proxy) and real_yield_proxy > 2.5 and code in ["NASDAQ","XLK","SOXX","BTC","ETH","SOL"]: macro_bias-=2

    elif quadrant=="Surriscaldamento":
        if cat in ["Commodities","Settoriali"]: macro_bias+=4
        if code in ["XLE","XME","XLB","XLF","KBE","WTI","BRENT","GOLD","COPPER","CORN","WHEAT","SOYBEAN","MOO","TIPS"]: macro_bias+=5
        if code in ["US10Y","US30Y","TLT","IEF","LQD","IG"]: macro_bias-=5
        if code in ["NASDAQ","XLK","SOXX","XLRE"]: macro_bias-=3
        if code in ["BTC","ETH","SOL","ADA"]: macro_bias-=2

    elif quadrant=="Stagflazione":
        # Forti favoriti: real asset, difensivi, oro, energia
        if code in ["GOLD","SILVER","GC=F","SI=F"]: macro_bias+=8
        if code in ["TIPS","XLU","XLP","XLV","WTI","BRENT","XLE","MOO","GAS"]: macro_bias+=6
        if code in ["PLATINUM","PALLADIUM","COFFEE","SUGAR","COTTON","CORN","WHEAT","SOYBEAN"]: macro_bias+=4
        if code in ["US2Y","SHY","DXY"]: macro_bias+=3
        if cat=="Commodities" and code not in ["GAS"]: macro_bias+=3
        # Forti penalizzati: growth, crypto, bond lunghi, tech
        if code in ["NASDAQ","XLK","SOXX","IBB","ICLN","XLY","XLRE"]: macro_bias-=7
        if code in ["BTC","ETH","SOL","ADA","DOT","AVAX","LINK","XRP"]: macro_bias-=6
        if code in ["US10Y","US30Y","TLT","IEF","LQD","IG"]: macro_bias-=5
        if code in ["RUSSELL","EMERGING","ASIA_EX_JP"]: macro_bias-=4
        if code in ["XLF","KBE","KIE"]: macro_bias-=3

    elif quadrant=="Recessione / Deflazione":
        if code in ["US2Y","US10Y","US30Y","XLV","XLP","XLU","GOLD"]: macro_bias+=5
        if cat=="Bond" and code not in ["HY"]: macro_bias+=3
        if code in ["TLT","IEF","LQD","IG","SHY"]: macro_bias+=3
        if code in ["XLE","XME","XLB","KBE","ITA","BTC","ETH","SOL","ADA","WTI","BRENT","HY"]: macro_bias-=5
        if cat=="Crypto": macro_bias-=4
        if cat=="Commodities" and code not in ["GOLD","SILVER"]: macro_bias-=3

    if not pd.isna(infl_delta):
        if infl_delta > 0.25 and code in ["GOLD","SILVER","TIPS","WTI","BRENT","XLE","XME","XLB","CORN","WHEAT"]: macro_bias += 2.5
        if infl_delta < -0.25 and code in ["US10Y","US30Y","XLK","NASDAQ","TLT","LQD"]: macro_bias += 2.0
    if not pd.isna(growth_delta):
        if growth_delta > 0.4 and code in ["RUSSELL","XLI","XLY","XLF","KBE","EMERGING","BTC","ETH","SOL"]: macro_bias += 2
        if growth_delta < -0.4 and code in ["XLV","XLP","XLU","US10Y","US30Y","GOLD","TIPS"]: macro_bias += 2.0
        if growth_delta < -0.4 and code in ["XLY","XLF","KBE","RUSSELL","WTI","BRENT","BTC","ETH"]: macro_bias -= 2.5
    if not pd.isna(gold_sp_ratio_3m):
        if gold_sp_ratio_3m > 5 and code in ["GOLD","SILVER","TIPS","XLV","XLP","XLU","US10Y","US30Y"]: macro_bias += 2.0
        if gold_sp_ratio_3m > 5 and code in ["BTC","ETH","SOL","ADA","NASDAQ","XLK","XLY"]: macro_bias -= 2.0
    if not pd.isna(tips_3m) and tips_3m > 3 and code in ["TIPS","GOLD","SILVER","XLE","XME","WTI"]:
        macro_bias += 2.0
    if not pd.isna(quadrant_strength):
        macro_bias *= clamp(0.90 + quadrant_strength/8, 0.9, 1.22)
    macro_bias = clamp(macro_bias, -15, 15)

    if corr60 is not None and not corr60.empty and code in corr60.index:
        s=corr60.loc[code].drop(labels=[code],errors="ignore").dropna()
        if not s.empty:
            pos_corr=float(s.max())
            if abs(pos_corr)<0.35: corr_sc+=2
    if market_regime=="High Volatility / Stress" and (cat=="Bond" or code in ["GOLD","XLV","XLP","XLU"]): corr_sc+=4

    if market_regime in ["Trend Bullish","Trend Bearish"]:
        macro_weight=0.06
        total=trend_score*0.36+momentum_score*0.28+meanrev_score*0.10+vol_score*0.10+pat_score*0.08+macro_bias*macro_weight+corr_sc*0.02
    elif market_regime=="Mean Reversion / Range":
        macro_weight=0.08
        total=trend_score*0.16+momentum_score*0.16+meanrev_score*0.30+vol_score*0.16+pat_score*0.10+macro_bias*macro_weight+corr_sc*0.04
    elif market_regime=="High Volatility / Stress":
        macro_weight=0.22
        total=trend_score*0.14+momentum_score*0.10+meanrev_score*0.16+vol_score*0.20+pat_score*0.08+macro_bias*macro_weight+corr_sc*0.10
    else:
        macro_weight=0.10
        total=trend_score*0.25+momentum_score*0.22+meanrev_score*0.18+vol_score*0.12+pat_score*0.09+macro_bias*macro_weight+corr_sc*0.04

    # In stagflazione e deflazione il macro bias deve pesare di più
    # perché il ciclo è il fattore dominante, non il tecnico puro
    if quadrant in ["Stagflazione", "Recessione / Deflazione"]:
        total = total - macro_bias*macro_weight + macro_bias*0.25

    score = clamp(50+total,0,100)
    conf  = clamp(45+min(abs(trend_score)/3,12)+min(abs(momentum_score)/4,10)+min(abs(macro_bias)*1.2,10)+min(abs(pat_score)*1.5,8)+min(regime.get("regime_confidence",50)/10,10),30,95)
    signal = "HIGH CONVICTION BUY" if score>=72 and conf>=62 else \
             "BUY"                 if score>=60 else \
             "HIGH CONVICTION SELL" if score<=28 and conf>=62 else \
             "SELL"                if score<=40 else "HOLD"

    notes=[]

    # 1. Struttura di trend principale
    if bullish_stack and sma_bullish:
        notes.append("EMA e SMA allineate al rialzo — struttura bullish solida")
    elif bullish_stack:
        notes.append("EMA allineate al rialzo (EMA20>EMA50>EMA200)")
    elif bearish_stack and sma_bearish:
        notes.append("EMA e SMA allineate al ribasso — struttura bearish solida")
    elif bearish_stack:
        notes.append("EMA allineate al ribasso (EMA20<EMA50<EMA200)")

    # 2. Forza del trend (ADX)
    if not np.isnan(adx14):
        if adx14 > 30:   notes.append(f"trend molto direzionale (ADX {adx14:.0f})")
        elif adx14 > 20: notes.append(f"trend in sviluppo (ADX {adx14:.0f})")
        elif adx14 < 15: notes.append(f"mercato laterale — ATR basso (ADX {adx14:.0f})")

    # 3. RSI — condizione di momentum
    if not np.isnan(rsi14v):
        if rsi14v < 28:   notes.append(f"RSI ipervenduto ({rsi14v:.0f}) — possibile rimbalzo")
        elif rsi14v < 38: notes.append(f"RSI scarico ({rsi14v:.0f}) — zona di interesse long")
        elif rsi14v > 72: notes.append(f"RSI ipercomprato ({rsi14v:.0f}) — attenzione estensione")
        elif rsi14v > 60: notes.append(f"RSI in forza ({rsi14v:.0f}) — momentum positivo")

    # 4. Pattern tecnici significativi (priorità ai più rilevanti)
    priority_patterns = ["Double Bottom","Double Top","Cup and Handle","Head and Shoulders",
                         "Inverse Head and Shoulders","Bull Flag / Consolidation","Bear Flag / Consolidation"]
    for pp in priority_patterns:
        if pp in patterns:
            notes.append(f"pattern: {pp}")
            break

    # 5. Breakout / Breakdown strutturali
    if "Breakout 50d" in patterns:
        notes.append(f"breakout massimi 50g — rottura resistenza rilevante")
    elif "Breakout 20d" in patterns:
        notes.append(f"breakout massimi 20g — momentum di breve")
    if "Breakdown 50d" in patterns:
        notes.append(f"breakdown minimi 50g — rottura supporto importante")
    elif "Breakdown 20d" in patterns:
        notes.append(f"breakdown minimi 20g — pressione ribassista")

    # 6. Candele significative
    if candles:
        candle_desc = {
            "Bullish Engulfing": "candela di inversione bullish (engulfing)",
            "Bearish Engulfing": "candela di inversione bearish (engulfing)",
            "Hammer": "hammer — segnale di rimbalzo da supporto",
            "Shooting Star": "shooting star — rifiuto di resistenza",
            "Morning Star": "morning star — inversione rialzista a 3 candele",
            "Evening Star": "evening star — inversione ribassista a 3 candele",
            "Doji": "doji — indecisione del mercato",
        }
        for c in candles[:1]:
            notes.append(candle_desc.get(c, f"segnale candlestick: {c}"))

    # 7. Prossimità a livelli chiave
    if not np.isnan(sr.get("Sup_20", np.nan)):
        d_sup20 = (last/sr["Sup_20"] - 1) * 100 if sr["Sup_20"] else np.nan
        d_res20 = (last/sr["Res_20"] - 1) * 100 if sr["Res_20"] else np.nan
        if not np.isnan(d_sup20) and abs(d_sup20) < 1.5:
            notes.append(f"sul supporto dei 20g ({num_fmt(sr['Sup_20'])})")
        if not np.isnan(d_res20) and abs(d_res20) < 1.5:
            notes.append(f"vicino alla resistenza dei 20g ({num_fmt(sr['Res_20'])})")

    # 8. Prossimità Fibonacci
    if nearest_fib and nearest_fib in fib_dist and not np.isnan(fib_dist[nearest_fib]) and abs(fib_dist[nearest_fib]) < 1.5:
        notes.append(f"in zona Fib {nearest_fib} — livello strutturale")

    # 9. Contesto macro (solo se rilevante)
    if macro_bias > 6:   notes.append(f"favorito dal quadrante macro ({quadrant})")
    elif macro_bias < -5: notes.append(f"penalizzato dal quadrante macro ({quadrant})"  )

    return {
        "Codice": code, "Nome": ASSETS[code]["name"], "Categoria": cat,
        "Prezzo": last, "Var 1D (%)": day_ret, "Var 1W (%)": week_ret,
        "Var 1M (%)": month_ret, "Var 3M (%)": qtr_ret,
        "RSI14": rsi14v, "ADX14": adx14, "BB %B": pb_val,
        "EMA20": ema20v, "EMA50": ema50v, "EMA200": ema200v,
        "MACD Hist": mh_val, "Stoch K": sk,
        "Williams %R": wr14, "CCI20": cci20v, "ROC20": roc20v,
        "Trend": trend_label, "Candles": ", ".join(candles),
        "Patterns": ", ".join(patterns),
        "Score": round(score,1), "Confidence": round(conf,1),
        "Signal": signal, "Notes": "; ".join(list(dict.fromkeys(notes))[:7])
    }


# =============================================================================
#  INGESTION & SNAPSHOT MULTI-TIMEFRAME
# =============================================================================

def fetch_history(ticker):
    try:
        hist = yf.Ticker(ticker).history(period=f"{LOOKBACK_DAYS}d", interval="1d", auto_adjust=False)
        if hist is None or hist.empty: return pd.DataFrame()
        hist = hist[["Open","High","Low","Close","Volume"]].copy().dropna(how="all")
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist
    except: return pd.DataFrame()

def ingest_all():
    print("\n[INGESTION] Download storico asset...")
    history={}; closes={}
    for code,meta in ASSETS.items():
        df = fetch_history(meta["ticker"])
        if len(df)<100:
            print(f"  - {code:10} dati insufficienti ({len(df)} righe)"); continue
        history[code]=df; closes[code]=df["Close"].rename(code)
        print(f"  ✓ {code:10} {len(df)} righe")
    prices = pd.concat(list(closes.values()),axis=1,sort=False).sort_index() if closes else pd.DataFrame()
    return history, prices

def build_snapshot_multitf(history, macro, regime, corr):
    """
    Costruisce tre snapshot: Daily, Weekly, Monthly.
    Ogni snapshot è un DataFrame ordinato per Score.
    """
    corr60 = corr.get("corr_60", pd.DataFrame())
    snapshots = {"Daily":[], "Weekly":[], "Monthly":[]}

    for code, df in history.items():
        try:
            # Daily: usa direttamente lo storico
            r_d = compute_score_for_df(df, macro, regime, corr60, code)
            if r_d: snapshots["Daily"].append(r_d)

            # Weekly: ricampiona lo storico
            df_w = resample_ohlcv(df, "W")
            if len(df_w)>=30:
                r_w = compute_score_for_df(df_w, macro, regime, corr60, code)
                if r_w: snapshots["Weekly"].append(r_w)

            # Monthly: ricampiona lo storico
            df_m = resample_ohlcv(df, "ME")
            if len(df_m)>=6:
                r_m = compute_score_for_df(df_m, macro, regime, corr60, code)
                if r_m: snapshots["Monthly"].append(r_m)

        except Exception as e:
            print(f"  ✗ snapshot {code}: {e}")

    result={}
    for tf, rows in snapshots.items():
        df_tf = pd.DataFrame(rows)
        result[tf] = df_tf.sort_values(["Score","Confidence"],ascending=False) if not df_tf.empty else df_tf
        print(f"  ✓ {tf}: {len(result[tf])} asset calcolati")

    return result

# =============================================================================
#  MACRO CONTEXT (FRED + market proxies)
# =============================================================================

FRED_SERIES = {
    "cpi":"CPIAUCSL","core_cpi":"CPILFESL","unemployment":"UNRATE",
    "jobless_claims":"ICSA","industrial_prod":"INDPRO","retail_sales":"RSAFS",
    "dgs10":"DGS10","dgs2":"DGS2"
}

def fred_csv_url(sid): return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
def fred_json_url(sid):
    base="https://api.stlouisfed.org/fred/series/observations"
    return f"{base}?series_id={sid}&api_key={FRED_API_KEY}&file_type=json" if FRED_API_KEY else f"{base}?series_id={sid}&file_type=json"

def download_fred(sid):
    try:
        r=requests.get(fred_json_url(sid),timeout=20); r.raise_for_status()
        obs=r.json().get("observations",[])
        if obs:
            df=pd.DataFrame(obs); df["date"]=pd.to_datetime(df["date"]); df["value"]=df["value"].apply(safe_float)
            s=pd.Series(df["value"].values,index=df["date"]).dropna()
            if not s.empty: return s
    except: pass
    try:
        r=requests.get(fred_csv_url(sid),timeout=20); r.raise_for_status()
        df=pd.read_csv(io.StringIO(r.text)); dc,vc=df.columns[:2]
        df[dc]=pd.to_datetime(df[dc]); df[vc]=pd.to_numeric(df[vc],errors="coerce")
        return pd.Series(df[vc].values,index=df[dc]).dropna()
    except: return pd.Series(dtype=float)

def monthly_yoy(s):
    s=s.dropna()
    if len(s)<13: return np.nan
    return (s.iloc[-1]/s.iloc[-13]-1)*100

def monthly_yoy_series(s):
    s=s.dropna()
    if len(s)<13: return pd.Series(dtype=float)
    out=(s/s.shift(12)-1)*100
    return out.dropna()

def recent_delta(series, lookback=3):
    s=series.dropna()
    if len(s)<lookback+1: return np.nan
    base=s.iloc[-(lookback+1):-1].mean()
    return s.iloc[-1]-base

def pct_change_n(series, n=63):
    s=series.dropna()
    if len(s)<=n: return np.nan
    return (s.iloc[-1]/s.iloc[-(n+1)]-1)*100

def safe_mean(vals):
    vals=[v for v in vals if not pd.isna(v)]
    return float(np.mean(vals)) if vals else np.nan

def build_macro_context(prices):
    # Controlla cache
    previous_cache = None
    force_refresh = str(os.environ.get("SARADA_FORCE_MACRO_REFRESH", "")).strip().lower() in {"1","true","yes","on"}
    force_refresh_flag = DATA_DIR / "force_macro_refresh.flag"
    if force_refresh_flag.exists():
        force_refresh = True
        print(f"\n[MACRO] Refresh forzato — file flag rilevato: {force_refresh_flag}")
    if MACRO_CACHE_FILE.exists():
        try:
            with open(MACRO_CACHE_FILE, encoding="utf-8") as f:
                cached=json.load(f)
            previous_cache = cached
            cached_time=datetime.strptime(cached.get("timestamp","01/01/2000 00:00:00"),"%d/%m/%Y %H:%M:%S")
            cache_age_days = (datetime.now()-cached_time).days
            if force_refresh:
                print(f"\n[MACRO] Refresh forzato — bypass cache ({cache_age_days}g)")
            elif cache_age_days < MACRO_REFRESH_DAYS:
                print(f"\n[MACRO] Uso cache ({cache_age_days}g) — prossimo aggiornamento tra {MACRO_REFRESH_DAYS-cache_age_days}g")
                return cached
            else:
                print(f"\n[MACRO] Cache scaduta ({cache_age_days}g) — refresh macro reale")
        except Exception:
            previous_cache = None
            if force_refresh:
                print("\n[MACRO] Refresh forzato — cache non leggibile, ricalcolo completo")
    elif force_refresh:
        print("\n[MACRO] Refresh forzato — nessuna cache disponibile")

    def yoy_series(s):
        s = s.dropna()
        return ((s / s.shift(12)) - 1) * 100 if len(s) >= 13 else pd.Series(dtype=float)

    def trailing_mean(s, n=3):
        s = s.dropna()
        return safe_float(s.tail(n).mean()) if len(s) >= n else np.nan

    def trailing_prev_mean(s, n=3):
        s = s.dropna()
        return safe_float(s.iloc[-2*n:-n].mean()) if len(s) >= 2*n else np.nan

    def norm(val, scale, lo=-2.5, hi=2.5):
        if pd.isna(val): return 0.0
        return float(np.clip(val/scale, lo, hi))

    print("\n[MACRO] Download serie FRED...")
    fred={name:download_fred(sid) for name,sid in FRED_SERIES.items()}
    for name,ser in fred.items():
        print(f"  {'✓' if not ser.empty else '✗'} {name:20} {len(ser)} osservazioni" if not ser.empty else f"  ✗ {name:20} N/D")

    vix=np.nan
    try:
        vh=fetch_history("^VIX"); vix=safe_float(vh["Close"].iloc[-1]) if not vh.empty else np.nan
    except: pass

    dxy  = safe_float(prices["DXY"].dropna().iloc[-1])  if "DXY"  in prices.columns else np.nan
    gold = safe_float(prices["GOLD"].dropna().iloc[-1]) if "GOLD" in prices.columns else np.nan
    oil  = safe_float(prices["WTI"].dropna().iloc[-1])  if "WTI"  in prices.columns else np.nan
    spx = prices["SP500"].dropna() if "SP500" in prices.columns else pd.Series(dtype=float)

    spx_3m=np.nan
    if len(spx)>65:
        spx_3m=(spx.iloc[-1]/spx.iloc[-63]-1)*100

    hy_lqd_mom=np.nan
    if "HY" in prices.columns and "IG" in prices.columns:
        ratio=(prices["HY"]/prices["IG"]).dropna()
        if len(ratio)>63: hy_lqd_mom=(ratio.iloc[-1]/ratio.iloc[-21]-1)*100

    cpi_yoy=monthly_yoy(fred["cpi"]); core_cpi_yoy=monthly_yoy(fred["core_cpi"])
    unemployment=safe_float(fred["unemployment"].dropna().iloc[-1]) if not fred["unemployment"].empty else np.nan
    jobless=safe_float(fred["jobless_claims"].dropna().iloc[-1]) if not fred["jobless_claims"].empty else np.nan
    ind_yoy=monthly_yoy(fred["industrial_prod"]); ret_yoy=monthly_yoy(fred["retail_sales"])
    dgs10=safe_float(fred["dgs10"].dropna().iloc[-1]) if not fred["dgs10"].empty else np.nan
    dgs2 =safe_float(fred["dgs2"].dropna().iloc[-1])  if not fred["dgs2"].empty  else np.nan
    yield_spread=dgs10-dgs2 if not pd.isna(dgs10) and not pd.isna(dgs2) else np.nan
    real_yield=dgs10-cpi_yoy if not pd.isna(dgs10) and not pd.isna(cpi_yoy) else np.nan

    cpi_yoy_hist = yoy_series(fred["cpi"])
    core_cpi_yoy_hist = yoy_series(fred["core_cpi"])
    ind_yoy_hist = yoy_series(fred["industrial_prod"])
    ret_yoy_hist = yoy_series(fred["retail_sales"])

    y10_hist = fred["dgs10"].dropna()
    y2_hist = fred["dgs2"].dropna()
    ys_idx = y10_hist.index.union(y2_hist.index)
    yield_spread_hist = (y10_hist.reindex(ys_idx).sort_index().ffill() - y2_hist.reindex(ys_idx).sort_index().ffill()).dropna()

    ry_idx = y10_hist.index.union(cpi_yoy_hist.index)
    real_yield_hist = (y10_hist.reindex(ry_idx).sort_index().ffill() - cpi_yoy_hist.reindex(ry_idx).sort_index().ffill()).dropna()

    infl_hist = pd.concat([cpi_yoy_hist.rename('cpi'), core_cpi_yoy_hist.rename('core')], axis=1).mean(axis=1).dropna()
    infl_now = safe_float(infl_hist.iloc[-1]) if not infl_hist.empty else np.nan
    infl_3m_avg = trailing_mean(infl_hist, 3)
    infl_prev_3m_avg = trailing_prev_mean(infl_hist, 3)
    inflation_delta = infl_now - infl_3m_avg if not pd.isna(infl_now) and not pd.isna(infl_3m_avg) else np.nan
    inflation_momentum = infl_3m_avg - infl_prev_3m_avg if not pd.isna(infl_3m_avg) and not pd.isna(infl_prev_3m_avg) else np.nan

    growth_components = pd.concat([
        ind_yoy_hist.rename('ind'),
        ret_yoy_hist.rename('ret'),
        (yield_spread_hist*2.0).rename('spread_proxy')
    ], axis=1).dropna(how='all')
    growth_hist = pd.concat([
        (growth_components['ind']/3).clip(-2.5,2.5) if 'ind' in growth_components else pd.Series(dtype=float),
        (growth_components['ret']/3).clip(-2.5,2.5) if 'ret' in growth_components else pd.Series(dtype=float),
        (growth_components['spread_proxy']/1.5).clip(-2.0,2.0) if 'spread_proxy' in growth_components else pd.Series(dtype=float)
    ], axis=1).mean(axis=1).dropna()
    growth_now = safe_float(growth_hist.iloc[-1]) if not growth_hist.empty else np.nan
    growth_3m_avg = trailing_mean(growth_hist, 3)
    growth_prev_3m_avg = trailing_prev_mean(growth_hist, 3)
    growth_delta = growth_3m_avg - growth_prev_3m_avg if not pd.isna(growth_3m_avg) and not pd.isna(growth_prev_3m_avg) else np.nan

    gold_sp_ratio=np.nan; gold_sp_ratio_3m=np.nan; gold_sp_ratio_hist = pd.Series(dtype=float)
    if "GOLD" in prices.columns and "SP500" in prices.columns:
        gold_series = prices["GOLD"].dropna()
        sp_series = prices["SP500"].dropna()
        idx = gold_series.index.intersection(sp_series.index)
        if len(idx) > 70:
            gold_sp_ratio_hist = (gold_series.loc[idx] / sp_series.loc[idx]).dropna()
            gold_sp_ratio = safe_float(gold_sp_ratio_hist.iloc[-1]) if not gold_sp_ratio_hist.empty else np.nan
            if len(gold_sp_ratio_hist) > 63:
                gold_sp_ratio_3m = (gold_sp_ratio_hist.iloc[-1] / gold_sp_ratio_hist.iloc[-63] - 1) * 100

    tips_3m=np.nan
    if "TIPS" in prices.columns:
        tips=prices["TIPS"].dropna()
        if len(tips)>63: tips_3m=(tips.iloc[-1]/tips.iloc[-63]-1)*100

    fg=np.nan; fg_label="N/A"
    try:
        d=requests.get("https://api.alternative.me/fng/?limit=1",timeout=10).json()["data"][0]
        fg=safe_float(d["value"]); fg_label=d["value_classification"]
    except: pass

    # Macro block scores
    gs=is_=ss=ls=0.0
    if not pd.isna(spx_3m): gs += np.clip(spx_3m/4,-2.5,2.5)
    if not pd.isna(yield_spread): gs += np.clip(yield_spread/0.75,-2,2)
    if not pd.isna(ind_yoy): gs += np.clip(ind_yoy/3,-2,2)
    if not pd.isna(ret_yoy): gs += np.clip(ret_yoy/3,-2,2)
    if not pd.isna(unemployment): gs += np.clip((4.5-unemployment)/0.7,-2,2)
    if not pd.isna(growth_delta): gs += np.clip(growth_delta/0.5,-1.5,1.5)

    if not pd.isna(cpi_yoy): is_ += np.clip((cpi_yoy-2)/1.0,-2.5,2.5)
    if not pd.isna(core_cpi_yoy): is_ += np.clip((core_cpi_yoy-2)/1.0,-2,2)
    if not pd.isna(oil) and "WTI" in prices.columns and prices["WTI"].dropna().shape[0]>21:
        oil_1m=(prices["WTI"].dropna().iloc[-1]/prices["WTI"].dropna().iloc[-21]-1)*100
        is_ += np.clip(oil_1m/10,-1.5,1.5)
    if not pd.isna(dxy) and "DXY" in prices.columns and prices["DXY"].dropna().shape[0]>21:
        dxy_1m=(prices["DXY"].dropna().iloc[-1]/prices["DXY"].dropna().iloc[-21]-1)*100
        is_ += np.clip((-dxy_1m)/5,-1,1)
    if not pd.isna(inflation_delta): is_ += np.clip(inflation_delta/0.35,-1.0,1.5)
    if not pd.isna(inflation_momentum): is_ += np.clip(inflation_momentum/0.25,-0.8,1.2)

    real_yield_delta = np.nan
    ry_3m_avg = trailing_mean(real_yield_hist, 3)
    ry_prev_3m_avg = trailing_prev_mean(real_yield_hist, 3)
    if not pd.isna(ry_3m_avg) and not pd.isna(ry_prev_3m_avg):
        real_yield_delta = ry_3m_avg - ry_prev_3m_avg
    if not pd.isna(real_yield):
        if real_yield < 0:
            is_ += 1.0; gs -= 0.75
        elif real_yield < 0.5:
            is_ += 0.35
    if not pd.isna(real_yield_delta) and real_yield_delta < -0.35:
        is_ += min(1.0, abs(real_yield_delta))
        gs -= min(1.0, abs(real_yield_delta) * 0.8)

    # Gold/SP500 ratio: segnale di stagflazione solo se oro sale MOLTO più di SP500
    # Non penalizzare la crescita solo perché il ratio è positivo (normale in risk-on)
    if not pd.isna(gold_sp_ratio_3m) and gold_sp_ratio_3m > 8:
        # Oro sale molto più dell'azionario → segnale difensivo/inflazionistico
        gs -= 1.0
        is_ += 0.5

    if not pd.isna(vix): ss += np.clip((vix-18)/6,-2,3)
    if not pd.isna(fg): ss += np.clip((50-fg)/15,-2,2)
    if not pd.isna(hy_lqd_mom): ss += np.clip((-hy_lqd_mom)/4,-2,2)
    if not pd.isna(jobless): ss += np.clip((jobless-250000)/100000,-1.5,2)
    if not pd.isna(real_yield): ls += np.clip((-real_yield)/1.5,-2,2)
    if not pd.isna(dgs10): ls += np.clip((4.2-dgs10)/0.8,-2,2)

    growth_level = growth_now if not pd.isna(growth_now) else gs
    growth_delta_use = growth_delta if not pd.isna(growth_delta) else 0.0
    infl_level = infl_now if not pd.isna(infl_now) else np.nanmean([x for x in [cpi_yoy, core_cpi_yoy] if not pd.isna(x)]) if any(not pd.isna(x) for x in [cpi_yoy, core_cpi_yoy]) else is_
    infl_delta_use = inflation_delta if not pd.isna(inflation_delta) else inflation_momentum if not pd.isna(inflation_momentum) else 0.0

    score_goldilocks = 0.0
    score_overheating = 0.0
    score_stagflation = 0.0
    score_deflation = 0.0

    # ── LIVELLO INFLAZIONE ────────────────────────────────────────────────────
    # Goldilocks: inflazione contenuta (sotto 3%)
    score_goldilocks += norm(2.8 - infl_level, 1.2, -2.0, 2.2)
    # Surriscaldamento: inflazione alta MA crescita ancora presente
    score_overheating += norm(infl_level - 3.0, 1.0, -1.5, 2.0)
    # Stagflazione: inflazione alta E persistente (soglia più bassa)
    score_stagflation += norm(infl_level - 2.5, 0.8, -0.8, 3.0)
    # Deflazione: inflazione bassa/negativa
    score_deflation += norm(2.2 - infl_level, 1.0, -1.5, 2.5)

    # ── DELTA INFLAZIONE ──────────────────────────────────────────────────────
    # Goldilocks: inflazione in calo
    score_goldilocks += norm(-infl_delta_use, 0.35, -1.5, 2.0)
    # Surriscaldamento: inflazione in forte accelerazione
    score_overheating += norm(infl_delta_use, 0.35, -1.0, 2.0)
    # Stagflazione: inflazione stagnante o in lieve salita (anche delta neutro è ok)
    score_stagflation += norm(infl_delta_use + 0.1, 0.30, -0.5, 2.2)
    score_deflation += norm(-infl_delta_use, 0.30, -1.2, 1.8)

    # ── LIVELLO CRESCITA ──────────────────────────────────────────────────────
    # Goldilocks e Surriscaldamento: crescita positiva
    score_goldilocks += norm(growth_level, 1.0, -2.0, 2.0)
    score_overheating += norm(growth_level, 0.8, -1.5, 2.2)
    # Stagflazione: crescita debole o negativa (soglia chiave)
    score_stagflation += norm(-growth_level + 0.2, 0.8, -1.0, 2.8)
    score_deflation += norm(-growth_level, 0.8, -1.0, 2.8)

    # ── DELTA CRESCITA ────────────────────────────────────────────────────────
    score_goldilocks += norm(growth_delta_use, 0.35, -1.5, 2.2)
    score_overheating += norm(growth_delta_use, 0.30, -1.2, 2.0)
    # Stagflazione: crescita in deterioramento anche se parte da livello ok
    score_stagflation += norm(-growth_delta_use + 0.1, 0.28, -1.0, 2.4)
    score_deflation += norm(-growth_delta_use, 0.25, -1.0, 2.5)

    # ── YIELD SPREAD ─────────────────────────────────────────────────────────
    # Curva invertita o piatta → segnale recessivo/stagflazionistico
    score_goldilocks += norm(yield_spread, 1.0, -1.0, 1.8)
    score_overheating += norm(yield_spread, 0.8, -1.0, 1.4)
    score_stagflation += norm(-yield_spread + 0.2, 0.6, -0.8, 2.0)
    score_deflation += norm(-yield_spread, 0.5, -1.0, 2.2)

    # ── REAL YIELD ────────────────────────────────────────────────────────────
    # Real yield molto alto → freno alla crescita (stagflazionistico)
    score_goldilocks += norm(real_yield, 1.5, -1.0, 1.4)
    score_overheating += norm(-real_yield + 0.5, 1.0, -1.0, 1.6)
    # Real yield alto con inflazione alta → classico stagflazione
    score_stagflation += norm(-real_yield + 2.0, 0.8, -0.8, 2.2)
    score_deflation += norm(real_yield, 1.2, -1.0, 1.8)

    score_goldilocks += norm(real_yield_delta, 0.35, -1.0, 1.5)
    score_overheating += norm(-real_yield_delta, 0.30, -1.0, 1.8)
    score_stagflation += norm(-real_yield_delta, 0.22, -0.8, 1.8)
    score_deflation += norm(real_yield_delta, 0.30, -1.0, 1.8)

    # ── GOLD/SP500 RATIO ─────────────────────────────────────────────────────
    # Oro che outperforma fortemente l'azionario = segnale difensivo
    score_goldilocks += norm(-gold_sp_ratio_3m, 5.0, -1.5, 1.6)
    score_overheating += norm(-gold_sp_ratio_3m, 8.0, -0.8, 1.0)
    # Oro che sovraperforma molto → stagflazione più che surriscaldamento
    score_stagflation += norm(gold_sp_ratio_3m, 3.5, -0.8, 2.5)
    score_deflation += norm(gold_sp_ratio_3m, 5.0, -0.8, 1.6)

    # ── TIPS ─────────────────────────────────────────────────────────────────
    # TIPS in salita = mercato si protegge dall'inflazione
    score_goldilocks += norm(-tips_3m, 4.0, -1.0, 1.0)
    score_overheating += norm(tips_3m, 4.0, -0.8, 1.4)
    # TIPS + inflazione alta → stagflazione
    score_stagflation += norm(tips_3m, 3.0, -0.8, 2.2)
    score_deflation += norm(-tips_3m, 4.5, -1.0, 1.4)

    # ── STRESS / LIQUIDITA ────────────────────────────────────────────────────
    score_goldilocks += norm(-ss, 1.0, -1.2, 1.8)
    score_overheating += norm(-ss, 1.4, -1.0, 1.0)
    score_stagflation += norm(ss, 1.0, -0.8, 2.2)
    score_deflation += norm(ss, 0.9, -1.0, 2.4)

    score_goldilocks += norm(-ls, 1.2, -1.0, 1.0)
    score_overheating += norm(-ls, 1.5, -1.0, 0.8)
    score_stagflation += norm(ls, 0.9, -0.8, 1.4)
    score_deflation += norm(ls, 0.8, -1.0, 1.8)

    # ── SPX MOMENTUM ─────────────────────────────────────────────────────────
    # Attenzione: in stagflazione il mercato può rimbalzare. Non usare SPX
    # positivo come prova contro la stagflazione — usarlo solo con moderazione.
    if not pd.isna(spx_3m):
        score_goldilocks += norm(spx_3m, 10.0, -1.0, 1.5)
        score_overheating += norm(spx_3m, 16.0, -0.8, 1.0)  # peso ridotto
        score_deflation += norm(-spx_3m, 8.0, -1.0, 1.8)
        # SPX positivo NON è prova contro stagflazione — lo escludiamo
    if not pd.isna(hy_lqd_mom):
        score_goldilocks += norm(hy_lqd_mom, 2.0, -1.0, 1.0)
        score_deflation += norm(-hy_lqd_mom, 2.0, -1.0, 1.4)
        score_stagflation += norm(-hy_lqd_mom, 2.5, -0.8, 1.2)

    # ── SEGNALE AGGIUNTIVO: DISOCCUPAZIONE IN SALITA = STAGFLAZIONE/DEFLAZIONE
    if not pd.isna(unemployment):
        unemp_vs_norm = unemployment - 4.0  # sopra 4% = pressione negativa
        if unemp_vs_norm > 0.5:
            score_stagflation += norm(unemp_vs_norm, 1.0, 0, 1.5)
            score_deflation   += norm(unemp_vs_norm, 1.2, 0, 1.2)
            score_goldilocks  -= norm(unemp_vs_norm, 1.5, 0, 1.0)
            score_overheating -= norm(unemp_vs_norm, 2.0, 0, 1.0)

    quadrant_scores = {
        "Goldilocks / Reflazione": round(score_goldilocks, 2),
        "Surriscaldamento": round(score_overheating, 2),
        "Stagflazione": round(score_stagflation, 2),
        "Recessione / Deflazione": round(score_deflation, 2),
    }
    ordered_scores = sorted(quadrant_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_q, top_score = ordered_scores[0]
    second_q, second_score = ordered_scores[1]
    score_margin = round(top_score - second_score, 2)

    # Debug diagnostico — visibile nel CMD
    print(f"\n  [MACRO DEBUG] Scores quadranti:")
    for qn, qs in sorted(quadrant_scores.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(max(0, qs) * 3)
        print(f"    {qn:35} {qs:+.2f}  {bar}")
    print(f"  [MACRO DEBUG] CPI={cpi_yoy:.2f}% Core={core_cpi_yoy:.2f}% Infl_delta={inflation_delta:.3f} Infl_mom={inflation_momentum:.3f}")
    print(f"  [MACRO DEBUG] Growth={growth_level:.2f} Growth_delta={growth_delta:.3f} RealYield={real_yield:.2f}")
    print(f"  [MACRO DEBUG] Gold/SP3M={gold_sp_ratio_3m:.1f}% TIPS3M={tips_3m:.1f}% VIX={vix:.1f} YSpread={yield_spread:.3f}")
    regime_conviction = clamp(50 + score_margin*10 + min(abs(ss)*4, 10) + min(abs(infl_delta_use)*8, 8) + min(abs(growth_delta_use)*8, 8), 35, 95)

    q_map = {
        "Goldilocks / Reflazione": (
            ["Azioni Growth","Tech (XLK/SOXX)","Small Cap (RUSSELL)","Crypto (BTC/ETH)","Ciclici (XLY/XLI)","Emerging Markets","Real Estate (XLRE)","High Yield (HY)"],
            "Crescita credibile con inflazione sotto controllo o in rallentamento. Contesto risk-on sano: favoriti azionario growth, crypto e ciclici. Bond lunghi penalizzati dall'aspettativa di tassi stabili."
        ),
        "Surriscaldamento": (
            ["Energia (XLE/WTI/BRENT)","Commodities (GOLD/COPPER/CORN)","Materiali (XLB/XME)","Value (XLF/KBE)","Banche","Agribusiness (MOO)","Inflazione-linked (TIPS)"],
            "Crescita forte con inflazione alta o in accelerazione. Commodities e settori ciclici reali favoriti. Bond a lunga durata penalizzati. Tech growth sotto pressione da tassi alti."
        ),
        "Stagflazione": (
            ["Oro (GOLD/GC=F)","Argento (SILVER)","TIPS (TIP)","Utilities (XLU)","Difensivi (XLV/XLP)","Energia (XLE/WTI)","Agribusiness (MOO)","Cash / T-Bill brevi (US2Y)"],
            "Inflazione persistente con crescita fragile o in deterioramento. Scenario ostile al risk-on puro. L'oro è il rifugio classico. Tech, crypto e bond lunghi penalizzati. Difensivi e real asset favoriti."
        ),
        "Recessione / Deflazione": (
            ["Treasury US lunghi (TLT/US30Y)","Bond qualità (IG/LQD)","Healthcare (XLV)","Beni di base (XLP)","Oro (GOLD)","Utilities (XLU)","Cash","Yen (USDJPY short)"],
            "Crescita debole con inflazione in raffreddamento e bias difensivo netto. Bond governativi lunghi e settori difensivi favoriti. Asset rischiosi, commodity cicliche e high yield penalizzati."
        ),
    }

    q = top_q
    transition_state = score_margin < 1.0
    if previous_cache:
        prev_q = previous_cache.get("quadrant", "")
        if prev_q in quadrant_scores and prev_q != top_q:
            prev_score = quadrant_scores[prev_q]
            if (top_score - prev_score) < 0.75:
                q = prev_q
                transition_state = True
                ordered_scores = sorted(quadrant_scores.items(), key=lambda kv: (kv[0] != q, -kv[1]))
                second_q = top_q
                second_score = top_score
                top_score = quadrant_scores[q]
                score_margin = round(top_score - second_score, 2)
                regime_conviction = clamp(regime_conviction - 8, 30, 95)

    fav, qdesc = q_map[q]
    macro_conf=clamp(50+min(abs(gs)*8,18)+min(abs(is_)*8,18)+min(abs(ss)*5,12)+min(score_margin*6,12),35,95)
    quadrant_strength = safe_float(np.nanmean([
        abs(gs), abs(is_), abs(growth_delta_use), abs(infl_delta_use), abs(ss), abs(score_margin)
    ]))

    probs = normalize_prob_dict({k:max(v - min(quadrant_scores.values()) + 0.25, 0.01) for k,v in quadrant_scores.items()})
    ordered_probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    trans=[{"quadrant":x,"probability":p} for x,p in ordered_probs if x!=q]
    stay_prob=clamp(0.55*probs.get(q,50)+0.30*macro_conf+0.15*regime_conviction-0.20*(trans[0]["probability"] if trans else 0),25,92)

    macro={
        "timestamp":now_str(),"cpi_yoy":cpi_yoy,"core_cpi_yoy":core_cpi_yoy,
        "unemployment":unemployment,"jobless_claims":jobless,"industrial_yoy":ind_yoy,
        "retail_yoy":ret_yoy,"dgs10":dgs10,"dgs2":dgs2,"yield_spread":yield_spread,
        "real_yield_proxy":real_yield,"real_yield_delta":real_yield_delta,
        "vix":vix,"dxy":dxy,"gold":gold,"oil":oil,"spx_3m":spx_3m,
        "fear_greed":fg,"fear_greed_label":fg_label,
        "growth_score":gs,"inflation_score":is_,"stress_score":ss,"liquidity_score":ls,
        "growth_level":growth_level,"growth_delta":growth_delta,"inflation_level":infl_now,
        "inflation_delta":inflation_delta,"inflation_momentum":inflation_momentum,
        "gold_sp_ratio":gold_sp_ratio,"gold_sp_ratio_3m":gold_sp_ratio_3m,"tips_3m":tips_3m,
        "quadrant":q,"quadrant_desc":qdesc,"asset_favorites":fav,
        "macro_confidence":macro_conf,"quadrant_strength":quadrant_strength,
        "quadrant_probabilities":probs,"stay_probability_1_3m":round(stay_prob,1),
        "most_likely_next_quadrant":second_q,"next_quadrant_probability":round(probs.get(second_q,0),1),
        "transition_candidates":trans[:3],
        "quadrant_scores":quadrant_scores,
        "second_quadrant":second_q,
        "second_quadrant_score":round(second_score,2),
        "quadrant_margin":score_margin,
        "regime_conviction":round(regime_conviction,1),
        "transition_state":transition_state,
    }
    with open(MACRO_CACHE_FILE,"w",encoding="utf-8") as f:
        json.dump(macro,f,ensure_ascii=False,indent=2,default=str)
    try:
        if force_refresh_flag.exists():
            force_refresh_flag.unlink()
            print(f"[MACRO] File flag refresh consumato e rimosso: {force_refresh_flag}")
    except Exception as e:
        print(f"[MACRO] Warning rimozione flag refresh: {e}")
    suffix = " · transizione" if transition_state else ""
    print(f"  ✓ Quadrante: {q}{suffix} (conviction: {regime_conviction:.0f})")
    return macro

# =============================================================================
#  REGIME DI MERCATO
# =============================================================================

def detect_market_regime(prices, macro):
    sp=prices["SP500"].dropna() if "SP500" in prices.columns else pd.Series(dtype=float)
    if len(sp)<100:
        return {"market_regime":"Transition","regime_confidence":40.0,"avg_corr_60d":np.nan,"breadth":np.nan,"trend_strength":np.nan,"market_vol_20d":np.nan}
    ema50=ema(sp,50); ema200=ema(sp,200)
    sp_hist=fetch_history("^GSPC"); adx_l,_,_=adx(sp_hist["High"],sp_hist["Low"],sp_hist["Close"]) if not sp_hist.empty else (pd.Series(dtype=float),None,None)
    trend_str=safe_float(adx_l.iloc[-1]) if len(adx_l.dropna()) else np.nan
    vol20=np.log(sp/sp.shift(1)).rolling(20).std()*np.sqrt(252)*100; mkt_vol=safe_float(vol20.iloc[-1]) if len(vol20.dropna()) else np.nan
    breadth_vals=[float(prices[c].dropna().iloc[-1]>ema(prices[c].dropna(),50).iloc[-1]) for c in prices.columns if prices[c].dropna().shape[0]>=60]
    breadth=np.mean(breadth_vals)*100 if breadth_vals else np.nan
    rets=np.log(prices/prices.shift(1)).dropna(how="all").dropna(axis=1,how="all")
    avg_corr=np.nan
    if len(rets)>=60 and rets.shape[1]>=5:
        corr_m=rets.tail(60).corr().values; vals=corr_m[np.triu_indices_from(corr_m,k=1)]; vals=vals[~np.isnan(vals)]
        avg_corr=float(np.mean(vals)) if len(vals) else np.nan
    bull=sp.iloc[-1]>ema50.iloc[-1]>ema200.iloc[-1]; bear=sp.iloc[-1]<ema50.iloc[-1]<ema200.iloc[-1]
    if bull and not pd.isna(trend_str) and trend_str>=20 and not pd.isna(breadth) and breadth>=55: regime="Trend Bullish"
    elif bear and not pd.isna(trend_str) and trend_str>=20 and not pd.isna(breadth) and breadth<=45: regime="Trend Bearish"
    elif (not pd.isna(mkt_vol) and mkt_vol>=28) or macro.get("stress_score",0)>=2.2: regime="High Volatility / Stress"
    elif not pd.isna(trend_str) and trend_str<18: regime="Mean Reversion / Range"
    else: regime="Transition"
    rc=clamp(50+(0 if pd.isna(trend_str) else min(abs((trend_str or 0)-20),15))+(0 if pd.isna(breadth) else min(abs((breadth or 50)-50)/2,15))+(0 if pd.isna(avg_corr) else min(abs(avg_corr)*25,15)),40,95)
    return {"market_regime":regime,"regime_confidence":rc,"avg_corr_60d":avg_corr,"breadth":breadth,"trend_strength":trend_str,"market_vol_20d":mkt_vol}

# =============================================================================
#  CORRELAZIONI + HEATMAP
# =============================================================================

def correlation_outputs(prices):
    """Calcola correlazioni su 30/60/90g. Niente heatmap — tabelle leggibili."""
    # Usa log returns con gestione NaN per evitare RuntimeWarning
    prices_clean = prices.replace(0, np.nan).dropna(how="all")
    with np.errstate(invalid='ignore', divide='ignore'):
        rets = np.log(prices_clean / prices_clean.shift(1)).replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(how="all").dropna(axis=1, how="all")
    out = {"returns": rets}

    for w in [30, 60, 90]:
        if len(rets) < w or rets.shape[1] < 5:
            out[f"corr_{w}"] = pd.DataFrame(); continue
        out[f"corr_{w}"] = rets.tail(w).corr()

    # Costruisci tabella coppie su 60g (base per top correlazioni)
    corr60 = out.get("corr_60"); pairs = []
    if isinstance(corr60, pd.DataFrame) and not corr60.empty:
        cols = list(corr60.columns)
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                # Aggiungi anche 30g e 90g per la console interattiva
                c30 = out["corr_30"].loc[cols[i], cols[j]] if isinstance(out.get("corr_30"), pd.DataFrame) and cols[i] in out["corr_30"].index and cols[j] in out["corr_30"].columns else np.nan
                c90 = out["corr_90"].loc[cols[i], cols[j]] if isinstance(out.get("corr_90"), pd.DataFrame) and cols[i] in out["corr_90"].index and cols[j] in out["corr_90"].columns else np.nan
                pairs.append({
                    "Asset A": cols[i], "Asset B": cols[j],
                    "Corr 30d": round(float(c30), 3) if not pd.isna(c30) else np.nan,
                    "Corr 60d": round(corr60.iloc[i, j], 3),
                    "Corr 90d": round(float(c90), 3) if not pd.isna(c90) else np.nan,
                })
    pair_df = pd.DataFrame(pairs).dropna(subset=["Corr 60d"]) if pairs else pd.DataFrame(columns=["Asset A","Asset B","Corr 30d","Corr 60d","Corr 90d"])
    out["top_positive"] = pair_df.sort_values("Corr 60d", ascending=False).head(20) if not pair_df.empty else pd.DataFrame()
    out["top_negative"] = pair_df.sort_values("Corr 60d", ascending=True).head(20)  if not pair_df.empty else pd.DataFrame()
    out["all_pairs"]    = pair_df  # usata dalla console interattiva
    return out

# img_b64 rimossa — heatmap non più generate

# =============================================================================
#  DASHBOARD HTML CON SWITCH TIMEFRAME
# =============================================================================

def render_table_rows(df):
    rows=[]
    for _,r in df.iterrows():
        sig=r["Signal"]
        if   sig=="HIGH CONVICTION BUY":  badge,fgc="#146c43","#d1fae5"
        elif sig=="BUY":                   badge,fgc="#14532d","#bbf7d0"
        elif sig=="HIGH CONVICTION SELL": badge,fgc="#7f1d1d","#fecaca"
        elif sig=="SELL":                  badge,fgc="#991b1b","#fecaca"
        else:                              badge,fgc="#78350f","#fde68a"
        p=r["Prezzo"]; v1d=r.get("Var 1D (%)",np.nan); v1m=r.get("Var 1M (%)",np.nan)
        rows.append(
            f"<tr>"
            f"<td class='left code'>{r['Codice']}</td>"
            f"<td class='left nm'>{r['Nome']}</td>"
            f"<td class='left'><span class='cat-badge'>{r['Categoria']}</span></td>"
            f"<td>{num_fmt(p)}</td>"
            f"<td class='{'pos' if not pd.isna(v1d) and v1d>0 else 'neg'}'>{pct_fmt(v1d)}</td>"
            f"<td class='{'pos' if not pd.isna(v1m) and v1m>0 else 'neg'}'>{pct_fmt(v1m)}</td>"
            f"<td>{num_fmt(r.get('RSI14',np.nan))}</td>"
            f"<td>{num_fmt(r.get('ADX14',np.nan))}</td>"
            f"<td class='left'>{r.get('Trend','—')}</td>"
            f"<td><b>{num_fmt(r['Score'])}</b></td>"
            f"<td>{num_fmt(r['Confidence'])}</td>"
            f"<td class='left'><span style='background:{badge};color:{fgc};padding:3px 10px;border-radius:999px;font-size:11px;font-weight:700'>{sig}</span></td>"
            f"<td class='left notes'>{r.get('Notes','')}</td>"
            f"</tr>"
        )
    return "\n".join(rows)

def build_dashboard(snapshots, macro, regime, corr):
    ts = now_str()
    q  = macro.get("quadrant","—")
    qdesc = macro.get("quadrant_desc","")
    fav   = macro.get("asset_favorites",[])
    mc    = macro.get("macro_confidence",np.nan)
    stay  = macro.get("stay_probability_1_3m",np.nan)
    next_q= macro.get("most_likely_next_quadrant","—")
    next_p= macro.get("next_quadrant_probability",np.nan)
    trans = macro.get("transition_candidates",[])
    fg    = macro.get("fear_greed",50)
    fgl   = macro.get("fear_greed_label","—")
    infl_delta = safe_float(macro.get("inflation_delta", np.nan))
    growth_delta = safe_float(macro.get("growth_delta", np.nan))
    real_yield = safe_float(macro.get("real_yield_proxy", np.nan))
    quad_strength = safe_float(macro.get("quadrant_strength", np.nan))
    gold_sp_ratio_3m = safe_float(macro.get("gold_sp_ratio_3m", np.nan))
    tips_3m = safe_float(macro.get("tips_3m", np.nan))
    regime_conviction = safe_float(macro.get("regime_conviction", np.nan))
    regime_transition = bool(macro.get("transition_state", False))
    quadrant_margin = safe_float(macro.get("quadrant_margin", np.nan))
    quadrant_scores = macro.get("quadrant_scores", {}) or {}
    second_q = macro.get("second_quadrant", next_q)

    q_grad={
        "Goldilocks / Reflazione": "135deg,#0d3320,#1a5c38",
        "Surriscaldamento":        "135deg,#3d1a00,#7a3500",
        "Stagflazione":            "135deg,#3d0000,#7a0000",
        "Recessione / Deflazione": "135deg,#0d1a3d,#1a2f6e",
    }.get(q,"135deg,#111,#222")

    macro_tags="".join(f"<span class='tag'>{x}</span>" for x in fav)
    trans_html="".join(f"<div style='margin-bottom:8px'><b>{x.get('quadrant','—')}</b>: {num_fmt(x.get('probability',np.nan))}%</div>" for x in trans) or "N/A"
    qscore_html = "".join(
        f"<div style='margin-bottom:8px'><b>{name}</b>: {num_fmt(val)}</div>"
        for name, val in sorted(quadrant_scores.items(), key=lambda kv: kv[1], reverse=True)
    ) or "N/A"

    tf_sections={}
    for tf, snap in snapshots.items():
        if snap.empty:
            tf_sections[tf]="<p style='color:#444;padding:1rem'>Dati insufficienti per questo timeframe.</p>"
            continue
        cats=snap["Categoria"].dropna().unique()
        sections=""
        sections+=f"<div class='section'><h2>Top opportunità — {tf}</h2><div class='table-wrap'><table><thead><tr><th class='left'>Codice</th><th class='left'>Nome</th><th class='left'>Cat.</th><th>Prezzo</th><th>1D</th><th>1M</th><th>RSI</th><th>ADX</th><th class='left'>Trend</th><th>Score</th><th>Conf.</th><th class='left'>Segnale</th><th class='left'>Note</th></tr></thead><tbody>{render_table_rows(snap.head(20))}</tbody></table></div></div>"
        for cat in cats:
            sub=snap[snap["Categoria"]==cat].copy()
            sections+=f"<div class='section'><h2>{cat} — {tf}</h2><div class='table-wrap'><table><thead><tr><th class='left'>Codice</th><th class='left'>Nome</th><th class='left'>Cat.</th><th>Prezzo</th><th>1D</th><th>1M</th><th>RSI</th><th>ADX</th><th class='left'>Trend</th><th>Score</th><th>Conf.</th><th class='left'>Segnale</th><th class='left'>Note</th></tr></thead><tbody>{render_table_rows(sub)}</tbody></table></div></div>"
        tf_sections[tf]=sections

    tp=corr.get("top_positive"); tn=corr.get("top_negative")
    tp_html="".join(f"<tr><td>{r['Asset A']}</td><td>{r['Asset B']}</td><td>{r['Corr 60d']:.3f}</td></tr>" for _,r in tp.head(12).iterrows()) if isinstance(tp,pd.DataFrame) and not tp.empty else ""
    tn_html="".join(f"<tr><td>{r['Asset A']}</td><td>{r['Asset B']}</td><td>{r['Corr 60d']:.3f}</td></tr>" for _,r in tn.head(12).iterrows()) if isinstance(tn,pd.DataFrame) and not tn.empty else ""
    n_assets=len(snapshots.get("Daily",pd.DataFrame()))
    # Costruisci dati per console correlazione interattiva
    all_pairs = corr.get("all_pairs", pd.DataFrame())
    assets_list = sorted(list(ASSETS.keys()))
    assets_opts = "".join(f"<option value='{a}'>{a} — {ASSETS[a]['name']}</option>" for a in assets_list)
    # Serializza coppie in JSON per il JS
    import json as _json
    corr_dict = {}
    if not all_pairs.empty:
        for _, row_ in all_pairs.iterrows():
            key = f"{row_['Asset A']}|{row_['Asset B']}"
            corr_dict[key] = {
                "c30": str(round(row_["Corr 30d"],3)) if not pd.isna(row_.get("Corr 30d",float("nan"))) else "null",
                "c60": str(round(row_["Corr 60d"],3)) if not pd.isna(row_.get("Corr 60d",float("nan"))) else "null",
                "c90": str(round(row_["Corr 90d"],3)) if not pd.isna(row_.get("Corr 90d",float("nan"))) else "null",
            }
    corr_json = _json.dumps(corr_dict, ensure_ascii=False)

    html=f"""<!DOCTYPE html>
<html lang='it'>
<head>
<meta charset='utf-8'>
<meta http-equiv='refresh' content='{INTERVAL_MINUTES*60}'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Sarada Trading System v3</title>
<link href='https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap' rel='stylesheet'>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#080808;color:#e6e6e6;font-family:'DM Sans',sans-serif;padding:24px;min-height:100vh}}
.header{{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;border-bottom:1px solid #1a1a1a;padding-bottom:20px;margin-bottom:24px;flex-wrap:wrap}}
.h1{{font-size:26px;font-weight:300;color:#fff;letter-spacing:-.02em;margin-bottom:4px}}
.sub{{color:#555;font-size:12px;font-family:'DM Mono',monospace}}
.fg-block{{text-align:right}}
.fg-num{{font-size:34px;font-weight:300;color:#fff;font-family:'DM Mono',monospace;line-height:1}}
.fg-lbl{{font-size:11px;color:#555;margin-top:2px}}
.fg-bar{{width:100px;height:5px;border-radius:3px;background:linear-gradient(to right,#e74c3c,#f39c12,#2ecc71);position:relative;margin-top:6px;margin-left:auto}}
.fg-n{{position:absolute;top:-5px;width:3px;height:15px;background:#fff;border-radius:2px;transform:translateX(-50%);left:{fg}%}}
.grid{{display:grid;grid-template-columns:2.1fr 1fr 1fr 1fr;gap:12px;margin-bottom:20px}}
.card{{background:#111;border:1px solid #1e1e1e;border-radius:14px;padding:18px}}
.big{{background:linear-gradient({q_grad});border-color:#ffffff18}}
.label{{color:#666;font-size:10px;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px}}
.value{{font-size:28px;color:#fff;font-weight:300;font-family:'DM Mono',monospace}}
.desc{{color:#aaa;font-size:12px;line-height:1.6;margin-top:6px}}
.tag{{display:inline-block;background:rgba(255,255,255,.08);color:rgba(255,255,255,.65);padding:4px 10px;margin:4px 4px 0 0;border-radius:999px;font-size:11px}}
.tf-switch{{display:flex;gap:8px;margin-bottom:20px;align-items:center}}
.tf-label{{font-size:11px;color:#555;text-transform:uppercase;letter-spacing:.1em;margin-right:4px}}
.tf-btn{{background:#111;border:1px solid #1e1e1e;color:#666;padding:7px 20px;border-radius:999px;font-size:12px;cursor:pointer;font-family:'DM Sans',sans-serif;transition:all .2s}}
.tf-btn:hover{{border-color:#333;color:#ccc}}
.tf-btn.active{{background:#fff;color:#000;border-color:#fff;font-weight:500}}
.tf-panel{{display:none}}.tf-panel.active{{display:block}}
.section{{margin-top:24px}}
.section h2{{font-size:16px;font-weight:400;color:#ccc;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #1a1a1a}}
.table-wrap{{overflow:auto;border:1px solid #1a1a1a;border-radius:12px;background:#0c0c0c}}
table{{width:100%;border-collapse:collapse;font-size:12px;min-width:1100px}}
th,td{{padding:9px 12px;border-bottom:1px solid #111;text-align:right;white-space:nowrap}}
th{{color:#555;background:#0f0f0f;font-size:10px;text-transform:uppercase;letter-spacing:.06em;font-weight:400;position:sticky;top:0}}
.left{{text-align:left}}.code{{font-weight:600;color:#fff;font-family:'DM Mono',monospace}}
.nm{{color:#999;max-width:160px;overflow:hidden;text-overflow:ellipsis}}
.notes{{color:#555;max-width:280px;white-space:normal;font-size:11px}}
.pos{{color:#4ade80}}.neg{{color:#f87171}}
.cat-badge{{background:#1a1a1a;color:#666;padding:2px 8px;border-radius:999px;font-size:10px;text-transform:uppercase;letter-spacing:.04em}}
tr:hover td{{background:rgba(255,255,255,.015)}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.heat{{width:100%;border-radius:10px;border:1px solid #1a1a1a}}
.regime-badge{{display:inline-block;background:#1a1a1a;color:#ccc;padding:6px 14px;border-radius:999px;font-size:12px;margin-top:8px}}
details.macro-detail{{background:#111;border:1px solid #1e1e1e;border-radius:14px;margin-bottom:20px;overflow:hidden}}
details.macro-detail summary{{padding:16px 20px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;font-size:14px;color:#888;font-weight:400;list-style:none;user-select:none}}
details.macro-detail summary::-webkit-details-marker{{display:none}}
details.macro-detail summary:hover{{color:#ccc}}
details.macro-detail[open] summary .chevron{{transform:rotate(180deg)}}
details.macro-detail .chevron{{transition:transform .2s;font-size:11px;color:#333}}
details.macro-detail .macro-detail-body{{padding:0 20px 20px}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr 1fr}}.two-col{{grid-template-columns:1fr}}body{{padding:12px}}}}
</style>
</head>
<body>
<div class='header'>
  <div>
    <div class='h1'>Sarada Trading System <span style='font-size:14px;color:#333'>v3.0</span></div>
    <div class='sub'>Aggiornato: {ts} &nbsp;·&nbsp; {n_assets} asset &nbsp;·&nbsp; refresh ogni {INTERVAL_MINUTES} min &nbsp;·&nbsp; Solo analisi quantitativa</div>
  </div>
  <div class='fg-block'>
    <div style='font-size:10px;color:#444;text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px'>Fear & Greed</div>
    <div class='fg-num'>{fg}</div>
    <div class='fg-lbl'>{fgl}</div>
    <div class='fg-bar'><div class='fg-n'></div></div>
  </div>
</div>
<div class='grid'>
  <div class='card big'>
    <div class='label' style='color:rgba(255,255,255,.35)'>Quadrante macro — Dalio framework</div>
    <div style='font-size:22px;font-weight:400;color:#fff;margin-bottom:6px'>{q}</div>
    <div class='desc'>{qdesc}</div>
    <div class='desc' style='margin-top:8px'>Secondo regime: <b style='color:#fff'>{second_q}</b> · margine {num_fmt(quadrant_margin)} · conviction {num_fmt(regime_conviction)}</div>
    <div class='desc' style='margin-top:4px'>{'Regime in transizione lieve' if regime_transition else 'Regime abbastanza definito'}</div>
    <div style='margin-top:10px'>{macro_tags}</div>
  </div>
  <div class='card'>
    <div class='label'>Macro confidence</div>
    <div class='value'>{num_fmt(mc)}</div>
    <div class='desc'>Growth {num_fmt(macro.get('growth_score',np.nan))} · Infl {num_fmt(macro.get('inflation_score',np.nan))} · Stress {num_fmt(macro.get('stress_score',np.nan))}</div>
  </div>
  <div class='card'>
    <div class='label'>Permanenza quadrante</div>
    <div class='value'>{num_fmt(stay)}%</div>
    <div class='desc'>Probabilità di restare in <b style='color:#ccc'>{q}</b> nei prossimi 1–3 mesi</div>
  </div>
  <div class='card'>
    <div class='label'>Prossimo quadrante probabile</div>
    <div style='font-size:17px;font-weight:400;color:#fff;margin-bottom:4px'>{next_q}</div>
    <div class='desc'>Prob. {num_fmt(next_p)}% &nbsp;·&nbsp; VIX {num_fmt(macro.get('vix',np.nan))} &nbsp;·&nbsp; DXY {num_fmt(macro.get('dxy',np.nan))}</div>
  </div>
</div>
<details class='macro-detail'>
  <summary>
    <span>📊 Dettaglio indicatori macro</span>
    <span class='chevron'>▼</span>
  </summary>
  <div class='macro-detail-body'>
    <div class='grid' style='grid-template-columns:repeat(4,1fr);margin-top:14px'>
      <div class='card'><div class='label'>CPI YoY</div><div class='value'>{num_fmt(macro.get('cpi_yoy',np.nan))}</div><div class='desc'>Inflazione headline annua</div></div>
      <div class='card'><div class='label'>Core CPI YoY</div><div class='value'>{num_fmt(macro.get('core_cpi_yoy',np.nan))}</div><div class='desc'>Inflazione core (ex food/energy)</div></div>
      <div class='card'><div class='label'>Disoccupazione</div><div class='value'>{num_fmt(macro.get('unemployment',np.nan))}%</div><div class='desc'>Tasso di disoccupazione USA</div></div>
      <div class='card'><div class='label'>Yield Spread 10Y-2Y</div><div class='value'>{num_fmt(macro.get('yield_spread',np.nan))}</div><div class='desc'>Negativo = curva invertita</div></div>
    </div>
    <div class='grid' style='grid-template-columns:repeat(4,1fr);margin-top:10px'>
      <div class='card'><div class='label'>Delta inflazione</div><div class='value'>{num_fmt(infl_delta)}</div><div class='desc'>Accelera (+) o rallenta (−)</div></div>
      <div class='card'><div class='label'>Delta crescita</div><div class='value'>{num_fmt(growth_delta)}</div><div class='desc'>Migliora (+) o deteriora (−)</div></div>
      <div class='card'><div class='label'>Real Yield</div><div class='value'>{num_fmt(real_yield)}</div><div class='desc'>10Y nominale − CPI. Negativo = supportivo</div></div>
      <div class='card'><div class='label'>Forza quadrante</div><div class='value'>{num_fmt(quad_strength)}</div><div class='desc'>Quanto è netto il regime attuale</div></div>
    </div>
    <div class='grid' style='grid-template-columns:repeat(4,1fr);margin-top:10px'>
      <div class='card'><div class='label'>Ratio Oro / SP500</div><div class='value'>{num_fmt(macro.get('gold_sp_ratio',np.nan))}</div><div class='desc'>Salente = risk-off / inflazione</div></div>
      <div class='card'><div class='label'>Gold/SP500 3M</div><div class='value'>{pct_fmt(gold_sp_ratio_3m)}</div><div class='desc'>Momentum relativo 3 mesi</div></div>
      <div class='card'><div class='label'>TIPS 3M</div><div class='value'>{pct_fmt(tips_3m)}</div><div class='desc'>Protezione inflazione richiesta</div></div>
      <div class='card'><div class='label'>Liquidity score</div><div class='value'>{num_fmt(macro.get('liquidity_score',np.nan))}</div><div class='desc'>Condizioni di liquidità sistemica</div></div>
    </div>
    <div class='two-col' style='margin-top:10px'>
      <div class='card'><div class='label'>Score comparativi quadranti</div><div class='desc'>{qscore_html}</div></div>
      <div class='card'><div class='label'>Transizione e regime</div><div class='desc'>
        Secondo quadrante: <b style='color:#ccc'>{second_q}</b><br>
        Margine: <b>{num_fmt(quadrant_margin)}</b> · Conviction: <b>{num_fmt(regime_conviction)}</b><br>
        Stato: <b>{'In transizione' if regime_transition else 'Definito'}</b><br>
        Regime mercato: <b style='color:#ccc'>{regime.get('market_regime','—')}</b> · Breadth {num_fmt(regime.get('breadth',np.nan))}%
      </div></div>
    </div>
    <div class='two-col' style='margin-top:10px'>
      <div class='card'><div class='label'>Permanenza stimata</div><div class='desc'><b style='color:#ccc'>{q}</b> · {num_fmt(stay)}% nei prossimi 1–3 mesi</div></div>
      <div class='card'><div class='label'>Quadranti alternativi</div><div class='desc'>{trans_html}</div></div>
    </div>
  </div>
</details>
<div class='section'>
  <h2>Ranking asset per vantaggio statistico</h2>
  <div class='tf-switch'>
    <span class='tf-label'>Timeframe:</span>
    <button class='tf-btn active' onclick='switchTF("Daily",this)'>Daily</button>
    <button class='tf-btn' onclick='switchTF("Weekly",this)'>Weekly</button>
    <button class='tf-btn' onclick='switchTF("Monthly",this)'>Monthly</button>
  </div>
  <div id='tf-Daily' class='tf-panel active'>{tf_sections.get("Daily","")}</div>
  <div id='tf-Weekly' class='tf-panel'>{tf_sections.get("Weekly","")}</div>
  <div id='tf-Monthly' class='tf-panel'>{tf_sections.get("Monthly","")}</div>
</div>
<div class='section'>
  <h2>Correlazioni tra asset (60 giorni)</h2>
  <div class='two-col'>
    <div class='card'><div class='label'>Top correlazioni positive</div><div class='table-wrap'><table style='min-width:100%'><thead><tr><th class='left'>Asset A</th><th class='left'>Asset B</th><th>Corr</th></tr></thead><tbody>{tp_html}</tbody></table></div></div>
    <div class='card'><div class='label'>Top correlazioni inverse (diversificazione)</div><div class='table-wrap'><table style='min-width:100%'><thead><tr><th class='left'>Asset A</th><th class='left'>Asset B</th><th>Corr</th></tr></thead><tbody>{tn_html}</tbody></table></div></div>
  </div>
</div>
<div class='section'>
  <h2>🔍 Console correlazione — confronta due asset</h2>
  <div class='card' style='max-width:700px'>
    <div style='display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap;margin-bottom:16px'>
      <div style='flex:1;min-width:160px'>
        <div class='label' style='margin-bottom:6px'>Asset A</div>
        <select id='corrA' onchange='calcCorr()' style='width:100%;background:#0c0c0c;border:1px solid #2a2a2a;color:#ccc;padding:8px 10px;border-radius:8px;font-size:13px;font-family:DM Mono,monospace'>
          {assets_opts}
        </select>
      </div>
      <div style='flex:1;min-width:160px'>
        <div class='label' style='margin-bottom:6px'>Asset B</div>
        <select id='corrB' onchange='calcCorr()' style='width:100%;background:#0c0c0c;border:1px solid #2a2a2a;color:#ccc;padding:8px 10px;border-radius:8px;font-size:13px;font-family:DM Mono,monospace'>
          {assets_opts}
        </select>
      </div>
    </div>
    <div id='corrResult' style='background:#0a0a0a;border:1px solid #1a1a1a;border-radius:10px;padding:16px;font-family:DM Mono,monospace'>
      <div style='color:#333;font-size:12px'>Seleziona due asset per vedere la correlazione</div>
    </div>
  </div>
</div>
<div style='margin-top:2rem;text-align:center;font-size:10px;color:#222;font-family:DM Mono,monospace;letter-spacing:.05em'>Sarada Trading System v3.1 &nbsp;·&nbsp; Solo analisi statistica — non consulenza finanziaria</div>
<script>
function switchTF(tf, btn) {{
  document.querySelectorAll('.tf-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tf-' + tf).classList.add('active');
  btn.classList.add('active');
}}

// Dati correlazione pre-calcolati
const CORR_DATA = {corr_json};

function corrColor(v) {{
  if (isNaN(v)) return '#444';
  if (v >= 0.8) return '#f87171';
  if (v >= 0.5) return '#fb923c';
  if (v >= 0.2) return '#facc15';
  if (v >= -0.2) return '#888';
  if (v >= -0.5) return '#60a5fa';
  return '#4ade80';
}}

function corrLabel(v) {{
  if (isNaN(v)) return 'N/D';
  if (v >= 0.8) return 'correlazione alta — si muovono insieme';
  if (v >= 0.5) return 'correlazione moderata';
  if (v >= 0.2) return 'correlazione lieve';
  if (v >= -0.2) return 'non correlati — buona diversificazione';
  if (v >= -0.5) return 'correlazione inversa moderata';
  return 'correlazione inversa forte — ottima diversificazione';
}}

function calcCorr() {{
  const a = document.getElementById('corrA').value;
  const b = document.getElementById('corrB').value;
  const div = document.getElementById('corrResult');
  if (!a || !b || a === b) {{
    div.innerHTML = "<div style='color:#333;font-size:12px'>Seleziona due asset diversi</div>";
    return;
  }}
  const key1 = a + '|' + b;
  const key2 = b + '|' + a;
  const d = CORR_DATA[key1] || CORR_DATA[key2];
  if (!d) {{
    div.innerHTML = "<div style='color:#555;font-size:12px'>Dati non disponibili per questa coppia</div>";
    return;
  }}
  const windows = [['30g', d.c30], ['60g', d.c60], ['90g', d.c90]];
  const boxes = windows.map(([lbl, v]) => {{
    const vn = parseFloat(v);
    const col = corrColor(vn);
    const disp = isNaN(vn) ? '—' : vn.toFixed(3);
    return `<div style='flex:1;background:#111;border:1px solid #1e1e1e;border-radius:10px;padding:12px 14px;text-align:center'>
      <div style='font-size:10px;color:#333;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px'>${{lbl}}</div>
      <div style='font-size:26px;font-weight:300;color:${{col}}'>${{disp}}</div>
    </div>`;
  }}).join('');
  const mainV = parseFloat(d.c60);
  div.innerHTML = `
    <div style='display:flex;gap:10px;margin-bottom:14px'>${{boxes}}</div>
    <div style='font-size:12px;color:${{corrColor(mainV)}};padding:8px 12px;background:#0c0c0c;border-radius:8px;border-left:3px solid ${{corrColor(mainV)}}'>
      ${{corrLabel(mainV)}}
    </div>`;
}}
</script>
</body>
</html>"""

    DASHBOARD_FILE.write_text(html, encoding="utf-8")
    print(f"\n  Dashboard: {DASHBOARD_FILE}")

def run_layer2_strategy(snapshots, history):
    """Esegue il Layer 2 operativo usando il file esterno collegato al Layer 1."""
    layer2_path = LAYER2_FILE if LAYER2_FILE.exists() else LAYER2_ALT_FILE
    if not layer2_path.exists():
        print(f"  ✗ Layer 2 non trovato: {LAYER2_FILE}")
        return None
    try:
        os.environ["JENNIFER_OUTPUT_DIR"] = str(OUTPUT_DIR)
        print(f"  [L2] Layer 2 output dir: {os.environ['JENNIFER_OUTPUT_DIR']}")
        print(f"  [L2] Avvio modulo: {layer2_path}")
        spec = importlib.util.spec_from_file_location("sarada_layer2_runtime", layer2_path)
        if spec is None or spec.loader is None:
            print("  ✗ Impossibile caricare il Layer 2")
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "generate_all_setups"):
            print("  ✗ Layer 2 caricato ma funzione generate_all_setups mancante")
            return None

        results = module.generate_all_setups(snapshots, history)
        if hasattr(module, "build_setup_dashboard"):
            module.build_setup_dashboard(results)
        print(f"  ✓ Layer 2 eseguito correttamente — output atteso in {OUTPUT_DIR}")
        return results
    except Exception as e:
        print(f"  ✗ Layer 2 error: {e}")
        return None

# =============================================================================
#  EXCEL
# =============================================================================

def save_excel(snapshots, macro, regime, corr):
    try:
        with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as writer:
            for tf,snap in snapshots.items():
                if not snap.empty:
                    snap.to_excel(writer, sheet_name=f"Ranking {tf}", index=False)
            pd.DataFrame([macro]).to_excel(writer, sheet_name="Macro", index=False)
            pd.DataFrame([regime]).to_excel(writer, sheet_name="Regime", index=False)
            tp=corr.get("top_positive"); tn=corr.get("top_negative")
            if isinstance(tp,pd.DataFrame) and not tp.empty: tp.to_excel(writer, sheet_name="Corr Positive", index=False)
            if isinstance(tn,pd.DataFrame) and not tn.empty: tn.to_excel(writer, sheet_name="Corr Negative", index=False)
        print(f"  Excel: {EXCEL_FILE}")
    except Exception as e:
        print(f"  ✗ Excel: {e}")

# =============================================================================
#  CICLO PRINCIPALE
# =============================================================================

def run_cycle():
    print("\n"+"="*70)
    print(f"  SARADA TRADING SYSTEM v3.1 — {now_str()}")
    print(f"  Ambiente: {'Cloud (GitHub Actions)' if IS_CLOUD else 'Locale (Windows)'}")
    print("="*70)

    history, prices = ingest_all()
    if prices.empty or len(history)<5:
        print("  Dati insufficienti. Riprovo al prossimo ciclo."); return

    macro  = build_macro_context(prices)
    regime = detect_market_regime(prices, macro)
    corr   = correlation_outputs(prices)

    print(f"\n[SNAPSHOT] Calcolo score multi-timeframe...")
    snapshots = build_snapshot_multitf(history, macro, regime, corr)

    save_excel(snapshots, macro, regime, corr)
    build_dashboard(snapshots, macro, regime, corr)

    print("\n[LAYER 2] Avvio operatività...")
    run_layer2_strategy(snapshots, history)

    daily=snapshots.get("Daily",pd.DataFrame())
    if not daily.empty:
        print("\n  TOP 8 — Daily:")
        for _,r in daily.head(8).iterrows():
            print(f"    {r['Codice']:10} Score:{r['Score']:5.1f}  Conf:{r['Confidence']:5.1f}  {r['Signal']}")
    print(f"\n  Prossimo ciclo tra {INTERVAL_MINUTES} minuti...")

# =============================================================================
#  AVVIO
# =============================================================================

def main():
    print("\n"+"="*70)
    print("  SARADA TRADING SYSTEM v3.2 — Macro Engine Upgraded")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Intervallo: {INTERVAL_MINUTES} minuti")
    print(f"  Asset: {len(ASSETS)}")
    print(f"  NOTA: Se il quadrante macro è errato, crea il file:")
    print(f"        {DATA_DIR / 'force_macro_refresh.flag'}")
    print(f"        per forzare il ricalcolo al prossimo ciclo.")
    print("="*70)

    if IS_CLOUD:
        # GitHub Actions: esegue una volta sola e termina
        run_cycle()
    else:
        # Locale: loop infinito
        while True:
            try:
                run_cycle()
            except KeyboardInterrupt:
                print("\n  Fermato dall'utente."); break
            except Exception as e:
                print(f"\n  [ERRORE] {e}")
            time.sleep(INTERVAL_MINUTES * 60)

if __name__ == "__main__":
    main()
