# -*- coding: utf-8 -*-
"""
Jennifer Trading System — Layer 2 v2.0 (Strategy Engine)
5 Blocchi: Candidate Selection, Setup Classifier, Trade Planner,
           Quality Engine, Exclusion Engine
"""
import os, sys, json, warnings
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)

IS_CLOUD = os.environ.get("GITHUB_ACTIONS")=="true" or "--single-run" in sys.argv
if IS_CLOUD:
    OUTPUT_DIR=Path("./docs"); DATA_DIR=Path("./data")
else:
    OUTPUT_DIR=Path(os.environ.get("JENNIFER_OUTPUT_DIR",r"C:\jennifer_trading")); DATA_DIR=OUTPUT_DIR/"data"
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

SETUP_FILE=OUTPUT_DIR/"jennifer_setups.html"
MIN_SCORE=55; MIN_CONFIDENCE=49; TOP_DAILY=12; TOP_WEEKLY=12; TOP_MONTHLY=6
MIN_RR=1.8; MAX_SL_PCT=12.0; MAX_CORR_EXCLUDE=0.88

def now_str(): return datetime.now().strftime("%d/%m/%Y %H:%M:%S")
def sf(x):
    try: return float(x)
    except: return np.nan
def fp(x):
    if pd.isna(x): return "—"
    ax=abs(x)
    if ax>=10000: return f"{x:,.0f}"
    if ax>=100: return f"{x:,.2f}"
    if ax>=1: return f"{x:.4f}"
    return f"{x:.6f}"

def _ema(s,n): return s.ewm(span=n,adjust=False).mean()
def _sma(s,n): return s.rolling(n).mean()
def _atr(h,l,c,n=14):
    pc=c.shift(1); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()
def _rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
    ag=g.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    al=l.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    return 100-(100/(1+ag/al.replace(0,np.nan)))
def _bol(s,n=20):
    m=_sma(s,n); std=s.rolling(n).std(); sup=m+2*std; inf=m-2*std; w=(2*std)/m.replace(0,np.nan)
    return sup,inf,m,w
def _zsc(s,n=20):
    m=s.rolling(n).mean(); sd=s.rolling(n).std().replace(0,np.nan); return (s-m)/sd
def _don(h,l,n=20): return h.rolling(n).max(),l.rolling(n).min()
def _fib(close,lb=90):
    s=close.dropna().tail(lb)
    if len(s)<15: return {}
    hi,lo=s.max(),s.min(); d=hi-lo
    if d==0: return {}
    return {"23.6":hi-d*.236,"38.2":hi-d*.382,"50.0":hi-d*.5,"61.8":hi-d*.618,"78.6":hi-d*.786}
def _sr(h,l,lbs=(10,20,50)):
    su,re=[],[]
    for lb in lbs:
        if len(h)>=lb: re.append(float(h.rolling(lb).max().iloc[-1])); su.append(float(l.rolling(lb).min().iloc[-1]))
    return sorted(set(su)),sorted(set(re))

def _nearest_level(levels,last,side=None,min_gap_pct=0.0):
    vals=[float(x) for x in levels if not pd.isna(x) and x>0]
    if side=="above": vals=[x for x in vals if x>=last*(1+min_gap_pct/100)]
    elif side=="below": vals=[x for x in vals if x<=last*(1-min_gap_pct/100)]
    if not vals: return np.nan
    return min(vals,key=lambda x:abs(x-last))

# Nuovi livelli swing: pivot locali, filtro vicinanza e separazione supporti/resistenze
def detect_swing_levels(high,low,window=3,max_levels=6,min_gap_pct=0.8):
    h=pd.Series(high).dropna(); l=pd.Series(low).dropna()
    if len(h)<window*2+3 or len(l)<window*2+3:
        return {"swing_supports":[], "swing_resistances":[], "last_swing_low":np.nan, "last_swing_high":np.nan}
    piv_hi=[]; piv_lo=[]
    for i in range(window, len(h)-window):
        hv=float(h.iloc[i]); lv=float(l.iloc[i])
        hs=h.iloc[i-window:i+window+1]; ls=l.iloc[i-window:i+window+1]
        if hv>=float(hs.max()) and (hs==hv).sum()==1: piv_hi.append((h.index[i], hv))
        if lv<=float(ls.min()) and (ls==lv).sum()==1: piv_lo.append((l.index[i], lv))
    def _compress(pivots):
        pivots=sorted(pivots, key=lambda x:x[0], reverse=True)
        kept=[]
        for dt,lv in pivots:
            if not kept:
                kept.append((dt,lv)); continue
            too_close=False
            for _,kv in kept:
                ref=max(abs(kv),1e-9)
                if abs(lv-kv)/ref*100 < min_gap_pct:
                    too_close=True
                    break
            if not too_close:
                kept.append((dt,lv))
            if len(kept)>=max_levels:
                break
        return kept
    hi_kept=_compress(piv_hi); lo_kept=_compress(piv_lo)
    return {
        "swing_supports": sorted([lv for _,lv in lo_kept]),
        "swing_resistances": sorted([lv for _,lv in hi_kept]),
        "last_swing_low": float(lo_kept[0][1]) if lo_kept else np.nan,
        "last_swing_high": float(hi_kept[0][1]) if hi_kept else np.nan,
    }

def _level_pack(high,low,close,lookback_bias=50):
    last=float(pd.Series(close).dropna().iloc[-1])
    su,re=_sr(high,low,(10,20,max(20,min(lookback_bias,len(close)-1))))
    swings=detect_swing_levels(high,low,window=3 if len(close)<80 else 4,min_gap_pct=0.75 if last<100 else 0.55)
    sup_all=sorted(set([x for x in swings["swing_supports"]+su if not pd.isna(x)]))
    res_all=sorted(set([x for x in swings["swing_resistances"]+re if not pd.isna(x)]))
    return {
        "swing_supports": swings["swing_supports"],
        "swing_resistances": swings["swing_resistances"],
        "sr_supports": su,
        "sr_resistances": re,
        "supports_all": sup_all,
        "resistances_all": res_all,
        "nearest_support": _nearest_level(swings["swing_supports"], last, side="below"),
        "nearest_resistance": _nearest_level(swings["swing_resistances"], last, side="above"),
        "fallback_support": _nearest_level(su, last, side="below"),
        "fallback_resistance": _nearest_level(re, last, side="above"),
        "last_swing_low": swings["last_swing_low"],
        "last_swing_high": swings["last_swing_high"],
    }

# Penalita su setup gia estesi rispetto a prezzo/EMA/swing/livello rotto
def compute_overextension_score(last,e20,last_swing_low,last_swing_high,ref_level,direction):
    pen=0.0
    if direction=="LONG":
        if not pd.isna(e20) and e20>0:
            d=(last/e20-1)*100
            if d>6: pen-=15
            elif d>5: pen-=11
            elif d>4: pen-=7
        if not pd.isna(last_swing_low) and last_swing_low>0:
            d=(last/last_swing_low-1)*100
            if d>12: pen-=6
            elif d>8: pen-=4
        if not pd.isna(ref_level) and ref_level>0:
            d=(last/ref_level-1)*100
            if d>3.2: pen-=8
            elif d>2.0: pen-=5
            elif d>1.2: pen-=3
    else:
        if not pd.isna(e20) and e20>0:
            d=(e20/last-1)*100
            if d>6: pen-=15
            elif d>5: pen-=11
            elif d>4: pen-=7
        if not pd.isna(last_swing_high) and last_swing_high>0:
            d=(last_swing_high/last-1)*100
            if d>12: pen-=6
            elif d>8: pen-=4
        if not pd.isna(ref_level) and ref_level>0:
            d=(ref_level/last-1)*100
            if d>3.2: pen-=8
            elif d>2.0: pen-=5
            elif d>1.2: pen-=3
    return int(max(-15, min(0, round(pen))))

def _setup_quality_type(distance_pct,conflicts,volatility_pct,pattern_clarity):
    score=0.0
    if distance_pct<=0.8: score+=2
    elif distance_pct<=1.6: score+=1
    elif distance_pct>=3.2: score-=2
    elif distance_pct>=2.2: score-=1
    if conflicts<=0: score+=1
    elif conflicts>=2: score-=2
    elif conflicts==1: score-=1
    if volatility_pct<=3.2: score+=1
    elif volatility_pct>=6.0: score-=1
    if pattern_clarity>=2: score+=1
    elif pattern_clarity<=0: score-=1
    if score>=3: return "clean"
    if score<=-1: return "dirty"
    return "average"


def resamp(df,rule):
    agg={c:("first" if c=="Open" else "max" if c=="High" else "min" if c=="Low" else "last" if c=="Close" else "sum") for c in ["Open","High","Low","Close","Volume"] if c in df.columns}
    return df.resample(rule).agg(agg).dropna(subset=["Close"])


# =============================================================================
#  NUOVI SCORE: CONFLUENCE / TREND POSITION / SPACE QUALITY
# =============================================================================

def compute_confluence_score(last, entry_mid, direction,
                              fib_levels, swing_supports, swing_resistances,
                              e20, e50, e200, bb_mid, bb_inf, bb_sup,
                              near_sup, near_res,
                              multi_fib_score=0, multi_fib_labels=None,
                              tl_score=0, tl_labels=None):
    """
    Confluence reale basata su cluster strutturali con pesi differenziati.
    Priorità: multi-fib > trendline > swing storico > EMA rilevante > Bollinger
    """
    score = 0; labels = []
    zone_lo = entry_mid * 0.985; zone_hi = entry_mid * 1.015
    def iz(v): return not pd.isna(v) and v > 0 and zone_lo <= v <= zone_hi

    # 1. Multi-fib confluence (peso massimo)
    if multi_fib_score > 0:
        score += min(multi_fib_score, 35)
        if multi_fib_labels: labels.extend(multi_fib_labels[:2])

    # 2. Trendline forte (peso alto)
    if tl_score > 0:
        score += min(tl_score, 25)
        if tl_labels: labels.extend(tl_labels[:1])

    # 3. Swing levels storici (peso alto — sono livelli "veri")
    sw_su=[s for s in swing_supports if iz(s)]
    sw_re=[r for r in swing_resistances if iz(r)]
    if direction=="LONG" and sw_su:
        score += min(len(sw_su)*10, 20)
        labels.append(f"Swing Sup x{len(sw_su)}")
    if direction=="SHORT" and sw_re:
        score += min(len(sw_re)*10, 20)
        labels.append(f"Swing Res x{len(sw_re)}")

    # 4. Supporto/resistenza strutturale rolling
    if direction=="LONG" and iz(near_sup): score+=8; labels.append("Supporto strutturale")
    if direction=="SHORT" and iz(near_res): score+=8; labels.append("Resistenza strutturale")

    # 5. EMA rilevanti (peso medio — solo se confermano, non determinano)
    ema_hits=[]
    if iz(e200): score+=10; ema_hits.append("EMA200")   # EMA200 = livello chiave
    elif iz(e50): score+=7; ema_hits.append("EMA50")
    elif iz(e20): score+=4; ema_hits.append("EMA20")
    if ema_hits: labels.append("+".join(ema_hits))

    # 6. Bollinger (peso basso — contestuale, non strutturale)
    if direction=="LONG" and iz(bb_inf): score+=3; labels.append("BB Inf")
    if direction=="SHORT" and iz(bb_sup): score+=3; labels.append("BB Sup")

    # Bonus cluster: se 3+ segnali distinti nello stesso punto
    distinct = len([x for x in [multi_fib_score>0, tl_score>0, bool(sw_su or sw_re),
                                  iz(near_sup) or iz(near_res), bool(ema_hits)] if x])
    if distinct >= 3: score += 12; labels.append("★ cluster forte")
    elif distinct >= 2: score += 5

    return min(score, 70), labels


def compute_trend_position_score(last, e20, e50, e200, hi_series, lo_series,
                                  leg_data=None):
    """
    Valuta posizione nel trend integrando dati di gamba.
    Usa quanto è avanzata la gamba corrente, non solo distanza dai massimi.
    """
    score = 0; label = "Mid Trend"

    # Posizione rispetto ai massimi recenti
    if len(hi_series)>=20:
        hi20=float(hi_series.rolling(20).max().iloc[-1])
        dist_hi=(hi20-last)/hi20*100 if hi20>0 else np.nan
    else: dist_hi=np.nan

    # EMA stack
    bull_stack=(not pd.isna(e20) and not pd.isna(e50) and not pd.isna(e200) and last>e20>e50>e200)
    bear_stack=(not pd.isna(e20) and not pd.isna(e50) and not pd.isna(e200) and last<e20<e50<e200)

    # Usa dati di gamba se disponibili
    leg_phase="unknown"; retr_pct=np.nan
    if leg_data:
        leg_phase=leg_data.get("leg_phase","unknown")
        retr_pct=sf(leg_data.get("retracement_pct",np.nan))

    # Classificazione primaria dalla gamba
    if leg_phase in ["ritracciamento_ideale","rimbalzo_ideale"]:
        label="Pullback Zone (ideale)"; score=16
    elif leg_phase in ["ritracciamento_leggero","rimbalzo_leggero"]:
        label="Early Pullback"; score=12
    elif leg_phase in ["impulso","impulso_ribassista"]:
        if not pd.isna(dist_hi) and dist_hi < 3:
            label="Late Trend (impulso esteso)"; score=-6
        else:
            label="Trend in Impulso"; score=6
    elif leg_phase in ["ritracciamento_profondo","rimbalzo_profondo"]:
        label="Pullback Profondo"; score=3
    else:
        # Fallback: usa distanza dai massimi
        if not pd.isna(dist_hi):
            if dist_hi>=8: label="Early Trend"; score=12
            elif dist_hi>=3: label="Mid Trend"; score=7
            else: label="Late Trend"; score=-4
        else:
            label="Mid Trend"; score=4

    if bull_stack or bear_stack: score+=4; label+=" (stack)"
    return score, label


def compute_space_quality_score(entry_mid, tp1, direction, resistances_all, supports_all):
    if pd.isna(entry_mid) or pd.isna(tp1) or entry_mid<=0: return 0
    if direction=="LONG":
        obs=[r for r in resistances_all if not pd.isna(r) and entry_mid<r<tp1]
    else:
        obs=[s for s in supports_all if not pd.isna(s) and tp1<s<entry_mid]
    n=len(obs); total=abs(tp1-entry_mid)
    if n==0: return 12
    elif n==1:
        od=abs(obs[0]-entry_mid)/total if total>0 else 0
        return 6 if od>0.7 else 2
    elif n==2: return -2
    else: return -6

# =============================================================================
#  NUOVE FUNZIONI STRUTTURALI v3
# =============================================================================

def detect_trend_legs(close, high, low, window=5):
    """
    Identifica l'ultima 'gamba' del trend: impulso + ritracciamento.
    Restituisce swing high/low della gamba, ampiezza, percentuale ritracciamento.
    """
    cl=pd.Series(close).dropna(); hi=pd.Series(high).dropna(); lo=pd.Series(low).dropna()
    n=len(cl)
    if n<20:
        return {"leg_swing_high":np.nan,"leg_swing_low":np.nan,"leg_amplitude":np.nan,
                "retracement_pct":np.nan,"leg_phase":"unknown","retracement_quality":"unknown"}

    w=min(window, max(2, n//8))
    pivots_hi=[]; pivots_lo=[]
    for i in range(w, n-w):
        hv=float(hi.iloc[i]); lv=float(lo.iloc[i])
        if hv==float(hi.iloc[i-w:i+w+1].max()) and list(hi.iloc[i-w:i+w+1]).count(hv)==1:
            pivots_hi.append((i,hv))
        if lv==float(lo.iloc[i-w:i+w+1].min()) and list(lo.iloc[i-w:i+w+1]).count(lv)==1:
            pivots_lo.append((i,lv))

    last=float(cl.iloc[-1])
    leg_high=np.nan; leg_low=np.nan; leg_amp=np.nan; retr_pct=np.nan
    phase="unknown"

    if pivots_hi and pivots_lo:
        # Trova ultimo swing high e ultimo swing low significativi
        last_hi_idx, last_hi_val = pivots_hi[-1]
        last_lo_idx, last_lo_val = pivots_lo[-1]

        if last_hi_idx > last_lo_idx:
            # Ultimo movimento è stato rialzista (impulso up poi ritracciamento)
            leg_high = last_hi_val
            # Cerca swing low prima del swing high
            pre_hi_lows = [(i,v) for i,v in pivots_lo if i < last_hi_idx]
            if pre_hi_lows:
                leg_low = pre_hi_lows[-1][1]
                leg_amp = leg_high - leg_low
                if leg_amp > 0:
                    retr_pct = (leg_high - last) / leg_amp * 100
                    retr_pct = min(max(retr_pct, 0), 120)
                    if retr_pct < 15: phase = "impulso"
                    elif retr_pct <= 40: phase = "ritracciamento_leggero"
                    elif retr_pct <= 68: phase = "ritracciamento_ideale"
                    else: phase = "ritracciamento_profondo"
        else:
            # Ultimo movimento è stato ribassista
            leg_low = last_lo_val
            pre_lo_highs = [(i,v) for i,v in pivots_hi if i < last_lo_idx]
            if pre_lo_highs:
                leg_high = pre_lo_highs[-1][1]
                leg_amp = leg_high - leg_low
                if leg_amp > 0:
                    retr_pct = (last - leg_low) / leg_amp * 100
                    retr_pct = min(max(retr_pct, 0), 120)
                    if retr_pct < 15: phase = "impulso_ribassista"
                    elif retr_pct <= 40: phase = "rimbalzo_leggero"
                    elif retr_pct <= 68: phase = "rimbalzo_ideale"
                    else: phase = "rimbalzo_profondo"

    # Qualità del ritracciamento per setup LONG
    if not pd.isna(retr_pct) and phase in ["ritracciamento_ideale","ritracciamento_leggero"]:
        if 35 <= retr_pct <= 65: rq = "ideale"
        elif retr_pct < 35: rq = "forza_alta"
        else: rq = "profondo_ma_sano"
    elif not pd.isna(retr_pct) and retr_pct > 70:
        rq = "rischio_debolezza"
    else:
        rq = "neutro"

    return {
        "leg_swing_high": leg_high,
        "leg_swing_low": leg_low,
        "leg_amplitude": leg_amp,
        "retracement_pct": round(retr_pct, 1) if not pd.isna(retr_pct) else np.nan,
        "leg_phase": phase,
        "retracement_quality": rq,
    }


def compute_retracement_quality_score(retr_pct, phase, direction):
    """
    Score basato sul ritracciamento della gamba corrente.
    Zona ideale 38.2-61.8% → score alto.
    """
    if pd.isna(retr_pct): return 0

    if direction == "LONG":
        if 36 <= retr_pct <= 65: return 16   # zona d'oro
        elif 24 <= retr_pct < 36: return 10   # pullback leggero — forza alta
        elif 65 < retr_pct <= 80: return 4    # profondo ma ancora ok
        elif retr_pct > 80: return -8          # rischio inversione
        else: return 6  # impulso in corso
    else:
        if 36 <= retr_pct <= 65: return 16
        elif 24 <= retr_pct < 36: return 10
        elif 65 < retr_pct <= 80: return 4
        elif retr_pct > 80: return -8
        else: return 6


def compute_multi_fib_confluence(last, fib_short, fib_long, direction, tolerance=0.015):
    """
    Cerca coincidenze tra Fibonacci della gamba attuale e Fibonacci del movimento ampio.
    Coincidenza = alta qualità strutturale.
    """
    score = 0; labels = []

    def levels_near(v1, v2):
        if pd.isna(v1) or pd.isna(v2) or v1<=0 or v2<=0: return False
        return abs(v1/v2 - 1) < tolerance

    def near_last(v):
        if pd.isna(v) or v<=0: return False
        return abs(last/v - 1) < 0.018

    # Cerca livelli fib vicini al prezzo attuale
    short_near = {k:v for k,v in fib_short.items() if near_last(v)}
    long_near  = {k:v for k,v in fib_long.items()  if near_last(v)}

    # Coincidenze tra i due set di fib
    for sk, sv in short_near.items():
        for lk, lv in long_near.items():
            if levels_near(sv, lv):
                score += 20  # coincidenza multi-fib = molto forte
                labels.append(f"MultiFib {sk}/{lk}")

    # Singoli livelli vicini al prezzo
    if short_near and not labels:
        score += 8
        labels.append(f"Fib gamba {list(short_near.keys())[0]}")
    if long_near and not labels:
        score += 6
        labels.append(f"Fib macro {list(long_near.keys())[0]}")

    return min(score, 35), labels


def detect_trendlines(high, low, min_touches=3, lookback=60):
    """
    Rileva trendline significative cercando rette che toccano almeno min_touches pivot.
    Restituisce lista di trendline con numero touch e distanza prezzo-linea attuale.
    """
    hi=pd.Series(high).dropna().tail(lookback)
    lo=pd.Series(low).dropna().tail(lookback)
    n=len(hi); last_idx=n-1
    if n < 15: return []

    last_hi=float(hi.iloc[-1]); last_lo=float(lo.iloc[-1])
    last_price=(last_hi+last_lo)/2

    # Pivot semplici
    piv_hi=[(i,float(hi.iloc[i])) for i in range(2,n-2) if float(hi.iloc[i])==float(hi.iloc[i-2:i+3].max())]
    piv_lo=[(i,float(lo.iloc[i])) for i in range(2,n-2) if float(lo.iloc[i])==float(lo.iloc[i-2:i+3].min())]

    trendlines=[]

    def check_line(pivots, line_type):
        for i in range(len(pivots)):
            for j in range(i+1, len(pivots)):
                x1,y1=pivots[i]; x2,y2=pivots[j]
                if x2-x1 < 5: continue  # distanza minima tra punti
                slope=(y2-y1)/(x2-x1)
                intercept=y1-slope*x1
                # Conta touch
                touches=0; distances=[]
                for xk,yk in pivots:
                    y_line=slope*xk+intercept
                    tol=abs(yk)*0.008  # 0.8% tolerance
                    if abs(yk-y_line)<tol: touches+=1
                    if xk==last_idx:
                        distances.append(abs(yk-y_line)/max(abs(yk),1e-9)*100)
                if touches>=min_touches:
                    y_now=slope*last_idx+intercept
                    dist_now=abs(last_price-y_now)/max(abs(last_price),1e-9)*100
                    trendlines.append({
                        "type":line_type,"touches":touches,"slope":round(slope,6),
                        "y_now":round(y_now,6),"dist_pct":round(dist_now,3)
                    })

    check_line(piv_hi, "resistance_trendline")
    check_line(piv_lo, "support_trendline")

    # Ordina per touches e distanza
    trendlines.sort(key=lambda x: (-x["touches"], x["dist_pct"]))
    return trendlines[:4]  # max 4 trendline significative


def compute_trendline_score(trendlines, direction, tolerance_pct=2.5):
    """
    Punteggio basato su trendline vicine al prezzo.
    Trendline con molti touch e prezzo vicino = score alto.
    """
    score=0; labels=[]
    for tl in trendlines:
        if tl["dist_pct"] > tolerance_pct: continue
        touches=tl["touches"]
        t_type=tl["type"]
        touch_bonus=min(touches*4, 16)
        prox_bonus=max(0, int((tolerance_pct - tl["dist_pct"]) / tolerance_pct * 8))
        if direction=="LONG" and t_type=="support_trendline":
            score += touch_bonus + prox_bonus
            labels.append(f"TL supporto ({touches}T)")
        elif direction=="SHORT" and t_type=="resistance_trendline":
            score += touch_bonus + prox_bonus
            labels.append(f"TL resistenza ({touches}T)")
    return min(score, 25), labels


def compute_structure_alignment_score(direction, trend_label, leg_phase,
                                       retr_quality, setup_name, tf_align_score):
    """
    Valuta la coerenza strutturale tra: trend, gamba, setup, TF.
    Incoerenza forte → penalizzazione.
    """
    score=0

    # Trend e direzione setup
    is_long=(direction=="LONG")
    if is_long and trend_label in ["Trend Bullish","Bullish"]: score+=6
    elif not is_long and trend_label in ["Trend Bearish","Bearish"]: score+=6
    elif trend_label=="Range": score+=1
    else: score-=6  # trend contro il setup

    # Fase gamba e tipo setup
    if setup_name in ["Pullback su Trend","Pullback Strutturale","Pullback Strutturale Short"]:
        if leg_phase in ["ritracciamento_ideale","ritracciamento_leggero"]: score+=8
        elif leg_phase in ["ritracciamento_profondo"]: score+=2
        elif leg_phase in ["impulso","impulso_ribassista"]: score-=4  # entry in impulso, non pullback
    elif setup_name=="Breakout":
        if leg_phase in ["impulso","ritracciamento_leggero"]: score+=5
        else: score+=1
    elif setup_name in ["Mean Reversion","Mean Reversion Short"]:
        if retr_quality in ["rischio_debolezza","profondo_ma_sano"]: score+=5
        else: score+=1
    elif setup_name in ["Late Trend Long","Late Trend Short"]:
        score-=3  # late trend sempre meno coerente

    # Qualità ritracciamento
    if retr_quality=="ideale": score+=5
    elif retr_quality=="forza_alta": score+=3
    elif retr_quality=="rischio_debolezza": score-=5

    # TF alignment
    if tf_align_score >= 15: score+=4
    elif tf_align_score >= 8: score+=2
    elif tf_align_score <= -10: score-=5

    return max(-20, min(20, score))



# =============================================================================
#  MOTORE STRUTTURALE — logica del trader umano
#  Basato su: XLB / Argento / Bitcoin / Wheat / analisi reale
# =============================================================================

def read_two_structures(cl, hi, lo):
    """
    Legge due strutture temporali come farebbe un trader guardando il grafico:
    - Struttura MACRO: l'intera storia disponibile (trova la grande gamba)
    - Struttura INTERMEDIA: l'ultima gamba significativa (~120 barre)
    
    Restituisce Fibonacci di ritracciamento e ESTENSIONI per entrambe.
    Le estensioni sono i target realistici (massimi storici, 1.272, 1.618).
    """
    n = len(cl)
    last = float(cl.iloc[-1])

    def fib_levels(lo_val, hi_val, direction="up"):
        if pd.isna(lo_val) or pd.isna(hi_val) or lo_val<=0 or hi_val<=0: return {}
        d = abs(hi_val - lo_val)
        if d == 0: return {}
        if direction == "up":
            return {
                # Ritracciamenti (zone entry pullback)
                "retr_236": hi_val - d*0.236,
                "retr_382": hi_val - d*0.382,
                "retr_500": hi_val - d*0.500,
                "retr_618": hi_val - d*0.618,
                "retr_786": hi_val - d*0.786,
                # Estensioni (target ambiziosi)
                "ext_1000": hi_val,              # ritorno ai massimi
                "ext_1272": lo_val + d*1.272,
                "ext_1618": lo_val + d*1.618,
                "ext_2618": lo_val + d*2.618,
            }
        else:
            return {
                "retr_236": lo_val + d*0.236,
                "retr_382": lo_val + d*0.382,
                "retr_500": lo_val + d*0.500,
                "retr_618": lo_val + d*0.618,
                "retr_786": lo_val + d*0.786,
                "ext_1000": lo_val,
                "ext_1272": hi_val - d*1.272,
                "ext_1618": hi_val - d*1.618,
            }

    # ── STRUTTURA MACRO ───────────────────────────────────────────────────────
    lb_macro = min(500, n)
    hi_macro = hi.tail(lb_macro); lo_macro = lo.tail(lb_macro)
    macro_abs_hi = float(hi_macro.max()); macro_abs_lo = float(lo_macro.min())
    macro_hi_idx = int(hi_macro.values.argmax()); macro_lo_idx = int(lo_macro.values.argmin())
    macro_dir = "up" if macro_hi_idx > macro_lo_idx else "down"
    macro_fib = fib_levels(macro_abs_lo, macro_abs_hi, macro_dir)

    # ── STRUTTURA INTERMEDIA ──────────────────────────────────────────────────
    lb_mid = min(120, n)
    hi_mid = hi.tail(lb_mid); lo_mid = lo.tail(lb_mid)
    mid_hi = float(hi_mid.max()); mid_lo = float(lo_mid.min())
    mid_hi_idx = int(hi_mid.values.argmax()); mid_lo_idx = int(lo_mid.values.argmin())
    mid_dir = "up" if mid_hi_idx > mid_lo_idx else "down"
    mid_fib = fib_levels(mid_lo, mid_hi, mid_dir)

    return {
        "macro_lo": macro_abs_lo, "macro_hi": macro_abs_hi, "macro_dir": macro_dir,
        "macro_fib": macro_fib,
        "mid_lo": mid_lo, "mid_hi": mid_hi, "mid_dir": mid_dir,
        "mid_fib": mid_fib,
    }


def find_real_confluence_zone(last, structs, e20, e50, e200,
                               trendlines, swing_supports, swing_resistances,
                               direction, atr_val):
    """
    Trova zone di VERA confluenza strutturale — come farebbe il trader.
    
    Regola: una zona vale solo se ha ALMENO 2 livelli diversi che coincidono:
    - Fib struttura macro + Fib gamba intermedia → massima qualità
    - Fib + trendline storica → alta qualità  
    - Fib + MA rilevante (50/200) → buona qualità
    - Fib + swing level storico → buona qualità
    
    Se non trova almeno 2 confluenze → nessuna zona valida.
    Restituisce lista di zone ordinate per qualità, ognuna con:
    (prezzo_zona, score_confluenza, lista_motivi)
    """
    tol = 0.020  # 2% tolerance — realistico per confluenze

    def near(a, b):
        if pd.isna(a) or pd.isna(b) or a<=0 or b<=0: return False
        return abs(a/b - 1) <= tol

    zones = []  # (price, score, reasons)

    # Livelli Fibonacci macro e intermedi separati per tipo
    macro_retr = {k: v for k,v in structs["macro_fib"].items() if "retr" in k}
    macro_ext  = {k: v for k,v in structs["macro_fib"].items() if "ext"  in k}
    mid_retr   = {k: v for k,v in structs["mid_fib"].items()   if "retr" in k}
    mid_ext    = {k: v for k,v in structs["mid_fib"].items()   if "ext"  in k}

    # Tutti i livelli candidati con il loro "tipo"
    all_levels = []
    for k, v in macro_retr.items():
        if not pd.isna(v) and v > 0: all_levels.append((v, f"Fib macro {k.replace('retr_','0.')}",  "fib_macro", 12))
    for k, v in mid_retr.items():
        if not pd.isna(v) and v > 0: all_levels.append((v, f"Fib gamba {k.replace('retr_','0.')}",   "fib_mid",   10))
    for ma_v, ma_n, ma_w in [(e200,"EMA200",9),(e50,"EMA50",7),(e20,"EMA20",4)]:
        if not pd.isna(ma_v) and ma_v>0: all_levels.append((ma_v, ma_n, "ema", ma_w))
    for sv in swing_supports:
        if not pd.isna(sv) and sv>0: all_levels.append((sv, f"supporto swing ({fp(sv)})", "swing", 8))
    for rv in swing_resistances:
        if not pd.isna(rv) and rv>0: all_levels.append((rv, f"resistenza swing ({fp(rv)})", "swing_res", 8))
    for tl in trendlines:
        yn = tl.get("y_now", 0); t = tl.get("touches", 0); tp = tl.get("type","")
        if yn > 0 and t >= 3:
            all_levels.append((yn, f"TL {tp.replace('_trendline','')} ({t} touch)", "trendline", 6+t*2))

    # Filtra per direzione e distanza realistica dal prezzo attuale
    # Per LONG: cerca livelli SOTTO il prezzo (zone di pullback)
    # Per SHORT: cerca livelli SOPRA il prezzo
    if direction == "LONG":
        candidates = [(v, lbl, typ, w) for v, lbl, typ, w in all_levels
                      if v < last * 1.02 and v > last * 0.60]  # non troppo lontano
    else:
        candidates = [(v, lbl, typ, w) for v, lbl, typ, w in all_levels
                      if v > last * 0.98 and v < last * 1.40]

    # Per ogni livello candidato, cerca confluenze con altri livelli
    processed = set()
    for i, (v1, lbl1, typ1, w1) in enumerate(candidates):
        if i in processed: continue
        cluster_prices = [v1]; cluster_reasons = [lbl1]
        cluster_types  = {typ1}; cluster_score = w1

        for j, (v2, lbl2, typ2, w2) in enumerate(candidates):
            if i == j or j in processed: continue
            if near(v1, v2) and typ2 not in cluster_types:
                cluster_prices.append(v2); cluster_reasons.append(lbl2)
                cluster_types.add(typ2); cluster_score += w2
                processed.add(j)

        # Zona valida solo con almeno 2 tipi diversi
        if len(cluster_types) >= 2:
            zone_price = float(np.mean(cluster_prices))
            # Bonus per combinazioni premium
            if "fib_macro" in cluster_types and "fib_mid" in cluster_types:
                cluster_score += 20  # Fib macro + Fib gamba = massima qualità
            if "trendline" in cluster_types:
                cluster_score += 10  # Trendline sempre importante
            zones.append((zone_price, cluster_score, cluster_reasons))
        processed.add(i)

    zones.sort(key=lambda x: x[1], reverse=True)
    return zones


def compute_structural_sl(entry, zones, swing_supports, swing_resistances,
                           direction, atr_val, structs):
    """
    Stop loss strutturale — sempre sul lato opposto all'entry.
    LONG: sl DEVE essere < entry. SHORT: sl DEVE essere > entry.
    
    Logica come il trader:
    - Argento: sotto il doppio massimo precedente (ora supporto)
    - Bitcoin: sotto i due livelli 0.618 che formano il canale
    - Wheat: sotto il 0.618 della gamba, coincidente con trendline
    - XLB: sotto il supporto di zona con trendline 6 touch
    """
    buffer = 0.006  # 0.6% oltre il livello strutturale

    if direction == "LONG":
        # SL deve stare SOTTO l'entry — sempre
        sl_candidates = []

        # 1. Fib 0.618 della struttura intermedia (come Wheat)
        fib618_mid = structs["mid_fib"].get("retr_618", np.nan)
        if not pd.isna(fib618_mid) and fib618_mid < entry:
            sl_candidates.append((fib618_mid, "fib 0.618 gamba"))

        # 2. Swing lows sotto l'entry (massimi precedenti diventati supporto)
        sw_below = sorted([s for s in swing_supports if s < entry * 0.997], reverse=True)
        for sv in sw_below[:3]:
            sl_candidates.append((sv, f"swing low ({fp(sv)})"))

        # 3. Fib 0.786 macro (sotto la zona entry — zona di invalidazione profonda)
        fib786_macro = structs["macro_fib"].get("retr_786", np.nan)
        if not pd.isna(fib786_macro) and fib786_macro < entry:
            sl_candidates.append((fib786_macro, "fib 0.786 macro"))

        # 4. Fib 0.618 macro
        fib618_macro = structs["macro_fib"].get("retr_618", np.nan)
        if not pd.isna(fib618_macro) and fib618_macro < entry:
            sl_candidates.append((fib618_macro, "fib 0.618 macro"))

        if sl_candidates:
            # Prendi il più vicino SOTTO l'entry (stop stretto = R/R migliore)
            valid = [(p, lbl) for p, lbl in sl_candidates if p < entry]
            if valid:
                valid.sort(key=lambda x: x[0], reverse=True)  # il più alto sotto entry
                best_p, best_lbl = valid[0]
                sl = best_p * (1 - buffer)  # un po' oltre per i falsi breakout
            else:
                sl = entry * (1 - 0.03)  # fallback 3%
        else:
            sl = entry - max(1.5 * atr_val, entry * 0.025)

        # Garanzia assoluta: sl deve stare SOTTO entry
        if sl >= entry:
            sl = entry * (1 - 0.03)

    else:  # SHORT — sl DEVE stare SOPRA entry
        sl_candidates = []

        fib618_mid = structs["mid_fib"].get("retr_618", np.nan)
        if not pd.isna(fib618_mid) and fib618_mid > entry:
            sl_candidates.append((fib618_mid, "fib 0.618 gamba"))

        sw_above = sorted([r for r in swing_resistances if r > entry * 1.003])
        for rv in sw_above[:3]:
            sl_candidates.append((rv, f"swing high ({fp(rv)})"))

        fib618_macro = structs["macro_fib"].get("retr_618", np.nan)
        if not pd.isna(fib618_macro) and fib618_macro > entry:
            sl_candidates.append((fib618_macro, "fib 0.618 macro"))

        if sl_candidates:
            valid = [(p, lbl) for p, lbl in sl_candidates if p > entry]
            if valid:
                valid.sort(key=lambda x: x[0])  # il più basso sopra entry
                best_p, best_lbl = valid[0]
                sl = best_p * (1 + buffer)
            else:
                sl = entry * (1 + 0.03)
        else:
            sl = entry + max(1.5 * atr_val, entry * 0.025)

        # Garanzia assoluta: sl deve stare SOPRA entry
        if sl <= entry:
            sl = entry * (1 + 0.03)

    sl_pct = abs(entry - sl) / entry * 100 if entry > 0 else 5.0
    return round(sl, 6), round(sl_pct, 2)


def compute_realistic_tp(entry, sl, direction, structs,
                          swing_resistances, swing_supports,
                          hi_series, lo_series, confluence_score):
    """
    Target su livelli che il mercato ha già visto — mai oltre i massimi storici.
    
    Gerarchia LONG (dal più credibile al meno):
    1. Fib 0.236 del macro trend (ritorno parziale — es. Wheat)
    2. Massimo della struttura intermedia (ritorno ai massimi recenti)
    3. Massimo della struttura macro (massimo storico disponibile)
    4. Swing resistenza storica significativa sotto il massimo
    5. Fib 0.382 macro come target intermedio
    VIETATO: estensioni oltre i massimi (1.272, 1.618, 2.618)
    
    Regola R/R: minimo 2.0x. Sotto → fallback su livello più conservativo.
    """
    risk = abs(entry - sl)
    if risk <= 0 or risk < entry * 0.003:
        risk = entry * 0.025

    macro_hi = structs.get("macro_hi", np.nan)
    macro_lo = structs.get("macro_lo", np.nan)
    mid_hi   = structs.get("mid_hi", np.nan)
    mid_lo   = structs.get("mid_lo", np.nan)

    tp_candidates = []  # (price, rr, rationale)

    if direction == "LONG":
        # Tutti i candidati devono essere SOTTO o AL massimo storico
        max_allowed = macro_hi if not pd.isna(macro_hi) else float("inf")

        # 1. Fib 0.236 del macro trend — livello già visto (come Wheat)
        retr236 = structs["macro_fib"].get("retr_236", np.nan)
        if not pd.isna(retr236) and entry < retr236 <= max_allowed:
            rr = (retr236 - entry) / risk
            tp_candidates.append((retr236, rr, f"fib 0.236 macro ({fp(retr236)}) — resistenza storica confermata"))

        # 2. Fib 0.382 macro — livello intermedio storico
        retr382 = structs["macro_fib"].get("retr_382", np.nan)
        if not pd.isna(retr382) and entry < retr382 <= max_allowed:
            rr = (retr382 - entry) / risk
            tp_candidates.append((retr382, rr, f"fib 0.382 macro ({fp(retr382)}) — primo livello strutturale"))

        # 3. Massimo struttura intermedia (ritorno ai massimi recenti — es. Argento, Bitcoin)
        if not pd.isna(mid_hi) and entry < mid_hi <= max_allowed * 1.001:
            rr = (mid_hi - entry) / risk
            tp_candidates.append((mid_hi, rr, f"ritorno ai massimi recenti ({fp(mid_hi)}) — già visto dal mercato"))

        # 4. Massimo struttura macro (massimo storico nella serie disponibile — es. ATH Bitcoin)
        if not pd.isna(macro_hi) and entry < macro_hi:
            rr = (macro_hi - entry) / risk
            tp_candidates.append((macro_hi, rr, f"massimo storico serie ({fp(macro_hi)})"))

        # 5. Swing resistenze storiche (massimi intermedi già testati)
        for sr in sorted([r for r in swing_resistances if entry < r <= max_allowed*1.001], reverse=True):
            rr = (sr - entry) / risk
            tp_candidates.append((sr, rr, f"resistenza storica ({fp(sr)}) — massimo già testato"))

        # Filtra validi: R/R >= 2.0 (più realistico di 2.5)
        valid = [(p,rr,rat) for p,rr,rat in tp_candidates if rr >= 2.0]
        valid.sort(key=lambda x: x[1], reverse=True)

        if valid:
            # Confluenza alta → prende il più ambizioso tra i validi
            # Confluenza media → prende quello con RR più vicino a 3x (equilibrato)
            if confluence_score >= 25 and len(valid) >= 1:
                best = valid[0]
            elif len(valid) >= 2:
                # Cerca quello con RR tra 2.5 e 5 — credibile
                balanced = [x for x in valid if 2.5 <= x[1] <= 5.0]
                best = balanced[0] if balanced else valid[0]
            else:
                best = valid[0]
        else:
            # Nessun candidato valido → usa il primo disponibile anche sotto 2R
            if tp_candidates:
                tp_candidates.sort(key=lambda x: x[1], reverse=True)
                best = tp_candidates[0]
            else:
                # Fallback assoluto: 2.5R ma SOLO se non supera il massimo storico
                tp_fb = entry + 2.5 * risk
                if not pd.isna(macro_hi): tp_fb = min(tp_fb, macro_hi)
                rr_fb = (tp_fb - entry) / risk
                best = (tp_fb, rr_fb, f"livello 2.5R ({fp(tp_fb)}) — struttura insufficiente")

    else:  # SHORT — simmetrico, mai sotto il minimo storico
        min_allowed = macro_lo if not pd.isna(macro_lo) else 0.0

        # 1. Fib 0.236 macro (lato ribassista)
        retr236 = structs["macro_fib"].get("retr_236", np.nan)
        if not pd.isna(retr236) and min_allowed <= retr236 < entry:
            rr = (entry - retr236) / risk
            tp_candidates.append((retr236, rr, f"fib 0.236 macro ({fp(retr236)})"))

        # 2. Massimo struttura intermedia (short: il target è verso il basso)
        if not pd.isna(mid_lo) and min_allowed <= mid_lo < entry:
            rr = (entry - mid_lo) / risk
            tp_candidates.append((mid_lo, rr, f"minimo struttura recente ({fp(mid_lo)})"))

        # 3. Minimo macro
        if not pd.isna(macro_lo) and macro_lo < entry:
            rr = (entry - macro_lo) / risk
            tp_candidates.append((macro_lo, rr, f"minimo storico ({fp(macro_lo)})"))

        # 4. Swing supports storici
        for ss in sorted([s for s in swing_supports if min_allowed <= s < entry]):
            rr = (entry - ss) / risk
            tp_candidates.append((ss, rr, f"supporto storico ({fp(ss)})"))

        valid = [(p,rr,rat) for p,rr,rat in tp_candidates if rr >= 2.0]
        valid.sort(key=lambda x: x[1], reverse=True)
        if valid:
            best = valid[0]
        elif tp_candidates:
            tp_candidates.sort(key=lambda x: x[1], reverse=True)
            best = tp_candidates[0]
        else:
            tp_fb = entry - 2.5 * risk
            if not pd.isna(macro_lo): tp_fb = max(tp_fb, macro_lo)
            best = (tp_fb, (entry-tp_fb)/risk, f"livello 2.5R ({fp(tp_fb)})")

    return round(best[0], 6), round(best[1], 2), best[2]

# BLOCCO 1
def select_candidates(snapshots,history):
    """
    Seleziona TUTTI gli asset con segnale BUY o HIGH CONVICTION BUY dal Layer 1.
    Non applica più un limite numerico fisso — prende tutti i segnali forti.
    Applica solo deduplicazione per correlazione su asset molto simili.
    """
    print("\n[L2-B1] Candidate Selection — tutti i BUY/HC-BUY...")
    cands={}
    for tf,snap in snapshots.items():
        if snap.empty: cands[tf]=pd.DataFrame(); continue
        # Prende TUTTI gli asset con segnale BUY o HIGH CONVICTION BUY
        mask_hc = snap["Signal"]=="HIGH CONVICTION BUY"
        mask_buy = snap["Signal"]=="BUY"
        mask_base = (snap["Score"]>=MIN_SCORE)&(snap["Confidence"]>=MIN_CONFIDENCE)&(snap["Codice"].isin(history.keys()))
        top = snap[mask_base & (mask_hc | mask_buy)].copy()
        # Ordina: prima HC BUY, poi BUY, poi per score
        top["_sig_order"] = top["Signal"].map({"HIGH CONVICTION BUY":0,"BUY":1}).fillna(2)
        top = top.sort_values(["_sig_order","Score"], ascending=[True,False]).drop(columns=["_sig_order"])
        # Deduplicazione per correlazione (solo su asset molto simili > 88%)
        if len(top)>1:
            codes=top["Codice"].tolist(); keep=[]; excl=set()
            for i,ca in enumerate(codes):
                if ca in excl: continue
                keep.append(ca)
                if ca not in history: continue
                cla=history[ca]["Close"].dropna()
                for cb in codes[i+1:]:
                    if cb in excl or cb not in history: continue
                    clb=history[cb]["Close"].dropna(); idx=cla.index.intersection(clb.index)
                    if len(idx)<20: continue
                    ra=cla.loc[idx].pct_change().dropna(); rb=clb.loc[idx].pct_change().dropna()
                    idx2=ra.index.intersection(rb.index)
                    if len(idx2)<15: continue
                    corr=float(ra.loc[idx2].corr(rb.loc[idx2]))
                    if not np.isnan(corr) and corr>=MAX_CORR_EXCLUDE: excl.add(cb)
            top=top[top["Codice"].isin(keep)]
        cands[tf]=top
        n_hc=len(top[top["Signal"]=="HIGH CONVICTION BUY"])
        n_buy=len(top[top["Signal"]=="BUY"])
        print(f"  {tf}: {len(top)} candidati — {n_hc} HC-BUY, {n_buy} BUY")
    return cands

# BLOCCO 2

def classify(code,df,row):
    if len(df)<20: return []
    cl=df["Close"].dropna(); hi=df["High"].dropna() if "High" in df.columns else cl
    lo=df["Low"].dropna() if "Low" in df.columns else cl
    n=len(cl); last=float(cl.iloc[-1]); ap=lambda t,a,b:min(t,max(a,n//b))
    rv=sf(_rsi(cl,ap(14,3,3)).iloc[-1])
    bsv,biv,bmv,bwv=_bol(cl,ap(20,5,3)); bsv=sf(bsv.iloc[-1]); biv=sf(biv.iloc[-1]); bmv=sf(bmv.iloc[-1]); bwv=sf(bwv.iloc[-1])
    e20=sf(_ema(cl,ap(20,3,3)).iloc[-1]); e50=sf(_ema(cl,ap(50,5,2)).iloc[-1]) if n>=10 else np.nan
    z=sf(_zsc(cl,ap(20,5,3)).iloc[-1])
    dh2,dl2=_don(hi,lo,ap(20,5,3)); dh5,_=_don(hi,lo,ap(50,10,2))
    dh2v=sf(dh2.iloc[-1]); dl2v=sf(dl2.iloc[-1]); dh5v=sf(dh5.iloc[-1]) if n>=15 else np.nan
    adv=sf(row.get("ADX14",np.nan)); sig=str(row.get("Signal","HOLD")); tr=str(row.get("Trend","Range"))
    ca=str(row.get("Candles","")); pa=str(row.get("Patterns",""))
    ibul=tr in ["Trend Bullish","Bullish"]; iber=tr in ["Trend Bearish","Bearish"]
    irng=tr=="Range"; ilong=sig in ["HIGH CONVICTION BUY","BUY"]; ishort=sig in ["HIGH CONVICTION SELL","SELL"]
    aok=pd.isna(adv) or adv>=18
    levels=_level_pack(hi,lo,cl,ap(50,10,2))
    su,re=levels["sr_supports"],levels["sr_resistances"]
    swing_su,swing_re=levels["swing_supports"],levels["swing_resistances"]
    near_sup=levels["nearest_support"] if not pd.isna(levels["nearest_support"]) else levels["fallback_support"]
    near_res=levels["nearest_resistance"] if not pd.isna(levels["nearest_resistance"]) else levels["fallback_resistance"]
    long_break_ref=near_res if not pd.isna(near_res) else dh2v
    short_break_ref=near_sup if not pd.isna(near_sup) else dl2v
    bc=any(c in ca for c in ["Bullish Engulfing","Hammer","Morning Star","Piercing"])
    bec=any(c in ca for c in ["Bearish Engulfing","Shooting Star","Evening Star","Dark Cloud"])
    out=[]

    def add_setup(name,distance_ref,pattern_clarity,extra_conflicts=0):
        dist_pct=abs(last/distance_ref-1)*100 if (not pd.isna(distance_ref) and distance_ref) else 9.9
        conflicts=extra_conflicts
        if ilong and iber: conflicts+=1
        if ishort and ibul: conflicts+=1
        if "Inside Bar" in pa and "Breakout" in name: conflicts+=1
        if not pd.isna(rv):
            if ilong and rv>75: conflicts+=1
            if ishort and rv<25: conflicts+=1
        sqt=_setup_quality_type(dist_pct,conflicts,sf((sf(bwv)*100) if not pd.isna(bwv) else np.nan),pattern_clarity)
        out.append({"name":name,"setup_quality_type":sqt,"distance_pct":round(dist_pct,2)})

    if ilong and not pd.isna(long_break_ref) and last>=long_break_ref*0.998 and (ibul or not iber) and aok and (pd.isna(rv) or rv<78):
        add_setup("Breakout", long_break_ref, 2 if ("Breakout 20d" in pa or "Breakout 50d" in pa) else 1)
    if ishort and not pd.isna(short_break_ref) and last<=short_break_ref*1.002 and (iber or irng) and aok:
        add_setup("Breakdown", short_break_ref, 2 if ("Breakdown 20d" in pa or "Breakdown 50d" in pa) else 1)

    nem=((not pd.isna(e20) and abs(last/e20-1)<0.025) or (not pd.isna(e50) and abs(last/e50-1)<0.035))
    if ilong and ibul and nem and ((pd.isna(long_break_ref)) or last<long_break_ref*0.998):
        add_setup("Pullback su Trend", e20 if not pd.isna(e20) else e50, 2 if bc else 1)
    if ishort and iber and nem:
        add_setup("Pullback Ribassista", e20 if not pd.isna(e20) else e50, 2 if bec else 1)

    if ilong and ((not pd.isna(rv) and rv<38) or (not pd.isna(biv) and last<=biv*1.01) or (not pd.isna(z) and z<-1.5)) and not iber:
        add_setup("Mean Reversion", near_sup if not pd.isna(near_sup) else biv, 1 if bc else 0, extra_conflicts=0 if irng else 1)
    if ishort and ((not pd.isna(rv) and rv>62) or (not pd.isna(bsv) and last>=bsv*0.99) or (not pd.isna(z) and z>1.5)) and not ibul:
        add_setup("Mean Reversion Short", near_res if not pd.isna(near_res) else bsv, 1 if bec else 0, extra_conflicts=0 if irng else 1)

    if ilong and bc and (irng or iber):
        add_setup("Reversal Rialzista", near_sup if not pd.isna(near_sup) else levels["last_swing_low"], 2)
    if ishort and bec and (irng or ibul):
        add_setup("Reversal Ribassista", near_res if not pd.isna(near_res) else levels["last_swing_high"], 2)

    if (ilong or ishort) and ((not pd.isna(bwv) and bwv<0.06) or "Inside Bar" in pa or "Volatility Squeeze" in pa):
        ref = long_break_ref if ilong else short_break_ref
        add_setup("Compressione", ref, 1, extra_conflicts=1 if pd.isna(ref) else 0)

    nsu=any(abs(last/s-1)<0.015 for s in swing_su if s>0) or any(abs(last/s-1)<0.012 for s in su if s>0)
    if ilong and nsu and bc and not iber:
        add_setup("Support Bounce", near_sup if not pd.isna(near_sup) else levels["last_swing_low"], 2)
    nre=any(abs(last/r-1)<0.015 for r in swing_re if r>0) or any(abs(last/r-1)<0.012 for r in re if r>0)
    if ishort and nre and bec:
        add_setup("Resistance Rejection", near_res if not pd.isna(near_res) else levels["last_swing_high"], 2)

    # ── NUOVI ARCHETIPI — sostituiscono Generic Long/Short ──────────────────
    if not out:
        # Pullback Strutturale: trend attivo + prezzo vicino a zona chiave
        fib_vals = _fib(cl, min(90, n))
        fib_zone = any(abs(last/v-1)<0.03 for v in fib_vals.values() if v>0)
        near_ma = (not pd.isna(e20) and abs(last/e20-1)<0.035) or                   (not pd.isna(e50) and abs(last/e50-1)<0.04)
        n_confluences = sum([fib_zone, near_ma,
                             any(abs(last/s-1)<0.025 for s in swing_su if s>0),
                             bool(bc or bec)])
        if ilong and ibul and n_confluences >= 2:
            add_setup("Pullback Strutturale", e50 if not pd.isna(e50) else e20, 2)
        elif ishort and iber and n_confluences >= 2:
            add_setup("Pullback Strutturale Short", e50 if not pd.isna(e50) else e20, 2)

        # Late Trend: prezzo vicino ai massimi / minimi, spazio limitato
        elif ilong:
            hi20 = float(hi.rolling(min(20,n)).max().iloc[-1]) if len(hi)>=10 else last
            dist_hi = (hi20-last)/hi20*100 if hi20>0 else 99
            if dist_hi < 3:
                add_setup("Late Trend Long", near_res if not pd.isna(near_res) else hi20, 0, extra_conflicts=1)
            else:
                add_setup("Continuation Debole Long", e20 if not pd.isna(e20) else last, 0, extra_conflicts=1)
        elif ishort:
            lo20 = float(lo.rolling(min(20,n)).min().iloc[-1]) if len(lo)>=10 else last
            dist_lo = (last-lo20)/last*100 if last>0 else 99
            if dist_lo < 3:
                add_setup("Late Trend Short", near_sup if not pd.isna(near_sup) else lo20, 0, extra_conflicts=1)
            else:
                add_setup("Continuation Debole Short", e20 if not pd.isna(e20) else last, 0, extra_conflicts=1)

    # De-duplicazione: se ci sono setup multipli, tieni solo il più rilevante.
    # Priorità: Pullback Strutturale > Pullback su Trend > Breakout > Reversal >
    #           Mean Reversion > Support Bounce > Compressione > Continuation > Late
    if len(out) > 1:
        priority_order = [
            "Pullback Strutturale","Pullback Strutturale Short",
            "Pullback su Trend","Pullback Ribassista",
            "Breakout","Breakdown",
            "Reversal Rialzista","Reversal Ribassista",
            "Support Bounce","Resistance Rejection",
            "Mean Reversion","Mean Reversion Short",
            "Compressione",
            "Continuation Debole Long","Continuation Debole Short",
            "Late Trend Long","Late Trend Short",
        ]
        def setup_priority(s):
            name = s["name"] if isinstance(s,dict) else s
            return priority_order.index(name) if name in priority_order else 99
        out.sort(key=setup_priority)
        out = [out[0]]  # prendi solo il migliore
    return out


# BLOCCO 3
# BLOCCO 3

def plan(code,df,st,row,tf,snaps_all):
    st_name=st["name"] if isinstance(st,dict) else st
    st_qtype=st.get("setup_quality_type","average") if isinstance(st,dict) else "average"
    if len(df)<15: return None
    cl=df["Close"].dropna(); hi=df["High"].dropna() if "High" in df.columns else cl
    lo=df["Low"].dropna() if "Low" in df.columns else cl
    n=len(cl); last=float(cl.iloc[-1]); ap=lambda t,a,b:min(t,max(a,n//b))
    av=sf(_atr(hi,lo,cl,ap(14,3,4)).iloc[-1])
    if pd.isna(av) or av<=0 or last<=0: return None
    ap_=av/last*100
    bsv,biv,bmv,_=_bol(cl,ap(20,5,3)); bsv=sf(bsv.iloc[-1]); biv=sf(biv.iloc[-1]); bmv=sf(bmv.iloc[-1])
    e20=sf(_ema(cl,ap(20,3,3)).iloc[-1]); e50=sf(_ema(cl,ap(50,5,2)).iloc[-1]) if n>=10 else np.nan
    fbs=_fib(cl,min(90,n))
    levels=_level_pack(hi,lo,cl,ap(100,15,1))
    su,re=levels["supports_all"],levels["resistances_all"]
    swing_su,swing_re=levels["swing_supports"],levels["swing_resistances"]
    dh2,dl2=_don(hi,lo,ap(20,5,3)); dh5,_=_don(hi,lo,ap(50,10,2))
    dh2v=sf(dh2.iloc[-1]); dl2v=sf(dl2.iloc[-1]); dh5v=sf(dh5.iloc[-1]) if n>=15 else np.nan
    rv=sf(row.get("RSI14",np.nan))
    il="Short" not in st_name and "Rejection" not in st_name and "Breakdown" not in st_name and "Ribassista" not in st_name

    near_sup=levels["nearest_support"] if not pd.isna(levels["nearest_support"]) else levels["fallback_support"]
    near_res=levels["nearest_resistance"] if not pd.isna(levels["nearest_resistance"]) else levels["fallback_resistance"]
    last_sw_low=levels["last_swing_low"] if not pd.isna(levels["last_swing_low"]) else near_sup
    last_sw_high=levels["last_swing_high"] if not pd.isna(levels["last_swing_high"]) else near_res

    def pick_above(ref, extra=None):
        vals=[x for x in re if x>ref]
        if extra is not None and not pd.isna(extra) and extra>ref: vals.append(extra)
        vals += [v for v in fbs.values() if v>ref]
        vals=sorted(set([float(x) for x in vals if not pd.isna(x)]))
        return vals

    def pick_below(ref, extra=None):
        vals=[x for x in su if x<ref]
        if extra is not None and not pd.isna(extra) and extra<ref: vals.append(extra)
        vals += [v for v in fbs.values() if v<ref]
        vals=sorted(set([float(x) for x in vals if not pd.isna(x)]), reverse=True)
        return vals

    if st_name=="Breakout":
        rf=near_res if not pd.isna(near_res) else (dh2v if not pd.isna(dh2v) else last)
        elo=rf*1.0005; ehi=max(rf*1.004,last*1.001)
        sl_ref=last_sw_low if not pd.isna(last_sw_low) and last_sw_low<elo else (rf-0.9*av)
        sl=min(sl_ref*0.997, elo-0.8*av)
        rsk=ehi-sl
        ups=pick_above(ehi, dh5v)
        tp1=ups[0] if ups else ehi+1.5*rsk
        tp2=ups[1] if len(ups)>1 else max(tp1+0.8*rsk, ehi+2.2*rsk)
        tp3=ups[2] if len(ups)>2 else max(tp2+0.9*rsk, ehi+3.1*rsk)
        inv=f"Ritorno sotto {fp(rf)}"
    elif st_name=="Breakdown":
        rf=near_sup if not pd.isna(near_sup) else (dl2v if not pd.isna(dl2v) else last)
        elo=min(rf*0.996,last*0.999); ehi=rf*0.9995
        sl_ref=last_sw_high if not pd.isna(last_sw_high) and last_sw_high>ehi else (rf+0.9*av)
        sl=max(sl_ref*1.003, ehi+0.8*av)
        rsk=sl-elo
        dns=pick_below(elo)
        tp1=dns[0] if dns else elo-1.5*rsk
        tp2=dns[1] if len(dns)>1 else min(tp1-0.8*rsk, elo-2.2*rsk)
        tp3=dns[2] if len(dns)>2 else min(tp2-0.9*rsk, elo-3.1*rsk)
        inv=f"Ritorno sopra {fp(rf)}"
    elif st_name=="Pullback su Trend":
        base=min([x for x in [e20,e50,near_sup] if not pd.isna(x)], default=last*0.97)
        top=max([x for x in [e20,e50,near_sup] if not pd.isna(x)], default=last*0.99)
        elo=base*0.999; ehi=top*1.002
        swl=last_sw_low if not pd.isna(last_sw_low) and last_sw_low<elo else float(lo.iloc[-10:].min())
        sl=min(swl*0.996, elo-0.9*av); rsk=ehi-sl
        ups=pick_above(ehi, near_res if not pd.isna(near_res) else float(hi.iloc[-20:].max()))
        tp1=ups[0] if ups else ehi+1.5*rsk
        tp2=ups[1] if len(ups)>1 else max(tp1+0.7*rsk, ehi+2.1*rsk)
        tp3=ups[2] if len(ups)>2 else max(tp2+0.8*rsk, ehi+2.9*rsk)
        inv=f"Rottura sotto {fp(swl)}"
    elif st_name=="Pullback Ribassista":
        base=min([x for x in [e20,e50,near_res] if not pd.isna(x)], default=last*1.01)
        top=max([x for x in [e20,e50,near_res] if not pd.isna(x)], default=last*1.03)
        elo=base*0.998; ehi=top*1.001
        swh=last_sw_high if not pd.isna(last_sw_high) and last_sw_high>ehi else float(hi.iloc[-10:].max())
        sl=max(swh*1.004, ehi+0.9*av); rsk=sl-elo
        dns=pick_below(elo, near_sup if not pd.isna(near_sup) else float(lo.iloc[-20:].min()))
        tp1=dns[0] if dns else elo-1.5*rsk
        tp2=dns[1] if len(dns)>1 else min(tp1-0.7*rsk, elo-2.1*rsk)
        tp3=dns[2] if len(dns)>2 else min(tp2-0.8*rsk, elo-2.9*rsk)
        inv=f"Rottura sopra {fp(swh)}"
    elif st_name=="Mean Reversion":
        ref=near_sup if not pd.isna(near_sup) else (biv if not pd.isna(biv) else last*0.99)
        elo=ref*0.999; ehi=last*1.0015
        swl=last_sw_low if not pd.isna(last_sw_low) else float(lo.iloc[-15:].min())
        sl=swl*0.994; rsk=ehi-sl
        ups=pick_above(ehi, bmv if not pd.isna(bmv) else e20)
        tp1=ups[0] if ups else ehi+1.4*rsk
        tp2=ups[1] if len(ups)>1 else max(tp1+0.6*rsk, ehi+2.0*rsk)
        tp3=ups[2] if len(ups)>2 else max(tp2+0.8*rsk, ehi+2.7*rsk)
        inv=f"Nuovo minimo sotto {fp(swl)}"
    elif st_name=="Mean Reversion Short":
        ref=near_res if not pd.isna(near_res) else (bsv if not pd.isna(bsv) else last*1.01)
        elo=last*0.9985; ehi=ref*1.001
        swh=last_sw_high if not pd.isna(last_sw_high) else float(hi.iloc[-15:].max())
        sl=swh*1.006; rsk=sl-elo
        dns=pick_below(elo, bmv if not pd.isna(bmv) else e20)
        tp1=dns[0] if dns else elo-1.4*rsk
        tp2=dns[1] if len(dns)>1 else min(tp1-0.6*rsk, elo-2.0*rsk)
        tp3=dns[2] if len(dns)>2 else min(tp2-0.8*rsk, elo-2.7*rsk)
        inv=f"Nuovo massimo sopra {fp(swh)}"
    elif st_name=="Reversal Rialzista":
        elo=last*0.999; ehi=float(hi.iloc[-1])*1.0015 if len(hi)>=1 else last*1.004
        swl=last_sw_low if not pd.isna(last_sw_low) else float(lo.iloc[-5:].min())
        sl=swl*0.994; rsk=ehi-sl
        ups=pick_above(ehi, near_res)
        tp1=ups[0] if ups else ehi+1.6*rsk
        tp2=ups[1] if len(ups)>1 else max(tp1+0.7*rsk, ehi+2.3*rsk)
        tp3=ups[2] if len(ups)>2 else max(tp2+0.9*rsk, ehi+3.1*rsk)
        inv=f"Rottura sotto {fp(swl)}"
    elif st_name=="Reversal Ribassista":
        elo=float(lo.iloc[-1])*0.9985 if len(lo)>=1 else last*0.995; ehi=last*1.001
        swh=last_sw_high if not pd.isna(last_sw_high) else float(hi.iloc[-5:].max())
        sl=swh*1.006; rsk=sl-elo
        dns=pick_below(elo, near_sup)
        tp1=dns[0] if dns else elo-1.6*rsk
        tp2=dns[1] if len(dns)>1 else min(tp1-0.7*rsk, elo-2.3*rsk)
        tp3=dns[2] if len(dns)>2 else min(tp2-0.9*rsk, elo-3.1*rsk)
        inv=f"Rottura sopra {fp(swh)}"
    elif st_name=="Compressione":
        br=(bsv-biv) if not pd.isna(bsv) and not pd.isna(biv) else av*2
        if il:
            rf=near_res if not pd.isna(near_res) else (bsv if not pd.isna(bsv) else last)
            elo=rf*1.0005; ehi=elo*1.0025
            sl_ref=last_sw_low if not pd.isna(last_sw_low) else (biv if not pd.isna(biv) else elo-br)
            sl=min(sl_ref*0.996, elo-0.8*av); rsk=ehi-sl
            ups=pick_above(ehi)
            tp1=ups[0] if ups else ehi+br
            tp2=ups[1] if len(ups)>1 else max(tp1+0.8*br, ehi+1.7*br)
            tp3=ups[2] if len(ups)>2 else max(tp2+0.8*br, ehi+2.4*br)
            inv=f"Rottura verso il basso di {fp(sl_ref)}"
        else:
            rf=near_sup if not pd.isna(near_sup) else (biv if not pd.isna(biv) else last)
            ehi=rf*0.9995; elo=ehi*0.9975
            sl_ref=last_sw_high if not pd.isna(last_sw_high) else (bsv if not pd.isna(bsv) else ehi+br)
            sl=max(sl_ref*1.004, ehi+0.8*av); rsk=sl-elo
            dns=pick_below(elo)
            tp1=dns[0] if dns else elo-br
            tp2=dns[1] if len(dns)>1 else min(tp1-0.8*br, elo-1.7*br)
            tp3=dns[2] if len(dns)>2 else min(tp2-0.8*br, elo-2.4*br)
            inv=f"Rottura verso lalto di {fp(sl_ref)}"
    elif st_name=="Support Bounce":
        sn=near_sup if not pd.isna(near_sup) else (min(swing_su,key=lambda s:abs(s-last)) if swing_su else last*0.97)
        elo=last*0.999; ehi=float(hi.iloc[-1])*1.0015 if len(hi)>=1 else last*1.004
        sl=(last_sw_low if not pd.isna(last_sw_low) else sn)*0.994
        rsk=ehi-sl; ups=pick_above(ehi, near_res)
        tp1=ups[0] if ups else ehi+1.6*rsk
        tp2=ups[1] if len(ups)>1 else max(tp1+0.7*rsk, ehi+2.3*rsk)
        tp3=ups[2] if len(ups)>2 else max(tp2+0.9*rsk, ehi+3.1*rsk)
        inv=f"Chiusura sotto {fp(sn)}"
    elif st_name=="Resistance Rejection":
        rn=near_res if not pd.isna(near_res) else (min(swing_re,key=lambda r:abs(r-last)) if swing_re else last*1.03)
        elo=float(lo.iloc[-1])*0.9985 if len(lo)>=1 else last*0.995; ehi=last*1.001
        sl=(last_sw_high if not pd.isna(last_sw_high) else rn)*1.006
        rsk=sl-elo; dns=pick_below(elo, near_sup)
        tp1=dns[0] if dns else elo-1.6*rsk
        tp2=dns[1] if len(dns)>1 else min(tp1-0.7*rsk, elo-2.3*rsk)
        tp3=dns[2] if len(dns)>2 else min(tp2-0.9*rsk, elo-3.1*rsk)
        inv=f"Chiusura sopra {fp(rn)}"
    elif st_name in ["Pullback Strutturale","Pullback Strutturale Short"]:
        # Entry: zona con più confluenze (usa leg swing low/high come riferimento)
        leg_d = detect_trend_legs(cl, hi, lo, window=min(5,max(2,n//10)))
        leg_sl = sf(leg_d.get("leg_swing_low", np.nan))
        leg_sh = sf(leg_d.get("leg_swing_high", np.nan))
        leg_amp = sf(leg_d.get("leg_amplitude", np.nan))
        # Entry zone basata sulla gamba: 38.2-61.8% di ritracciamento
        if il:
            if not pd.isna(leg_sh) and not pd.isna(leg_sl) and leg_amp>0:
                fib382 = leg_sh - leg_amp * 0.382
                fib618 = leg_sh - leg_amp * 0.618
                elo = min(fib618, min([x for x in [e20,e50,near_sup] if not pd.isna(x)], default=fib618)) * 0.998
                ehi = max(fib382, max([x for x in [e20,near_sup] if not pd.isna(x)], default=fib382)) * 1.002
            else:
                base=min([x for x in [e20,e50,near_sup] if not pd.isna(x)], default=last*0.97)
                top_=max([x for x in [e20,e50] if not pd.isna(x)], default=last*0.99)
                elo=base*0.998; ehi=top_*1.002
            # SL sotto swing low reale della gamba (non ATR)
            swl = leg_sl if not pd.isna(leg_sl) and leg_sl < elo else (last_sw_low if not pd.isna(last_sw_low) and last_sw_low < elo else float(lo.iloc[-10:].min()))
            sl=swl*0.993; rsk=ehi-sl
            ups=pick_above(ehi, near_res if not pd.isna(near_res) else float(hi.iloc[-20:].max()))
            # TP1: primo livello strutturale sopra
            tp1=ups[0] if ups else ehi+1.8*rsk
            # TP2: estensione coerente della gamba
            if not pd.isna(leg_amp) and leg_amp>0:
                tp2=float(cl.iloc[-1])+(leg_amp*0.618) if ups and len(ups)<2 else (ups[1] if len(ups)>1 else max(tp1+0.8*rsk, ehi+2.3*rsk))
            else:
                tp2=ups[1] if len(ups)>1 else max(tp1+0.8*rsk, ehi+2.3*rsk)
            # TP3: estensione fib o swing high precedente
            if not pd.isna(leg_sh): tp3=max(leg_sh, tp2+0.5*rsk)
            else: tp3=ups[2] if len(ups)>2 else max(tp2+0.9*rsk, ehi+3.2*rsk)
            inv=f"Rottura sotto swing low gamba {fp(swl)}"
        else:
            # SHORT: logica speculare
            if not pd.isna(leg_sh) and not pd.isna(leg_sl) and leg_amp>0:
                fib382 = leg_sl + leg_amp * 0.382
                fib618 = leg_sl + leg_amp * 0.618
                elo = min([x for x in [e20,near_res] if not pd.isna(x)], default=fib382) * 0.998
                ehi = max(fib382, max([x for x in [e20,e50,near_res] if not pd.isna(x)], default=fib618)) * 1.002
            else:
                base=min([x for x in [e20,e50,near_res] if not pd.isna(x)], default=last*1.01)
                top_=max([x for x in [e20,e50,near_res] if not pd.isna(x)], default=last*1.03)
                elo=base*0.998; ehi=top_*1.001
            swh = leg_sh if not pd.isna(leg_sh) and leg_sh > ehi else (last_sw_high if not pd.isna(last_sw_high) and last_sw_high>ehi else float(hi.iloc[-10:].max()))
            sl=swh*1.007; rsk=sl-elo
            dns=pick_below(elo, near_sup)
            tp1=dns[0] if dns else elo-1.8*rsk
            tp2=dns[1] if len(dns)>1 else min(tp1-0.8*rsk, elo-2.3*rsk)
            tp3=dns[2] if len(dns)>2 else min(tp2-0.9*rsk, elo-3.2*rsk)
            inv=f"Rottura sopra swing high gamba {fp(swh)}"
    elif st_name in ["Continuation Debole Long","Continuation Debole Short"]:
        # Entry meno aggressiva, SL ATR-based, TP conservativi
        if il:
            elo=last*0.997; ehi=last*1.002; sl=elo-1.3*av; rsk=ehi-sl
            ups=pick_above(ehi, near_res)
            tp1=ups[0] if ups else ehi+1.6*rsk; tp2=ups[1] if len(ups)>1 else ehi+2.2*rsk; tp3=ehi+3.0*rsk
        else:
            elo=last*0.998; ehi=last*1.003; sl=ehi+1.3*av; rsk=sl-elo
            dns=pick_below(elo, near_sup)
            tp1=dns[0] if dns else elo-1.6*rsk; tp2=dns[1] if len(dns)>1 else elo-2.2*rsk; tp3=elo-3.0*rsk
        inv=f"SL {fp(sl)} — continuazione debole"
    elif st_name in ["Late Trend Long","Late Trend Short"]:
        # Entry vicino ai massimi/minimi — size ridotta, TP limitato
        if il:
            elo=last*0.999; ehi=last*1.002; sl=elo-1.2*av; rsk=ehi-sl
            ups=pick_above(ehi, near_res)
            tp1=ups[0] if ups else ehi+1.4*rsk; tp2=ups[1] if len(ups)>1 else ehi+1.9*rsk; tp3=ehi+2.5*rsk
        else:
            elo=last*0.998; ehi=last*1.001; sl=ehi+1.2*av; rsk=sl-elo
            dns=pick_below(elo, near_sup)
            tp1=dns[0] if dns else elo-1.4*rsk; tp2=dns[1] if len(dns)>1 else elo-1.9*rsk; tp3=elo-2.5*rsk
        inv=f"Rottura SL {fp(sl)} — late trend, rischio esaurimento"
    else:
        if il:
            elo=last*0.998; ehi=last*1.003; sl=elo-1.5*av; rsk=ehi-sl; tp1=ehi+1.8*rsk; tp2=ehi+2.5*rsk; tp3=ehi+3.5*rsk
        else:
            elo=last*0.997; ehi=last*1.002; sl=ehi+1.5*av; rsk=sl-elo; tp1=elo-1.8*rsk; tp2=elo-2.5*rsk; tp3=elo-3.5*rsk
        inv=f"Rottura SL {fp(sl)}"

    elo=min(elo,ehi); ehi=max(elo,ehi); em=(elo+ehi)/2
    if il: rm=em-sl if em>sl else av; rr1=(tp1-em)/rm if rm>0 else 0; rr2=(tp2-em)/rm if rm>0 else 0
    else: rm=sl-em if sl>em else av; rr1=(em-tp1)/rm if rm>0 else 0; rr2=(em-tp2)/rm if rm>0 else 0
    sp=abs(em-sl)/em*100 if em>0 else np.nan

    tfs={}
    for tfc,sn in snaps_all.items():
        if sn.empty: continue
        r=sn[sn["Codice"]==code]
        if not r.empty: tfs[tfc]=str(r["Signal"].iloc[0])
    bc_=sum(1 for s in tfs.values() if "BUY" in s); sc__=sum(1 for s in tfs.values() if "SELL" in s)
    if il:
        if bc_==3: ta=15; tl="Tutti i TF allineati ▲"
        elif bc_==2: ta=8; tl="2/3 TF allineati ▲"
        elif sc__>=2: ta=-10; tl="⚠ Conflitto TF ribassista"
        else: ta=2; tl="TF misti"
    else:
        if sc__==3: ta=15; tl="Tutti i TF allineati ▼"
        elif sc__==2: ta=8; tl="2/3 TF allineati ▼"
        elif bc_>=2: ta=-10; tl="⚠ Conflitto TF rialzista"
        else: ta=2; tl="TF misti"

    over_pen=compute_overextension_score(last,e20,last_sw_low,last_sw_high,near_res if il else near_sup,"LONG" if il else "SHORT")

    pts=[]
    t=str(row.get("Trend","")); pts.append(t.lower()) if t else None
    p=str(row.get("Patterns","")); pts.append(p.split(",")[0].strip().lower()) if p and p!="nan" else None
    c_=str(row.get("Candles","")); pts.append(c_.split(",")[0].strip().lower()) if c_ and c_!="nan" else None
    mq=row.get("_quadrant",""); pts.append(f"macro: {mq.lower()}") if mq else None
    mot="; ".join([x for x in pts if x][:4]) or st_name.lower()
    rc_v="Low" if ap_<1.5 and not pd.isna(sp) and sp<3 else "High" if ap_>4 or (not pd.isna(sp) and sp>7) else "Medium"
    rn_v="setup pulito e gestibile" if rc_v=="Low" else "stop ampio — ridurre size" if not pd.isna(sp) and sp>8 else "asset molto volatile" if ap_>4 else "volatilità normale"

    next_obstacle = near_res if il else near_sup
    if il:
        if (pd.isna(next_obstacle) or next_obstacle <= em):
            next_obstacle = tp1
        room_pct = ((next_obstacle-em)/em*100) if (not pd.isna(next_obstacle) and em>0) else np.nan
        struct_anchor = near_sup if not pd.isna(near_sup) else last_sw_low
    else:
        if (pd.isna(next_obstacle) or next_obstacle >= em):
            next_obstacle = tp1
        room_pct = ((em-next_obstacle)/em*100) if (not pd.isna(next_obstacle) and em>0) else np.nan
        struct_anchor = near_res if not pd.isna(near_res) else last_sw_high

    structure_distance = abs(em-struct_anchor)/em*100 if (not pd.isna(struct_anchor) and em>0) else np.nan

    room_score = 0
    if not pd.isna(room_pct):
        if room_pct >= 6.0: room_score += 8
        elif room_pct >= 4.0: room_score += 6
        elif room_pct >= 2.5: room_score += 3
        elif room_pct >= 1.2: room_score += 1
        else: room_score -= 4
    if rr1 >= 2.4: room_score += 2
    elif rr1 < 1.35: room_score -= 2

    readiness_score = 0
    if not pd.isna(sp):
        if sp <= 3.5: readiness_score += 4
        elif sp <= 5.5: readiness_score += 2
        elif sp >= 9.0: readiness_score -= 3
    if ta >= 8: readiness_score += 3
    elif ta >= 0: readiness_score += 1
    elif ta <= -10: readiness_score -= 3
    if over_pen <= -10: readiness_score -= 2

    trend_txt=str(row.get("Trend",""))
    context_score = 0
    if il and trend_txt in ["Trend Bullish","Bullish"]: context_score += 3
    elif (not il) and trend_txt in ["Trend Bearish","Bearish"]: context_score += 3
    elif trend_txt=="Range" and st_name in ["Mean Reversion","Mean Reversion Short","Support Bounce","Resistance Rejection","Compressione"]: context_score += 2
    else: context_score -= 1
    if st_qtype=="clean": context_score += 2
    elif st_qtype=="dirty": context_score -= 2

    aggression_score = 0
    if over_pen <= -10: aggression_score += 3
    elif over_pen <= -6: aggression_score += 2
    elif over_pen <= -3: aggression_score += 1
    if rr1 < 1.55: aggression_score += 1
    if rc_v=="High": aggression_score += 1

    # Nuovi score strutturali v3
    e200v = sf(_ema(cl, min(200,max(8,n-2))).iloc[-1]) if n>=10 else np.nan
    fbs_short = _fib(cl, min(90,n))
    fbs_long  = _fib(cl, min(250,n))   # fib su movimento più ampio

    # Gamba trend
    leg_data = detect_trend_legs(cl, hi, lo, window=min(5,max(2,n//10)))
    retr_score = compute_retracement_quality_score(
        leg_data.get("retracement_pct", np.nan),
        leg_data.get("leg_phase","unknown"),
        "LONG" if il else "SHORT")

    # Multi-fib confluence
    mf_score, mf_labels = compute_multi_fib_confluence(
        last, fbs_short, fbs_long, "LONG" if il else "SHORT")

    # Trendline reali
    trendlines = detect_trendlines(hi, lo, min_touches=3, lookback=min(80,n))
    tl_score, tl_labels = compute_trendline_score(trendlines, "LONG" if il else "SHORT")

    # Confluence reale (con pesi nuovi)
    confluence_sc, confluence_labels = compute_confluence_score(
        last, em, "LONG" if il else "SHORT",
        fbs_short, levels["swing_supports"], levels["swing_resistances"],
        e20, e50, e200v, bmv, biv, bsv, near_sup, near_res,
        multi_fib_score=mf_score, multi_fib_labels=mf_labels,
        tl_score=tl_score, tl_labels=tl_labels)
    confluence_label = ", ".join(confluence_labels[:4]) if confluence_labels else "nessuna"

    tp1_for_space = tp1 if not pd.isna(tp1) else em
    space_quality_sc = compute_space_quality_score(
        em, tp1_for_space, "LONG" if il else "SHORT",
        levels["resistances_all"], levels["supports_all"])

    trend_pos_sc, trend_pos_label = compute_trend_position_score(
        last, e20, e50, e200v, hi, lo, leg_data=leg_data)

    # Structure alignment
    trend_txt_for_align = str(row.get("Trend",""))
    struct_align_sc = compute_structure_alignment_score(
        "LONG" if il else "SHORT",
        trend_txt_for_align,
        leg_data.get("leg_phase","unknown"),
        leg_data.get("retracement_quality","neutro"),
        st_name, ta)

    # ════════════════════════════════════════════════════════════════════════
    #  MOTORE STRUTTURALE — logica trader umano
    #  1 entry su confluenza reale / 1 TP ambizioso su livello storico
    # ════════════════════════════════════════════════════════════════════════

    # 1. Leggi le due strutture temporali
    e200v = sf(_ema(cl, min(200, max(8, n-2))).iloc[-1]) if n >= 10 else np.nan
    structs = read_two_structures(cl, hi, lo)

    # 2. Cerca zona di confluenza reale
    trendlines_v = detect_trendlines(hi, lo, min_touches=3, lookback=min(100, n))
    zones = find_real_confluence_zone(
        last, structs, e20, e50, e200v,
        trendlines_v, levels["swing_supports"], levels["swing_resistances"],
        "LONG" if il else "SHORT", av)

    if zones:
        entry_price = zones[0][0]
        conf_score  = zones[0][1]
        conf_reasons = zones[0][2]
    else:
        # Nessuna confluenza trovata → usa midpoint classico come fallback
        entry_price  = em
        conf_score   = 0
        conf_reasons = ["nessuna confluenza strutturale — entry zona classica"]

    # Distanza entry dal prezzo attuale
    dist_entry = abs(last / entry_price - 1) * 100 if entry_price > 0 else 0

    # 3. Stop loss strutturale
    sl_struct, sl_pct_struct = compute_structural_sl(
        entry_price, zones, levels["swing_supports"], levels["swing_resistances"],
        "LONG" if il else "SHORT", av, structs)

    # SL finale — sempre sul lato corretto rispetto all'entry
    # LONG: sl < entry. SHORT: sl > entry. Garanzia doppia.
    if il:
        # Prendi il più alto tra sl_struct e sl classico (meno rischioso per LONG)
        sl_candidates_final = [x for x in [sl_struct, sl] if not pd.isna(x) and x > 0 and x < entry_price]
        sl_final = max(sl_candidates_final) if sl_candidates_final else entry_price * 0.97
        # Garanzia assoluta
        if sl_final >= entry_price:
            sl_final = entry_price * 0.97
    else:
        sl_candidates_final = [x for x in [sl_struct, sl] if not pd.isna(x) and x > 0 and x > entry_price]
        sl_final = min(sl_candidates_final) if sl_candidates_final else entry_price * 1.03
        if sl_final <= entry_price:
            sl_final = entry_price * 1.03
    sl_final = round(sl_final, 6)
    sl_pct_final = abs(entry_price - sl_final) / entry_price * 100 if entry_price > 0 else 5.0

    # 4. TP realistico su livello storico
    tp_main, rr_main, tp_rationale = compute_realistic_tp(
        entry_price, sl_final,
        "LONG" if il else "SHORT",
        structs, levels["swing_resistances"], levels["swing_supports"],
        hi, lo, conf_score)

    # Garanzia TP — deve stare dal lato giusto rispetto all'entry
    # e MAI oltre il massimo storico della serie
    macro_hi_v = structs.get("macro_hi", np.nan)
    macro_lo_v = structs.get("macro_lo", np.nan)
    if il:
        if tp_main <= entry_price:  # TP sotto entry per LONG → errore
            tp_main = entry_price + 2.5 * abs(entry_price - sl_final)
            tp_rationale = "corretto: TP forzato sopra entry (errore strutturale)"
        # Non superare il massimo storico disponibile
        if not pd.isna(macro_hi_v) and tp_main > macro_hi_v * 1.001:
            tp_main = round(macro_hi_v * 0.999, 6)
            tp_rationale = f"limitato al massimo storico ({fp(macro_hi_v)})"
    else:
        if tp_main >= entry_price:  # TP sopra entry per SHORT → errore
            tp_main = entry_price - 2.5 * abs(sl_final - entry_price)
            tp_rationale = "corretto: TP forzato sotto entry (errore strutturale)"
        if not pd.isna(macro_lo_v) and tp_main < macro_lo_v * 0.999:
            tp_main = round(macro_lo_v * 1.001, 6)
            tp_rationale = f"limitato al minimo storico ({fp(macro_lo_v)})"

    # Ricalcola RR dopo aggiustamenti
    risk_v = abs(entry_price - sl_final)
    if risk_v > 0:
        rr_main = round(abs(tp_main - entry_price) / risk_v, 2)

    tp_pct_main = abs(tp_main - entry_price) / entry_price * 100 if entry_price > 0 else 0

    # 5. Risk class
    if ap_ < 1.5 and sl_pct_final < 3:  rc_final = "Low"
    elif ap_ > 4.0 or sl_pct_final > 8: rc_final = "High"
    else:                                 rc_final = "Medium"

    # 6. Motivi tecnici
    motivo_entry = "; ".join(conf_reasons[:3]) if conf_reasons else "confluenza tecnica"

    if il:
        if len(levels["swing_supports"]) > 0:
            nearest_sl_lev = max([s for s in levels["swing_supports"] if s < sl_final*1.02], default=sl_final)
            motivo_stop = f"rottura struttura a {fp(sl_final)} — swing low/fib violato"
        else:
            motivo_stop = f"rottura struttura a {fp(sl_final)}"
    else:
        motivo_stop = f"violazione resistenza a {fp(sl_final)} — struttura short invalidata"

    mq = row.get("_quadrant", "")
    trend_lbl = str(row.get("Trend", ""))
    macro_dir = structs.get("macro_dir","")
    contesto_parts = [x for x in [trend_lbl.lower(), f"macro {macro_dir}", f"quadrante: {mq.lower()}" if mq else ""] if x]
    contesto = "; ".join(contesto_parts[:3])

    return {
        "Codice":code, "Nome":row.get("Nome",code), "Categoria":row.get("Categoria",""),
        "Timeframe":tf, "Setup":st_name, "Direzione":"LONG" if il else "SHORT",
        "Signal L1":str(row.get("Signal","")), "Score L1":sf(row.get("Score",50)),
        "Confidence L1":sf(row.get("Confidence",50)),
        "TF Align":tl, "TF Align Score":ta,
        "Prezzo":last,
        "Entry":round(entry_price, 6),
        "Entry Low":round(elo,6), "Entry High":round(ehi,6),
        "Dist Entry %":round(dist_entry, 2),
        "Stop Loss":round(sl_final, 6),
        "SL %":round(sl_pct_final, 2),
        "TP":round(tp_main, 6),
        "TP %":round(tp_pct_main, 2),
        "RR":round(rr_main, 2),
        # retrocompatibilità quality engine
        "RR1":round(rr_main, 2), "TP1":round(tp_main,6), "TP1 %":round(tp_pct_main,2),
        "RR2":round(rr_main, 2), "TP2":round(tp_main,6), "TP2 %":round(tp_pct_main,2),
        "ATR %":round(ap_, 2), "Risk Class":rc_final,
        "Motivo Entry":motivo_entry,
        "Motivo Stop":motivo_stop,
        "Motivo TP":tp_rationale,
        "Contesto":contesto,
        "Invalidazione":inv,
        "RSI":round(rv,1) if not pd.isna(rv) else np.nan,
        "Confluence Score":conf_score,
        "Macro Dir":macro_dir,
        "Macro Lo":round(structs.get("macro_lo",np.nan),4) if not pd.isna(structs.get("macro_lo",np.nan)) else np.nan,
        "Macro Hi":round(structs.get("macro_hi",np.nan),4) if not pd.isna(structs.get("macro_hi",np.nan)) else np.nan,
        "Macro Ext 1.618":round(structs["macro_fib"].get("ext_1618",np.nan),4) if not pd.isna(structs["macro_fib"].get("ext_1618",np.nan)) else np.nan,
        "setup_quality_type":st_qtype, "Overextension Score":over_pen,
        "Room Score":room_score, "Readiness Score":readiness_score,
        "Context Score":context_score, "Aggression Score":aggression_score,
    }



def evaluate_trade_context(p):
    rr1=sf(p.get("RR1",0)); oq=sf(p.get("Overextension Score",0)); ta=sf(p.get("TF Align Score",0))
    room=sf(p.get("Room Score",0)); ready=sf(p.get("Readiness Score",0)); ctx=sf(p.get("Context Score",0))
    q=sf(p.get("Quality Score",0)); st=str(p.get("Setup","")); sqt=str(p.get("setup_quality_type","average"))
    conf=sf(p.get("Confluence Score",0)); tpos=sf(p.get("Trend Position Score",0)); spq=sf(p.get("Space Quality Score",0))

    # Setup naturalmente "opportunistici" o "watchlist"
    late_trend = st in ["Late Trend Long","Late Trend Short"]
    continuation_weak = st in ["Continuation Debole Long","Continuation Debole Short"]
    structural_pullback = st in ["Pullback Strutturale","Pullback Strutturale Short"]

    retr_sc=sf(p.get("Retracement Score",0)); struct_al=sf(p.get("Structure Alignment Score",0))
    mf_sc=sf(p.get("Multi Fib Score",0)); tl_sc_v=sf(p.get("Trendline Score",0))
    leg_phase=str(p.get("Leg Phase","unknown"))

    # Setup in zona pullback ideale = automaticamente candidato Actionable
    ideal_pullback = (leg_phase in ["ritracciamento_ideale","rimbalzo_ideale"] and retr_sc>=12)
    strong_confluence = (conf>=25 or mf_sc>=15 or tl_sc_v>=16)

    # Actionable: struttura vera + convergenza alta + posizione buona
    premium = (
        q>=58 and rr1>=1.7 and ta>=0 and oq>-10 and room>=1 and ready>=0
        and struct_al>=0
        and (sqt in ["clean","average"]) and not late_trend
        and (ideal_pullback or strong_confluence or structural_pullback)
    )
    # Watchlist: setup valido ma incompleto — manca trigger o convergenza
    watchlist_ok = (
        q>=40 and rr1>=1.35 and room>=-2 and ctx>=-1 and struct_al>=-5
        and not (late_trend and rr1<1.5)
    )
    # Opportunistic: late trend, continuation debole, setup interessante ma non perfetto
    opportunistic_forced = late_trend or continuation_weak

    if premium: return "Actionable"
    if opportunistic_forced: return "Opportunistic"
    if watchlist_ok: return "Watchlist"
    return "Opportunistic"

# BLOCCO 4

def quality(p):
    """
    Score 0-100. R/R è DOMINANTE.
    Hard floor: R/R<2.5→max B, R/R<1.8→max C, R/R<1.2→D.
    """
    if p is None: return 0,"D"
    q=0.
    sig=str(p.get("Signal L1","")); sc=sf(p.get("Score L1",50))
    st=str(p.get("Setup",""))
    rr=sf(p.get("RR", sf(p.get("RR1",0))))
    sp=sf(p.get("SL %",10)); rc=p.get("Risk Class","Medium"); ta=sf(p.get("TF Align Score",2))
    oq=sf(p.get("Overextension Score",0)); sqt=str(p.get("setup_quality_type","average"))
    room_score=sf(p.get("Room Score",0)); readiness=sf(p.get("Readiness Score",0))
    context_score=sf(p.get("Context Score",0))
    conf=sf(p.get("Confluence Score",0))
    macro_dir=str(p.get("Macro Dir",""))
    dire=str(p.get("Direzione","LONG"))
    dist_entry=sf(p.get("Dist Entry %",0))

    # ── R/R DOMINANTE (0-35) ──────────────────────────────────────────────────
    if   rr>=6.0: q+=35
    elif rr>=5.0: q+=31
    elif rr>=4.0: q+=27
    elif rr>=3.0: q+=22
    elif rr>=2.5: q+=17
    elif rr>=2.0: q+=12
    elif rr>=1.7: q+=7
    elif rr>=1.2: q+=3
    else:         q+=0

    # ── Confluenza strutturale (0-20) ─────────────────────────────────────────
    # Premia chi ha trovato vera confluenza (multi-fib + trendline + swing)
    if conf>=40: q+=20
    elif conf>=25: q+=15
    elif conf>=15: q+=10
    elif conf>=8:  q+=5
    else:          q+=0

    # ── Entry su pullback (bonus se aspetta — non insegue) ───────────────────
    # dist_entry>0 significa che l'entry è sotto il prezzo (sta aspettando)
    if dist_entry >= 3: q+=8   # aspetta zona lontana — paziente
    elif dist_entry >= 1: q+=4
    elif dist_entry == 0: q+=0  # entry sul prezzo corrente — meno interessante

    # ── Segnale L1 (0-12) ────────────────────────────────────────────────────
    if "HIGH CONVICTION" in sig: q+=12
    elif "BUY" in sig or "SELL" in sig: q+=8
    else: q+=3

    # ── Tipo setup (0-10) ────────────────────────────────────────────────────
    cl_s={"Breakout":9,"Breakdown":9,"Support Bounce":10,"Resistance Rejection":10,
          "Pullback su Trend":10,"Pullback Strutturale":10,"Pullback Strutturale Short":10,
          "Reversal Rialzista":8,"Reversal Ribassista":8,
          "Mean Reversion":7,"Mean Reversion Short":7,"Compressione":8,
          "Continuation Debole Long":3,"Continuation Debole Short":3,
          "Late Trend Long":2,"Late Trend Short":2}
    q+=cl_s.get(st,5)

    # ── Score L1 (0-8) ───────────────────────────────────────────────────────
    if sc>=72: q+=8
    elif sc>=62: q+=6
    elif sc>=55: q+=4
    else: q+=2

    # ── Allineamento macro struttura (bonus) ──────────────────────────────────
    if dire=="LONG" and "up" in macro_dir: q+=4
    elif dire=="SHORT" and "down" in macro_dir: q+=4

    # ── Risk class + SL ──────────────────────────────────────────────────────
    if rc=="Low": q+=5
    elif rc=="Medium": q+=3
    if not pd.isna(sp):
        if sp>10: q-=6
        elif sp>7: q-=3
        elif sp<3: q+=3

    # ── Contestuali ──────────────────────────────────────────────────────────
    q+=max(-5,min(5,room_score))
    q+=max(-4,min(4,readiness))
    q+=max(-4,min(4,context_score))
    q+=max(-3,min(6,ta+3))
    q+=max(-8,min(0,oq))
    if sqt=="clean": q+=3
    elif sqt=="dirty": q-=4

    q=max(0,min(100,q))

    # ── HARD FLOOR ────────────────────────────────────────────────────────────
    if   rr>=2.5: g_max="A"
    elif rr>=1.8: g_max="B"
    elif rr>=1.2: g_max="C"
    else:         g_max="D"

    if   q>=80: g_raw="A"
    elif q>=63: g_raw="B"
    elif q>=45: g_raw="C"
    else:       g_raw="D"

    order={"A":0,"B":1,"C":2,"D":3}
    g=g_raw if order[g_raw]>=order[g_max] else g_max
    return round(q),g


# BLOCCO 5
# BLOCCO 5

def exclude(p,qs):
    if p is None: return True,"Dati insufficienti"
    rr1=sf(p.get("RR1",0)); sp=sf(p.get("SL %",99)); rc=p.get("Risk Class",""); ta=sf(p.get("TF Align Score",0)); st=p.get("Setup","")
    oq=sf(p.get("Overextension Score",0)); room=sf(p.get("Room Score",0)); ctx=sf(p.get("Context Score",0)); rd=sf(p.get("Readiness Score",0))
    sqt=str(p.get("setup_quality_type","average"))
    rr_v = sf(p.get("RR", rr1))  # usa RR unificato se disponibile

    # R/R minimo assoluto: sistema lavora con soldi reali
    if rr_v < 2.0: return True, f"R/R {rr_v:.1f}x insufficiente (min 2.0x)"
    if not pd.isna(sp) and sp > 12.0: return True, f"Stop troppo ampio ({sp:.1f}%) — rischio eccessivo"
    if rc=="High" and rr_v < 2.5: return True, "Alta volatilità con R/R insufficiente"
    if ta<=-18 and ctx<=0: return True, "Forte conflitto tra timeframe"
    if qs<28: return True, "Qualità setup troppo bassa"
    if oq<=-15 and room<0 and rd<0: return True, "Setup esteso senza spazio verso il target"
    if st in ["Late Trend Long","Late Trend Short"] and rr_v<2.2: return True, f"Late Trend con R/R insufficiente ({rr_v:.1f}x)"
    if st in ["Continuation Debole Long","Continuation Debole Short"] and qs<38: return True, "Continuation debole — struttura insufficiente"
    if sqt=="dirty" and qs<40: return True, "Setup sporco — non adatto per operatività automatica"
    return False, ""


# ORCHESTRATORE
# ORCHESTRATORE


def generate_all_setups(snapshots,history):
    print("\n[LAYER 2] Strategy Engine v2.0...")
    mq=""
    try:
        mp=DATA_DIR/"macro_cache.json"
        if mp.exists():
            with open(mp) as f: mq=json.load(f).get("quadrant","")
    except: pass
    cands=select_candidates(snapshots,history)
    tfr={"Daily":None,"Weekly":"W","Monthly":"ME"}; tfm={"Daily":25,"Weekly":12,"Monthly":6}
    vld={"Daily":[],"Weekly":[],"Monthly":[]}; rej={"Daily":[],"Weekly":[],"Monthly":[]}
    watch={"Daily":[],"Weekly":[],"Monthly":[]}; opp={"Daily":[],"Weekly":[],"Monthly":[]}
    for tf,cd in cands.items():
        if cd.empty: continue
        rl=tfr[tf]; ml=tfm[tf]; print(f"\n  [{tf}] {len(cd)} candidati...")
        for _,row in cd.iterrows():
            code=row["Codice"]
            if code not in history: continue
            try:
                dfd=history[code]; dft=resamp(dfd,rl) if rl else dfd.copy()
                if len(dft)<ml: continue
                rd=row.to_dict(); rd["_quadrant"]=mq
                sts=classify(code,dft,rd)
                for st in sts:
                    p=plan(code,dft,st,rd,tf,snapshots)
                    if p is None: continue
                    qs,qg=quality(p); p["Quality Score"]=qs; p["Quality"]=qg
                    p["Setup Type"]=evaluate_trade_context(p)
                    rr_v=sf(p.get("RR",sf(p.get("RR1",0))))
                    slp=p.get("SL %",0)
                    if p["Setup Type"]=="Actionable" and qg in ["A","B"] and p["Risk Class"]!="High":
                        p["Posizione"]=f"25-35% capitale · SL {slp:.1f}%"
                    elif p["Setup Type"]=="Actionable":
                        p["Posizione"]=f"15-25% capitale · SL {slp:.1f}%"
                    elif p["Setup Type"]=="Watchlist":
                        p["Posizione"]=f"10-15% — aspetta conferma entry"
                    else:
                        p["Posizione"]=f"5-10% — opportunistico"
                    ex,rs=exclude(p,qs)
                    if ex:
                        rej[tf].append({"Codice":code,"Nome":rd.get("Nome",code),"Setup":p["Setup"],"Timeframe":tf,"Motivo Esclusione":rs,"RR1":sf(p.get("RR1",0)),"Quality":qg})
                    else:
                        if p["Setup Type"]=="Actionable":
                            vld[tf].append(p)
                            print(f"    OK {code:8} [{p['Setup'][:22]:22}] {p['Setup Type'][:3]} Q:{qg} RR:{p['RR1']:.1f}x")
                        elif p["Setup Type"]=="Watchlist":
                            watch[tf].append(p)
                            print(f"    WL {code:8} [{p['Setup'][:22]:22}] Q:{qg} RR:{p['RR1']:.1f}x")
                        else:
                            opp[tf].append(p)
                            print(f"    OP {code:8} [{p['Setup'][:22]:22}] Q:{qg} RR:{p['RR1']:.1f}x")
            except Exception as e: print(f"    ERR {code}: {e}")
    res={}
    for tf in tfr:
        actionable_df=pd.DataFrame(vld[tf])
        watch_df=pd.DataFrame(watch[tf])
        opp_df=pd.DataFrame(opp[tf])
        rs=rej[tf]
        for df_ in [actionable_df, watch_df, opp_df]:
            if not df_.empty:
                rr_col="RR" if "RR" in df_.columns else "RR1"
                # R/R primario assoluto. Il più succoso sempre in cima.
                df_["_s"]=df_[rr_col]*60 + df_["Quality Score"]*0.4
        if not actionable_df.empty:
            actionable_df=actionable_df.sort_values("_s",ascending=False).drop(columns=["_s"])
        if not watch_df.empty:
            watch_df=watch_df.sort_values("_s",ascending=False).drop(columns=["_s"])
        if not opp_df.empty:
            opp_df=opp_df.sort_values("_s",ascending=False).drop(columns=["_s"])
        if not actionable_df.empty or not watch_df.empty or not opp_df.empty:
            valid_df=pd.concat([actionable_df,watch_df,opp_df], ignore_index=True, sort=False)
            res[tf]={"valid":valid_df,"rejected":pd.DataFrame(rs),"actionable_setups":actionable_df,"watchlist_setups":watch_df,"opportunistic_setups":opp_df}
            print(f"\n  OK {tf}: {len(actionable_df)} actionable, {len(watch_df)} watchlist, {len(opp_df)} opportunistic, {len(rs)} scartati")
        else:
            res[tf]={"valid":pd.DataFrame(),"rejected":pd.DataFrame(rs),"actionable_setups":pd.DataFrame(),"watchlist_setups":pd.DataFrame(),"opportunistic_setups":pd.DataFrame()}
            print(f"\n  -- {tf}: 0 actionable, 0 watchlist, 0 opportunistic, {len(rs)} scartati")
    return res


# DASHBOARD
# DASHBOARD
# DASHBOARD
def build_setup_dashboard(results):
    def db(d): return ("<span style='background:#0d3320;color:#4ade80;padding:3px 12px;border-radius:999px;font-size:11px;font-weight:700'>▲ LONG</span>" if d=="LONG" else "<span style='background:#3d0000;color:#f87171;padding:3px 12px;border-radius:999px;font-size:11px;font-weight:700'>▼ SHORT</span>")
    def qb(g,q):
        cc={"A":("#0d3320","#4ade80"),"B":("#1a1f00","#a3e635"),"C":("#1f1500","#facc15"),"D":("#2a0a0a","#f87171")}
        bg,fg=cc.get(g,("#1a1a1a","#888")); return f"<span style='background:{bg};color:{fg};padding:3px 12px;border-radius:999px;font-size:12px;font-weight:700'>Q:{g} <span style='opacity:.7;font-size:10px'>{q}</span></span>"
    def rrc(r): return "#4ade80" if not pd.isna(r) and r>=3 else ("#facc15" if not pd.isna(r) and r>=2 else "#f87171")
    def rcb(r): c={"Low":"#4ade80","Medium":"#facc15","High":"#f87171"}.get(r,"#888"); return f"<span style='color:{c};font-size:12px;font-weight:600'>{r}</span>"
    def card(r):
        il=r["Direzione"]=="LONG"; ac="#4ade80" if il else "#f87171"; bg="#091a09" if il else "#1a0909"
        tfa=r.get("TF Align",""); tfc="color:#4ade80" if "allineati" in tfa and "⚠" not in tfa else ("color:#f87171" if "⚠" in tfa else "color:#666")
        rr_v=sf(r.get("RR",r.get("RR1",0))); tp_v=r.get("TP",r.get("TP1",float("nan")))
        tp_pct_v=r.get("TP %",r.get("TP1 %",0)); entry_v=r.get("Entry",r.get("Entry Low",r["Prezzo"]))
        dist_e=r.get("Dist Entry %",0); macro_ext=r.get("Macro Ext 1.618",float("nan"))
        conf_v=int(r.get("Confluence Score",0)); macro_dir=str(r.get("Macro Dir",""))
        def ex(v):
            if pd.isna(v): return "—"
            try: return fp(float(v))
            except: return str(v)
        rr_color=rrc(rr_v)
        dist_badge=(f"<span style='background:#1a1400;color:#b8860b;padding:2px 8px;border-radius:6px;font-size:10px'>⏳ aspetta −{dist_e:.1f}%</span>" if dist_e>1 else f"<span style='background:#0a1a0a;color:#4ade80;padding:2px 8px;border-radius:6px;font-size:10px'>▶ entry ora</span>")
        macro_badge=(f"<span style='background:#111;color:#555;padding:2px 7px;border-radius:6px;font-size:10px'>macro {macro_dir}</span>" if macro_dir else "")
        return f"""<div style='background:#0c0c0c;border:1px solid #1a1a1a;border-left:4px solid {ac};border-radius:14px;padding:1.25rem 1.5rem;margin-bottom:14px'>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:14px'>
    <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap'>
      {db(r["Direzione"])} {qb(r["Quality"],int(r["Quality Score"]))}
      <span style='background:#141414;color:#777;padding:3px 10px;border-radius:999px;font-size:10px;font-weight:700'>{r.get("Setup Type","")}</span>
      <span style='font-size:20px;font-weight:700;color:#fff;font-family:DM Mono,monospace'>{r["Codice"]}</span>
      <span style='color:#555;font-size:13px'>{r["Nome"]}</span>
      <span style='background:#141414;color:#444;padding:2px 8px;border-radius:6px;font-size:10px;text-transform:uppercase'>{r["Categoria"]}</span>
      {dist_badge} {macro_badge}
    </div>
    <div style='text-align:right'>
      <div style='font-size:14px;font-weight:500;color:{ac}'>{r["Setup"]}</div>
      <div style='font-size:11px;{tfc};margin-top:2px'>{tfa}</div>
      <div style='margin-top:4px'><span style='font-size:11px;color:#333'>R/R </span><span style='color:{rr_color};font-weight:700;font-size:18px;font-family:DM Mono,monospace'>{rr_v:.1f}x</span></div>
    </div>
  </div>
  <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:12px'>
    <div style='background:#141414;border-radius:10px;padding:10px 14px'>
      <div style='font-size:9px;color:#2a2a2a;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>Prezzo</div>
      <div style='font-size:15px;color:#555;font-family:DM Mono,monospace'>{ex(r["Prezzo"])}</div>
    </div>
    <div style='background:{bg};border-radius:10px;padding:10px 14px;border:1px solid {ac}33'>
      <div style='font-size:9px;color:#555;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>Entry</div>
      <div style='font-size:15px;color:{ac};font-weight:700;font-family:DM Mono,monospace'>{ex(entry_v)}</div>
      <div style='font-size:10px;color:#444;margin-top:3px'>Confluenza: {conf_v} pt</div>
    </div>
    <div style='background:#1a0a0a;border-radius:10px;padding:10px 14px'>
      <div style='font-size:9px;color:#2a2a2a;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>Stop Loss</div>
      <div style='font-size:15px;color:#f87171;font-family:DM Mono,monospace'>{ex(r["Stop Loss"])}</div>
      <div style='font-size:10px;color:#444;margin-top:3px'>−{r["SL %"]:.1f}%</div>
    </div>
    <div style='background:#091a09;border-radius:10px;padding:10px 14px;border:1px solid {ac}22'>
      <div style='font-size:9px;color:#2a2a2a;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>Target</div>
      <div style='font-size:15px;color:{ac};font-weight:700;font-family:DM Mono,monospace'>{ex(tp_v)}</div>
      <div style='font-size:10px;color:#4ade80;margin-top:3px'>+{tp_pct_v:.1f}% · R/R {rr_v:.1f}x</div>
      {"<div style='font-size:9px;color:#333;margin-top:2px'>Ext 1.618: "+ex(macro_ext)+"</div>" if not pd.isna(macro_ext) else ""}
    </div>
  </div>
  <div style='background:#0f0f0f;border:1px solid #141414;border-radius:10px;padding:11px 14px;margin-bottom:10px'>
    <div style='font-size:9px;color:#1e1e1e;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px'>Razionale strutturale</div>
    <div style='display:flex;flex-direction:column;gap:5px;font-size:12px'>
      <div><span style='color:#4ade80;font-weight:600;font-size:10px'>📍 ENTRY</span> <span style='color:#777'>{r.get("Motivo Entry","—")}</span></div>
      <div><span style='color:#f87171;font-weight:600;font-size:10px'>🛑 STOP</span> <span style='color:#555'>{r.get("Motivo Stop","—")}</span></div>
      <div><span style='color:#a3e635;font-weight:600;font-size:10px'>🎯 TARGET</span> <span style='color:#555'>{r.get("Motivo TP","—")}</span></div>
    </div>
  </div>
  <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;font-size:11px;color:#333'>
    <span>{r.get("Contesto","")}</span>
    <div style='background:#111;border:1px solid #1a1a1a;border-radius:8px;padding:4px 12px;color:#444'>💼 {r.get("Posizione","")}</div>
  </div>
</div>"""
    def rtab(df):
        if df.empty: return ""
        rows="".join(f"<tr><td style='font-family:DM Mono,monospace;color:#555;padding:8px 12px'>{r['Codice']}</td><td style='color:#444;padding:8px 12px'>{r['Nome']}</td><td style='color:#444;padding:8px 12px'>{r['Setup']}</td><td style='color:#f87171;font-size:11px;padding:8px 12px'>{r['Motivo Esclusione']}</td><td style='padding:8px 12px'>{r.get('Quality','—')}</td><td style='color:#555;padding:8px 12px'>{r.get('RR1',0):.1f}x</td></tr>" for _,r in df.iterrows())
        return f"<div style='margin-top:2rem'><h3 style='font-size:11px;color:#2a2a2a;font-weight:400;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px'>Setup scartati</h3><div style='overflow:auto;border:1px solid #111;border-radius:10px'><table style='width:100%;border-collapse:collapse;font-size:12px;min-width:500px'><thead><tr style='border-bottom:1px solid #141414'><th style='padding:8px 12px;text-align:left;color:#2a2a2a;font-weight:400;font-size:10px'>Codice</th><th style='padding:8px 12px;text-align:left;color:#2a2a2a;font-weight:400;font-size:10px'>Nome</th><th style='padding:8px 12px;text-align:left;color:#2a2a2a;font-weight:400;font-size:10px'>Setup</th><th style='padding:8px 12px;text-align:left;color:#2a2a2a;font-weight:400;font-size:10px'>Motivo</th><th style='padding:8px 12px;text-align:left;color:#2a2a2a;font-weight:400;font-size:10px'>Q</th><th style='padding:8px 12px;text-align:left;color:#2a2a2a;font-weight:400;font-size:10px'>R/R</th></tr></thead><tbody>{rows}</tbody></table></div></div>"
    tfh={}; tot={}
    for tf,data in results.items():
        dfv=data.get("valid",pd.DataFrame()); dfr=data.get("rejected",pd.DataFrame()); tot[tf]=len(dfv)
        if dfv.empty: tfh[tf]="<p style='color:#2a2a2a;padding:2rem;text-align:center;font-family:DM Mono,monospace'>Nessun setup valido.</p>"+rtab(dfr); continue
        sec=""
        sections=[("Actionable", data.get("actionable_setups",pd.DataFrame()), "#4ade80"),
                  ("Watchlist", data.get("watchlist_setups",pd.DataFrame()), "#a3e635"),
                  ("Opportunistic", data.get("opportunistic_setups",pd.DataFrame()), "#facc15")]
        for title,grp,color in sections:
            if grp.empty: continue
            sec+=f"<h3 style='color:{color};font-size:13px;font-weight:500;margin:0 0 10px;letter-spacing:.04em;text-transform:uppercase'>{title} — {len(grp)} setup</h3>"
            for _,r in grp.iterrows(): sec+=card(r)
            sec+="<div style='margin-bottom:20px'></div>"
        sec+=rtab(dfr); tfh[tf]=sec
    ts=now_str()
    html=f"""<!DOCTYPE html><html lang='it'><head><meta charset='utf-8'><meta http-equiv='refresh' content='900'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Jennifer Setups</title><link href='https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap' rel='stylesheet'><style>*{{box-sizing:border-box;margin:0;padding:0}}body{{background:#080808;color:#e0e0e0;font-family:'DM Sans',sans-serif;padding:24px;min-height:100vh}}.hdr{{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1px solid #111;padding-bottom:20px;margin-bottom:20px;flex-wrap:wrap;gap:12px}}.h1{{font-size:22px;font-weight:300;color:#fff;letter-spacing:-.02em;margin-bottom:4px}}.sub{{font-size:11px;color:#2a2a2a;font-family:'DM Mono',monospace}}.warn{{background:#0e0b00;border:1px solid #2a2000;border-radius:10px;padding:12px 16px;font-size:12px;color:#555;margin-bottom:18px;line-height:1.7}}.leg{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px;font-size:12px;color:#444}}.tf-sw{{display:flex;gap:8px;margin-bottom:20px;align-items:center;flex-wrap:wrap}}.tfl{{font-size:10px;color:#2a2a2a;text-transform:uppercase;letter-spacing:.1em;margin-right:4px}}.tf-btn{{background:#0f0f0f;border:1px solid #1a1a1a;color:#444;padding:7px 18px;border-radius:999px;font-size:12px;cursor:pointer;font-family:'DM Sans',sans-serif;transition:all .2s;display:flex;align-items:center;gap:6px}}.tf-btn:hover{{border-color:#2a2a2a;color:#999}}.tf-btn.active{{background:#fff;color:#000;border-color:#fff;font-weight:500}}.cnt{{background:rgba(0,0,0,.25);color:inherit;padding:1px 7px;border-radius:999px;font-size:10px}}.tf-btn.active .cnt{{background:rgba(0,0,0,.15)}}.tf-panel{{display:none}}.tf-panel.active{{display:block}}.back{{display:inline-flex;align-items:center;gap:6px;background:#0f0f0f;border:1px solid #1a1a1a;color:#444;padding:7px 14px;border-radius:999px;font-size:12px;text-decoration:none}}.back:hover{{border-color:#2a2a2a;color:#999}}.footer{{margin-top:2rem;text-align:center;font-size:10px;color:#141414;font-family:DM Mono,monospace}}@media(max-width:600px){{body{{padding:12px}}}}</style></head><body>
<div class='hdr'><div><div style='font-size:10px;color:#1e1e1e;font-family:DM Mono,monospace;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px'>Jennifer — Layer 2 Strategy Engine v2</div><div class='h1'>Trade Setups Operativi</div><div class='sub'>Aggiornato {ts} · R/R min {MIN_RR}x · Score ≥{MIN_SCORE} · Conf ≥{MIN_CONFIDENCE}</div></div><a class='back' href='index.html'>← Screening L1</a></div>
<div class='warn'>⚠️ <strong style='color:#b8860b'>Disclaimer:</strong> Setup generati algoritmicamente. Non costituiscono consulenza finanziaria. Entry zone, SL e TP sono indicativi.</div>
<div class='leg'><span><span style='color:#4ade80;font-weight:700'>Q:A</span> Eccellente ≥78</span><span><span style='color:#a3e635;font-weight:700'>Q:B</span> Buono 62–77</span><span><span style='color:#facc15;font-weight:700'>Q:C</span> Accettabile 48–61</span><span><span style='color:#f87171;font-weight:700'>Q:D</span> Scartato</span></div>
<div class='tf-sw'><span class='tfl'>Timeframe:</span><button class='tf-btn active' onclick='sw("Daily",this)'>Daily <span class='cnt'>{tot.get("Daily",0)}</span></button><button class='tf-btn' onclick='sw("Weekly",this)'>Weekly <span class='cnt'>{tot.get("Weekly",0)}</span></button><button class='tf-btn' onclick='sw("Monthly",this)'>Monthly <span class='cnt'>{tot.get("Monthly",0)}</span></button></div>
<div id='tf-Daily' class='tf-panel active'>{tfh.get("Daily","")}</div>
<div id='tf-Weekly' class='tf-panel'>{tfh.get("Weekly","")}</div>
<div id='tf-Monthly' class='tf-panel'>{tfh.get("Monthly","")}</div>
<div class='footer'>Jennifer Trading System — Layer 2 Strategy Engine v2.0 · Solo analisi statistica</div>
<script>function sw(tf,btn){{document.querySelectorAll('.tf-panel').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.tf-btn').forEach(b=>b.classList.remove('active'));document.getElementById('tf-'+tf).classList.add('active');btn.classList.add('active');}}</script>
</body></html>"""
    SETUP_FILE.write_text(html,encoding="utf-8"); print(f"  Setup dashboard: {SETUP_FILE}"); return SETUP_FILE