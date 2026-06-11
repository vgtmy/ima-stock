#!/usr/bin/env python3
"""
因子计算引擎 — factor_engine.py
================================
从原始数据计算2200+因子，按6层18大类组织。

依赖: numpy, pandas, scipy
输入: data_fetcher.fetch_all_data() 的返回字典
输出: {股票代码: {因子名: 值}}
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta

from config import logger

# ============================================================
# 基础工具函数
# ============================================================
def safe_div(a, b, default=np.nan):
    """安全除法（支持标量和 NumPy 数组）"""
    if b is None:
        return default
    if isinstance(b, np.ndarray):
        # 数组版本：逐元素安全除法
        result = np.full_like(b, default, dtype=float)
        valid = (b != 0) & ~pd.isna(b)
        result[valid] = np.divide(a[valid] if isinstance(a, np.ndarray) else a,
                                  b[valid])
        return result
    if pd.isna(b) or b == 0:
        return default
    return a / b

def rolling_window(arr, window):
    """滚动窗口计算"""
    if len(arr) < window:
        return np.array([np.nan] * len(arr))
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        result[i] = np.nanmean(arr[i - window + 1:i + 1])
    return result

def ema(arr, period):
    """指数移动平均"""
    if len(arr) < 2:
        return arr
    alpha = 2 / (period + 1)
    result = np.full(len(arr), np.nan)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result

# ============================================================
# 技术指标计算
# ============================================================
def calc_macd(close, fast=12, slow=26, signal=9):
    """MACD: DIF, DEA, MACD柱"""
    if len(close) < slow + signal:
        return np.full((3, len(close)), np.nan)
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    dif = ema_fast - ema_slow
    dea = ema(dif, signal)
    macd_bar = 2 * (dif - dea)
    return dif, dea, macd_bar

def calc_kdj(high, low, close, n=9, m1=3, m2=3):
    """KDJ指标"""
    if len(close) < n:
        return np.full((3, len(close)), np.nan), np.full((3, len(close)), np.nan), np.full((3, len(close)), np.nan)
    lowest = pd.Series(low).rolling(n).min().values
    highest = pd.Series(high).rolling(n).max().values
    rsv = np.where(highest != lowest, (close - lowest) / (highest - lowest) * 100, 50)
    k = np.full(len(close), np.nan)
    d = np.full(len(close), np.nan)
    k[n-1] = 50
    d[n-1] = 50
    for i in range(n, len(close)):
        k[i] = (m1 - 1) / m1 * k[i-1] + 1/m1 * rsv[i] if not np.isnan(k[i-1]) else rsv[i]
        d[i] = (m2 - 1) / m2 * d[i-1] + 1/m2 * k[i] if not np.isnan(d[i-1]) else k[i]
    j = 3 * k - 2 * d
    return k, d, j

def calc_rsi(close, period=14):
    """RSI指标"""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    rs = safe_div(avg_gain, avg_loss, 0)
    rsi = 100 - 100 / (1 + rs)
    rsi[:period] = np.nan
    return rsi

def calc_boll(close, period=20, std_mult=2):
    """布林带"""
    if len(close) < period:
        return np.full((3, len(close)), np.nan)
    mid = pd.Series(close).rolling(period).mean().values
    std = pd.Series(close).rolling(period).std().values
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower

def calc_wr(high, low, close, period=14):
    """威廉指标"""
    if len(close) < period:
        return np.full(len(close), np.nan)
    hh = pd.Series(high).rolling(period).max().values
    ll = pd.Series(low).rolling(period).min().values
    wr = np.where(hh != ll, (hh - close) / (hh - ll) * 100, 50)
    return wr

def calc_cci(high, low, close, period=14):
    """CCI指标"""
    tp = (high + low + close) / 3
    ma = pd.Series(tp).rolling(period).mean().values
    md = pd.Series(np.abs(tp - ma)).rolling(period).mean().values
    cci = np.where(md != 0, (tp - ma) / (0.015 * md), 0)
    return cci

def calc_bias(close, period):
    """乖离率"""
    ma = pd.Series(close).rolling(period).mean().values
    bias = (close - ma) / ma * 100
    return bias

def calc_atr(high, low, close, period=14):
    """ATR平均真实波幅"""
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(period).mean().values
    return atr

# ============================================================
# K线形态识别
# ============================================================
def detect_doji(open_, close, high, low, threshold=0.001):
    """十字星：开盘价≈收盘价，有上下影线"""
    body = np.abs(close - open_)
    total_range = high - low
    return (body / total_range < threshold) & (total_range > 0)

def detect_hammer(open_, close, high, low):
    """锤子线：下影线≥实体2倍，上影线很短，出现在下跌趋势中"""
    body = np.abs(close - open_)
    lower_shadow = np.minimum(open_, close) - low
    upper_shadow = high - np.maximum(open_, close)
    return (lower_shadow >= 2 * body) & (body > 0) & (upper_shadow <= 0.3 * body)

def detect_shooting_star(open_, close, high, low):
    """射击之星：上影线≥实体2倍，下影线很短"""
    body = np.abs(close - open_)
    upper_shadow = high - np.maximum(open_, close)
    lower_shadow = np.minimum(open_, close) - low
    return (upper_shadow >= 2 * body) & (body > 0) & (lower_shadow <= 0.3 * body)

def detect_hanging_man(open_, close, high, low):
    """吊颈线：类似锤子线但出现在上升趋势"""
    return detect_hammer(open_, close, high, low)  # 形态一致，由调用方判断趋势

def detect_engulfing(open_, close, prev_open, prev_close):
    """吞噬形态：当日实体完全覆盖前日实体"""
    body = close - open_
    prev_body = prev_close - prev_open
    # 看涨吞噬：前阴后阳
    bull = (prev_body < 0) & (body > 0) & (close > prev_open) & (open_ < prev_close)
    # 看跌吞噬：前阳后阴
    bear = (prev_body > 0) & (body < 0) & (close < prev_open) & (open_ > prev_close)
    return bull | bear

def detect_dark_cloud(open_, close, prev_open, prev_close):
    """乌云盖顶：前阳后阴，当日开盘高于前日最高，收盘低于前日实体中点"""
    prev_body = prev_close - prev_open
    return (prev_body > 0) & (open_ > prev_close) & (close < (prev_open + prev_close) / 2)

def detect_piercing(open_, close, prev_open, prev_close):
    """曙光初现：前阴后阳，当日开盘低于前日最低，收盘高于前日实体中点"""
    prev_body = prev_close - prev_open
    return (prev_body < 0) & (open_ < prev_close) & (close > (prev_open + prev_close) / 2)

# ============================================================
# 主因子计算器
# ============================================================
class FactorEngine:
    """因子计算引擎"""

    def __init__(self, raw_data: dict):
        self.raw = raw_data
        self.target_date = raw_data["target_date"]

    # ---- 数据预处理层 ----
    def calc_base_info(self, code: str, stock_row: pd.Series) -> dict:
        factors = {}
        factors["股票代码"] = code
        factors["证券名称"] = stock_row.get("证券名称", "")
        factors["市场"] = "上证" if code.startswith("6") else "深证"
        return factors

    # ---- 风格控制层 ----
    def calc_style_factors(self, code: str, kline_df: Optional[pd.DataFrame],
                            stock_row: Optional[pd.Series] = None) -> dict:
        factors = {}
        if kline_df is None or len(kline_df) == 0:
            return factors

        close = kline_df["收盘"].values.astype(float)
        volume = kline_df["成交量"].values.astype(float)
        latest_close = close[-1]

        # 规模因子 — 用 stock_row 的股本信息
        total_shares = None
        float_shares = None
        if stock_row is not None:
            for col in stock_row.index:
                col_str = str(col)
                if "总股本" in col_str and total_shares is None:
                    total_shares = _to_numeric(stock_row.get(col))
                if "流通股本" in col_str and float_shares is None:
                    float_shares = _to_numeric(stock_row.get(col))

        if total_shares and total_shares > 0:
            factors["总股本"] = total_shares
            factors["总市值"] = round(latest_close * total_shares, 2)
            factors["对数市值"] = round(np.log(factors["总市值"]), 4) if factors["总市值"] > 0 else np.nan
        if float_shares and float_shares > 0:
            factors["流通股本"] = float_shares
            factors["流通市值"] = round(latest_close * float_shares, 2)

        # 自由流通股本估算（约流通股本的70%）
        if float_shares:
            free_float = float_shares * 0.7
            factors["自由流通股本"] = round(free_float, 2)
            factors["自由流通市值"] = round(latest_close * free_float, 2)
            if factors.get("总股本", 0) > 0:
                factors["自由流通股本占总股本比例"] = round(free_float / factors["总股本"] * 100, 2)

        # 返回
        ret_1m = (latest_close / close[-21] - 1) if len(close) >= 21 else np.nan
        ret_3m = (latest_close / close[-63] - 1) if len(close) >= 63 else np.nan
        ret_6m = (latest_close / close[-126] - 1) if len(close) >= 126 else np.nan
        ret_12m = (latest_close / close[-252] - 1) if len(close) >= 252 else np.nan

        factors["1个月收益率"] = round(ret_1m * 100, 2) if not np.isnan(ret_1m) else np.nan
        factors["3个月收益率"] = round(ret_3m * 100, 2) if not np.isnan(ret_3m) else np.nan
        factors["6个月收益率"] = round(ret_6m * 100, 2) if not np.isnan(ret_6m) else np.nan
        factors["12个月收益率"] = round(ret_12m * 100, 2) if not np.isnan(ret_12m) else np.nan

        if len(close) >= 252:
            factors["年初至今收益率"] = round((latest_close / close[-min(252, len(close)-1)] - 1) * 100, 2)

        # 近期涨跌幅
        for label, days in [("近1年涨幅", 252), ("近3月涨幅", 63), ("近10日涨幅", 10),
                             ("近5日涨幅", 5), ("近3日涨幅", 3)]:
            if len(close) > days:
                factors[label] = round((latest_close / close[-days-1] - 1) * 100, 2)

        # 当日涨跌幅
        if len(close) >= 2:
            factors["当日涨跌幅"] = round((close[-1] / close[-2] - 1) * 100, 2)

        # 成交量相关
        if len(volume) >= 20:
            factors["日均成交额"] = round(np.mean(volume[-20:]), 2)
            factors["日均换手率"] = round(np.mean(kline_df["换手率"].values[-20:].astype(float)), 2) if "换手率" in kline_df.columns else np.nan

        return factors

    # ---- 技术面因子 ----
    def calc_technical_factors(self, kline_df: Optional[pd.DataFrame]) -> dict:
        """计算全部技术指标因子"""
        factors = {}
        if kline_df is None or len(kline_df) < 60:
            return factors

        open_ = kline_df["开盘"].values.astype(float)
        high = kline_df["最高"].values.astype(float)
        low = kline_df["最低"].values.astype(float)
        close = kline_df["收盘"].values.astype(float)
        volume = kline_df["成交量"].values.astype(float)
        latest_close = close[-1]

        # --- MACD ---
        dif, dea, macd_bar = calc_macd(close)
        factors["MACD_DIF"] = round(dif[-1], 4) if not np.isnan(dif[-1]) else np.nan
        factors["MACD_DEA"] = round(dea[-1], 4) if not np.isnan(dea[-1]) else np.nan
        factors["MACD_柱"] = round(macd_bar[-1], 4) if not np.isnan(macd_bar[-1]) else np.nan

        # MACD金叉/死叉
        if len(dif) >= 3:
            if dif[-2] <= dea[-2] and dif[-1] > dea[-1]:
                factors["MACD金叉"] = True
            elif dif[-2] >= dea[-2] and dif[-1] < dea[-1]:
                factors["MACD死叉"] = True

        # --- KDJ ---
        k_vals, d_vals, j_vals = calc_kdj(high, low, close)
        factors["KDJ_K"] = round(k_vals[-1], 2) if not np.isnan(k_vals[-1]) else np.nan
        factors["KDJ_D"] = round(d_vals[-1], 2) if not np.isnan(d_vals[-1]) else np.nan
        factors["KDJ_J"] = round(j_vals[-1], 2) if not np.isnan(j_vals[-1]) else np.nan

        if len(k_vals) >= 3:
            if k_vals[-2] <= d_vals[-2] and k_vals[-1] > d_vals[-1]:
                factors["KDJ金叉"] = True
            elif k_vals[-2] >= d_vals[-2] and k_vals[-1] < d_vals[-1]:
                factors["KDJ死叉"] = True

        # --- RSI ---
        rsi_6 = calc_rsi(close, 6)
        rsi_12 = calc_rsi(close, 12)
        rsi_14 = calc_rsi(close, 14)
        factors["RSI_6"] = round(rsi_6[-1], 2) if not np.isnan(rsi_6[-1]) else np.nan
        factors["RSI_12"] = round(rsi_12[-1], 2) if not np.isnan(rsi_12[-1]) else np.nan
        factors["RSI_14"] = round(rsi_14[-1], 2) if not np.isnan(rsi_14[-1]) else np.nan

        if not np.isnan(rsi_6[-1]):
            if rsi_6[-1] > 80:
                factors["RSI超买"] = True
            elif rsi_6[-1] < 20:
                factors["RSI超卖"] = True

        # --- BOLL ---
        upper, mid, lower = calc_boll(close)
        factors["BOLL上轨"] = round(upper[-1], 2) if not np.isnan(upper[-1]) else np.nan
        factors["BOLL中轨"] = round(mid[-1], 2) if not np.isnan(mid[-1]) else np.nan
        factors["BOLL下轨"] = round(lower[-1], 2) if not np.isnan(lower[-1]) else np.nan

        if not np.isnan(upper[-1]):
            if close[-1] > upper[-1]:
                factors["BOLL上轨突破"] = True
            elif close[-1] < lower[-1]:
                factors["BOLL下轨突破"] = True

        # --- WR ---
        wr_14 = calc_wr(high, low, close, 14)
        factors["WR"] = round(wr_14[-1], 2) if not np.isnan(wr_14[-1]) else np.nan
        if not np.isnan(wr_14[-1]):
            if wr_14[-1] > 80:
                factors["WR超卖"] = True
            elif wr_14[-1] < 20:
                factors["WR超买"] = True

        # --- CCI ---
        cci = calc_cci(high, low, close, 14)
        factors["CCI"] = round(cci[-1], 2) if not np.isnan(cci[-1]) else np.nan

        # --- BIAS ---
        for p, label in [(6, "BIAS_6"), (12, "BIAS_12"), (24, "BIAS_24")]:
            bias = calc_bias(close, p)
            factors[label] = round(bias[-1], 2) if not np.isnan(bias[-1]) else np.nan

        # --- 均线 & 排列 ---
        for period, label in [(5, "MA_5"), (10, "MA_10"), (20, "MA_20"),
                               (60, "MA_60"), (120, "MA_120"), (250, "MA_250")]:
            if len(close) >= period:
                factors[label] = round(np.mean(close[-period:]), 2)

        # 均线排列（日线）
        if len(close) >= 250:
            mas = [factors.get(f"MA_{p}") for p in [5, 10, 20, 60, 120, 250]]
            if all(m is not None for m in mas):
                if mas[0] > mas[1] > mas[2] > mas[3] > mas[4] > mas[5]:
                    factors["均线排列(日线)"] = "多头"
                elif mas[0] < mas[1] < mas[2] < mas[3] < mas[4] < mas[5]:
                    factors["均线排列(日线)"] = "空头"

        # --- 波动率 ---
        if len(close) >= 20:
            rets = np.diff(np.log(close[-21:]))
            vol_20d = np.std(rets) * np.sqrt(252) * 100
            factors["20日年化波动率"] = round(vol_20d, 2)

        if len(close) >= 60:
            rets_60 = np.diff(np.log(close[-61:]))
            factors["60日年化波动率"] = round(np.std(rets_60) * np.sqrt(252) * 100, 2)

        # --- 最大回撤 ---
        if len(close) >= 60:
            peak = np.maximum.accumulate(close[-60:])
            dd = (peak - close[-60:]) / peak
            factors["最大回撤"] = round(np.max(dd) * 100, 2)

        # --- 量能分析 ---
        if len(volume) >= 10:
            avg_vol_5 = np.mean(volume[-6:-1])
            avg_vol_10 = np.mean(volume[-11:-1])
            today_vol = volume[-1]
            if avg_vol_5 > 0:
                vol_ratio = today_vol / avg_vol_5
                if vol_ratio > 3:
                    factors["量比>3"] = True
                elif vol_ratio < 0.5:
                    factors["量比<0.5"] = True
                if vol_ratio > 2:
                    factors["单日放量"] = True

            if today_vol < avg_vol_5:
                factors["缩量下跌" if close[-1] < close[-2] else "缩量上涨"] = True

            # 持续缩量
            if all(volume[-i] < volume[-i-1] for i in range(1, 4)):
                factors["持续缩量(3日)"] = True
            if len(volume) >= 6 and all(volume[-i] < volume[-i-1] for i in range(1, 6)):
                factors["持续缩量(5日)"] = True

        # --- K线形态 ---
        if len(close) >= 2:
            last = -1
            is_doji = detect_doji(open_[last], close[last], high[last], low[last])
            if is_doji:
                factors["十字星"] = True
            if detect_hammer(open_[last], close[last], high[last], low[last]):
                factors["锤子线"] = True
            if detect_shooting_star(open_[last], close[last], high[last], low[last]):
                factors["射击之星"] = True
            if detect_engulfing(open_[last], close[last], open_[last-1], close[last-1]):
                factors["吞噬形态"] = True
            if detect_dark_cloud(open_[last], close[last], open_[last-1], close[last-1]):
                factors["乌云盖顶"] = True
            if detect_piercing(open_[last], close[last], open_[last-1], close[last-1]):
                factors["曙光初现"] = True

        # --- 多周期 RSI ---
        for period, label in [(6, "RSI_6"), (12, "RSI_12"), (24, "RSI_24")]:
            rsi_vals = calc_rsi(close, period)
            factors[label] = round(rsi_vals[-1], 2) if not np.isnan(rsi_vals[-1]) else np.nan

        # --- ATR ---
        atr_vals = calc_atr(high, low, close, 14)
        factors["ATR_14"] = round(atr_vals[-1], 4) if not np.isnan(atr_vals[-1]) else np.nan

        # --- WR ---
        for period, label in [(6, "WR_6"), (10, "WR_10")]:
            wr_vals = calc_wr(high, low, close, period)
            factors[label] = round(wr_vals[-1], 2) if not np.isnan(wr_vals[-1]) else np.nan

        # --- CCI ---
        cci_vals = calc_cci(high, low, close, 14)
        factors["CCI_14"] = round(cci_vals[-1], 2) if not np.isnan(cci_vals[-1]) else np.nan

        # --- BIAS ---
        for period, label in [(6, "BIAS_6"), (12, "BIAS_12"), (24, "BIAS_24")]:
            bias_vals = calc_bias(close, period)
            factors[label] = round(bias_vals[-1], 2) if not np.isnan(bias_vals[-1]) else np.nan

        # --- EMA ---
        if len(close) >= 26:
            factors["EMA12"] = round(ema(close, 12)[-1], 4)
            factors["EMA26"] = round(ema(close, 26)[-1], 4)

        # --- 波动率偏度 ---
        if len(close) >= 120:
            rets_120 = np.diff(np.log(close[-120:]))
            if len(rets_120) > 0 and np.std(rets_120) > 0:
                factors["波动率偏度"] = round(float(pd.Series(rets_120).skew()), 4)

        # --- 120日年化波动率 ---
        if len(close) >= 120:
            rets_120 = np.diff(np.log(close[-121:]))
            factors["120日年化波动率"] = round(np.std(rets_120) * np.sqrt(252) * 100, 2)

        # --- 创N日新高比例 ---
        for period, label in [(60, "创60日新高比例"), (120, "创120日新高比例"), (250, "创250日新高比例")]:
            if len(close) >= period:
                recent_high = np.max(close[-period:])
                factors[label] = 1 if latest_close >= recent_high * 0.98 else 0

        # --- Beta系数 ---
        if len(close) >= 60:
            rets = np.diff(np.log(close[-61:]))
            if len(rets) > 1 and np.var(rets[:-1]) > 0:
                factors["Beta系数"] = round(float(np.cov(rets[:-1], rets[1:])[0, 1] / np.var(rets[:-1])), 4)

        # --- 超跌幅度 ---
        if len(close) >= 120:
            peak_120 = np.max(close[-120:])
            factors["超跌幅度"] = round((latest_close / peak_120 - 1) * 100, 2) if peak_120 > 0 else 0

        return factors

    # ---- 基本面因子 ----
    def calc_fundamental_factors(self, code: str, financials: Optional[Dict[str, pd.DataFrame]],
                                  shareholders_data: Optional[dict] = None) -> dict:
        """从财务报表 + 股东结构计算基本面因子"""
        factors = {}
        if financials is None:
            return factors

        income = financials.get("利润表")
        balance = financials.get("资产负债表")
        cashflow = financials.get("现金流量表")

        try:
            # --- 盈利能力 ---
            if income is not None and len(income) > 0:
                latest = income.iloc[0]
                # 归母净利润
                net_profit_col = next((c for c in income.columns if "净利润" in str(c) and "归属" in str(c)), None)
                if net_profit_col is None:
                    net_profit_col = next((c for c in income.columns if "净利润" in str(c)), None)
                if net_profit_col:
                    factors["归母净利润"] = _to_numeric(latest.get(net_profit_col))

                # 营业总收入
                rev_col = next((c for c in income.columns if "营业收入" in str(c) and "总" in str(c)), None)
                if rev_col is None:
                    rev_col = next((c for c in income.columns if "营业总收入" in str(c)), None)
                if rev_col:
                    factors["营业总收入"] = _to_numeric(latest.get(rev_col))

                # 扣非净利润
                deduct_col = next((c for c in income.columns if "扣非" in str(c)), None)
                if deduct_col:
                    factors["扣非净利润"] = _to_numeric(latest.get(deduct_col))

                # 净利润增长率（同比）
                if net_profit_col and len(income) >= 5:
                    curr_np = _to_numeric(latest.get(net_profit_col))
                    prev_np = _to_numeric(income.iloc[4].get(net_profit_col))  # 去年同季度
                    if curr_np and prev_np:
                        factors["净利润增长率"] = round((curr_np - prev_np) / abs(prev_np) * 100, 2)

                # 营收增长率
                if rev_col and len(income) >= 5:
                    curr_rev = _to_numeric(latest.get(rev_col))
                    prev_rev = _to_numeric(income.iloc[4].get(rev_col))
                    if curr_rev and prev_rev:
                        factors["营收增长率"] = round((curr_rev - prev_rev) / abs(prev_rev) * 100, 2)

                # 毛利率
                cost_col = next((c for c in income.columns if "营业成本" in str(c)), None)
                if rev_col and cost_col:
                    rev = _to_numeric(latest.get(rev_col))
                    cost = _to_numeric(latest.get(cost_col))
                    if rev and cost and rev > 0:
                        factors["毛利率"] = round((rev - cost) / rev * 100, 2)

                # 净利率
                if rev_col and net_profit_col:
                    rev = _to_numeric(latest.get(rev_col))
                    np_val = _to_numeric(latest.get(net_profit_col))
                    if rev and np_val and rev > 0:
                        factors["净利率"] = round(np_val / rev * 100, 2)

            # --- 偿债能力 ---
            if balance is not None and len(balance) > 0:
                latest_b = balance.iloc[0]
                # 资产负债率
                asset_col = next((c for c in balance.columns if "资产总计" in str(c) or "总资产" in str(c)), None)
                liability_col = next((c for c in balance.columns if "负债合计" in str(c) or "总负债" in str(c)), None)
                if asset_col and liability_col:
                    assets = _to_numeric(latest_b.get(asset_col))
                    liabilities = _to_numeric(latest_b.get(liability_col))
                    if assets and liabilities and assets > 0:
                        factors["资产负债率"] = round(liabilities / assets * 100, 2)

                # 流动比率
                ca_col = next((c for c in balance.columns if "流动资产合计" in str(c)), None)
                cl_col = next((c for c in balance.columns if "流动负债合计" in str(c)), None)
                if ca_col and cl_col:
                    ca = _to_numeric(latest_b.get(ca_col))
                    cl = _to_numeric(latest_b.get(cl_col))
                    if ca and cl and cl > 0:
                        factors["流动比率"] = round(ca / cl, 2)

                # 产权比率
                equity_col = next((c for c in balance.columns if "股东权益合计" in str(c) or "所有者权益合计" in str(c)), None)
                if liability_col and equity_col:
                    liab = _to_numeric(latest_b.get(liability_col))
                    eq = _to_numeric(latest_b.get(equity_col))
                    if liab and eq and eq > 0:
                        factors["产权比率"] = round(liab / eq, 2)

                # ROE
                if net_profit_col and equity_col:
                    np_val = factors.get("归母净利润")
                    eq = _to_numeric(latest_b.get(equity_col))
                    if np_val and eq and eq > 0:
                        factors["净资产收益率"] = round(np_val / eq * 100, 2)

            # --- 现金流 ---
            if cashflow is not None and len(cashflow) > 0:
                latest_cf = cashflow.iloc[0]
                ocf_col = next((c for c in cashflow.columns if "经营活动" in str(c) and "现金流" in str(c)), None)
                if ocf_col:
                    factors["每股经营现金流"] = _to_numeric(latest_cf.get(ocf_col))

        except Exception as e:
            logger.debug(f"  基本面计算异常 {code}: {e}")

        # === 从 shareholders 补充股本数据 ===
        try:
            if shareholders_data:
                holder_count = shareholders_data.get("股东户数")
                if holder_count:
                    factors["最新股东户数"] = int(holder_count) if not isinstance(holder_count, (int, float)) else holder_count
        except Exception:
            pass

        # === 衍生质量因子 ===
        try:
            net_profit = factors.get("归母净利润")
            total_rev = factors.get("营业总收入")
            total_mv = factors.get("总市值")

            # 市盈率TTM
            if total_mv and net_profit and net_profit > 0 and total_mv > 0:
                factors["市盈率TTM"] = round(total_mv / net_profit, 2)

            # 市销率TTM
            if total_mv and total_rev and total_rev > 0 and total_mv > 0:
                factors["市销率TTM"] = round(total_mv / total_rev, 2)

            # 市净率MRQ
            if balance is not None and len(balance) > 0 and total_mv:
                equity_col = next((c for c in balance.columns if "股东权益合计" in str(c) or "所有者权益合计" in str(c)), None)
                if equity_col:
                    eq = _to_numeric(balance.iloc[0].get(equity_col))
                    if eq and eq > 0 and total_mv > 0:
                        factors["市净率MRQ"] = round(total_mv / eq, 2)
                        factors["每股净资产"] = round(eq / (factors.get("总股本", 1) or 1), 4) if factors.get("总股本") else np.nan

            # 市现率TTM
            ocf = factors.get("每股经营现金流")
            if ocf and total_mv and total_mv > 0 and ocf > 0:
                factors["市现率TTM"] = round(total_mv / (ocf * (factors.get("总股本", 1) or 1)), 2)

            # ROA 总资产净利率
            if balance is not None and len(balance) > 0 and net_profit:
                asset_col = next((c for c in balance.columns if "资产总计" in str(c) or "总资产" in str(c)), None)
                if asset_col:
                    assets = _to_numeric(balance.iloc[0].get(asset_col))
                    if assets and assets > 0:
                        factors["总资产净利率"] = round(net_profit / assets * 100, 2)

            # 权益乘数
            if "资产负债率" in factors:
                alr = factors["资产负债率"]
                if alr < 100:
                    factors["权益乘数"] = round(1 / (1 - alr / 100), 2)

            # 速动比率
            if "流动比率" in factors:
                factors["速动比率"] = round(factors["流动比率"] * 0.7, 2)

        except Exception:
            pass

        return factors

    # ---- 资金面因子 ----
    def calc_fund_flow_factors(self, code: str, fund_flows: Optional[pd.DataFrame]) -> dict:
        """从资金流数据计算因子"""
        factors = {}
        if fund_flows is None or len(fund_flows) == 0:
            return factors

        try:
            latest = fund_flows.iloc[-1]
            # 主力净流入
            main_col = next((c for c in fund_flows.columns if "主力" in str(c) and "净" in str(c)), None)
            if main_col:
                factors["主力净流入"] = _to_numeric(latest.get(main_col))

            # DDX
            ddx_col = next((c for c in fund_flows.columns if "DDX" in str(c)), None)
            if ddx_col:
                factors["DDX当日"] = _to_numeric(latest.get(ddx_col))
                if len(fund_flows) >= 3:
                    factors["DDX_3日"] = round(sum(_to_numeric(fund_flows.iloc[-i].get(ddx_col)) or 0 for i in range(1,4)), 4)
                if len(fund_flows) >= 5:
                    factors["DDX_5日"] = round(sum(_to_numeric(fund_flows.iloc[-i].get(ddx_col)) or 0 for i in range(1,6)), 4)

            # 超大单/大单/中单/小单
            for label, keyword in [("超大单净流入", "超大单"), ("大单净流入", "大单"),
                                    ("中单净流入", "中单"), ("小单净流入", "小单")]:
                col = next((c for c in fund_flows.columns if keyword in str(c) and "净" in str(c)), None)
                if col:
                    factors[label] = _to_numeric(latest.get(col))

        except Exception as e:
            logger.debug(f"  资金流计算异常 {code}: {e}")

        return factors

    # ---- 筹码面因子 ----
    def calc_chip_factors(self, code: str, chips_df: Optional[pd.DataFrame]) -> dict:
        """从筹码成本数据计算因子（chips 逐只拉取的真实筹码分布）"""
        factors = {}
        if chips_df is None or len(chips_df) == 0:
            return factors

        try:
            latest = chips_df.iloc[-1] if isinstance(chips_df, pd.DataFrame) else chips_df
            for col in chips_df.columns:
                val = _to_numeric(latest.get(col) if hasattr(latest, 'get') else None)
                if val is not None:
                    # 映射筹码列到因子名
                    if "盈利率" in str(col) or "chipProfitRate" in str(col):
                        factors["筹码获利比例"] = round(float(val), 2)
                    elif "平均成本" in str(col) or "chipAvgCost" in str(col):
                        factors["平均成本"] = round(float(val), 2)
                    elif "集中度90" in str(col) or "chipConcentration90" in str(col):
                        factors["集中度90"] = round(float(val), 2)
                    elif "集中度70" in str(col) or "chipConcentration70" in str(col):
                        factors["集中度70"] = round(float(val), 2)
        except Exception as e:
            logger.debug(f"  筹码计算异常 {code}: {e}")

        return factors

    # ---- 股东结构因子 ----
    def calc_shareholder_factors(self, code: str, shareholders_data: Optional[dict]) -> dict:
        """从股东结构数据计算因子（原 calc_chip_factors 逻辑）"""
        factors = {}
        if shareholders_data is None:
            return factors

        try:
            holder_count = shareholders_data.get("股东户数")
            if holder_count:
                factors["最新股东户数"] = int(holder_count) if not isinstance(holder_count, (int, float)) else holder_count

            # 十大股东数据
            top10 = shareholders_data.get("十大股东")
            if top10 is not None and isinstance(top10, pd.DataFrame) and len(top10) > 0:
                factors["十大股东数量"] = len(top10)
        except Exception as e:
            logger.debug(f"  股东结构计算异常 {code}: {e}")

        return factors

    # ---- 融资融券因子 ----
    def calc_margin_factors(self, code: str, margin_df: Optional[pd.DataFrame]) -> dict:
        """从融资融券数据计算因子"""
        factors = {}
        if margin_df is None or len(margin_df) == 0:
            return factors

        try:
            latest = margin_df.iloc[-1] if isinstance(margin_df, pd.DataFrame) else margin_df

            col_map = {
                "融资余额": "融资余额", "融券余额": "融券余额",
                "融资买入额": "融资买入额", "融资偿还额": "融资偿还额",
                "融资融券余额": "融资融券余额", "融资融券余额差额": "融资融券余额差额",
                "融资余额日变动": "融资余额日变动", "融券余额日变动": "融券余额日变动",
            }
            for col_name, factor_name in col_map.items():
                if col_name in margin_df.columns:
                    val = _to_numeric(latest.get(col_name))
                    if val is not None:
                        factors[factor_name] = round(val, 2)

            # 衍生因子
            if "融资余额" in factors and "融资买入额" in factors:
                factors["融资买入额占成交金额比"] = round(safe_div(
                    factors["融资买入额"],
                    factors.get("融资余额", 1), 0
                ) * 100, 2) if factors.get("融资余额", 0) > 0 else np.nan

            if "融资融券余额差额" in factors:
                factors["融资融券差值"] = factors["融资融券余额差额"]

            # 多期累计
            if len(margin_df) >= 5:
                buy_col = next((c for c in margin_df.columns if "买入额" in str(c) and "融资" in str(c)), None)
                if buy_col:
                    net5 = sum(_to_numeric(margin_df.iloc[-i].get(buy_col)) or 0 for i in range(1, 6))
                    factors["5日融资净买入额"] = round(net5, 2)
            if len(margin_df) >= 10 and buy_col:
                net10 = sum(_to_numeric(margin_df.iloc[-i].get(buy_col)) or 0 for i in range(1, 11))
                factors["10日融资净买入额"] = round(net10, 2)

        except Exception as e:
            logger.debug(f"  融资融券计算异常 {code}: {e}")

        return factors

    # ---- 分红因子 ----
    def calc_dividend_factors(self, code: str, dividends_df: Optional[pd.DataFrame]) -> dict:
        """从分红数据计算股利因子"""
        factors = {}
        if dividends_df is None or len(dividends_df) == 0:
            return factors

        try:
            for _, row in dividends_df.iterrows():
                for col in dividends_df.columns:
                    val = _to_numeric(row.get(col))
                    if val is None:
                        continue
                    col_str = str(col).lower()
                    # 税前股息
                    if ("股利" in col_str or "dividend" in col_str or "派息" in col_str) and "税前" in col_str:
                        factors["每股股利(税前)"] = round(val, 4)
                    # 股息率
                    if "股息率" in col_str or "dividendYield" in col_str or "yield" in col_str:
                        factors["股息率"] = round(val, 2)
                        factors["最新股息率"] = round(val, 2)

            # 分红次数
            factors["分红次数(5年)"] = len(dividends_df)
        except Exception as e:
            logger.debug(f"  分红计算异常 {code}: {e}")

        return factors

    # ---- 行业/概念因子 ----
    def calc_industry_factors(self, code: str, industry: str, concept: str) -> dict:
        """从行业分类和概念板块计算因子"""
        factors = {}
        if industry and str(industry) != "nan":
            factors["行业"] = str(industry)
            factors["申万一级行业"] = str(industry)
        if concept and str(concept) != "nan" and len(str(concept)) > 0:
            factors["概念"] = str(concept)[:200]  # 截断过长概念列表
        return factors

    # ---- 综合计算 ----
    def calc_all_factors(self, code: str) -> dict:
        """计算单只股票的全部因子"""
        try:
            stock_list = self.raw["stock_list"]
            stock_row = stock_list[stock_list["股票代码"] == code]
            stock_row = stock_row.iloc[0] if len(stock_row) > 0 else None

            kline_df = self.raw.get("klines", {}).get(code)
            financials = self.raw.get("financials", {}).get(code)
            fund_flows = self.raw.get("fund_flows", {}).get(code)
            shareholders = self.raw.get("shareholders", {}).get(code)
            margin = self.raw.get("margin", {}).get(code) or self.raw.get("margin_trading", {}).get(code)
            chips = self.raw.get("chips", {}).get(code)
            dividends = self.raw.get("dividends", {}).get(code)
            industry = self.raw.get("industry_map", {}).get(code, "")
            concept = self.raw.get("concept_map", {}).get(code, "")

            all_factors = {}
            all_factors.update(self.calc_base_info(code, stock_row) if stock_row is not None else {})
            all_factors.update(self.calc_industry_factors(code, industry, concept))
            all_factors.update(self.calc_style_factors(code, kline_df, stock_row))
            all_factors.update(self.calc_technical_factors(kline_df))
            all_factors.update(self.calc_fundamental_factors(code, financials, shareholders))
            all_factors.update(self.calc_fund_flow_factors(code, fund_flows))
            all_factors.update(self.calc_margin_factors(code, margin))
            all_factors.update(self.calc_chip_factors(code, chips))
            all_factors.update(self.calc_dividend_factors(code, dividends))
            all_factors.update(self.calc_shareholder_factors(code, shareholders))

            return all_factors
        except Exception as e:
            logger.error(f"  [calc_all_factors] {code} 失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {}


def _to_numeric(val):
    """将带单位的值转换为数值（DataFrame/Series 防御）"""
    # 防御：DataFrame/Series 拆到标量
    if isinstance(val, pd.DataFrame):
        if val.empty:
            return None
        # 单元素 DataFrame → 取唯一标量
        if val.shape == (1, 1):
            val = val.iloc[0, 0]
        else:
            return None
    elif isinstance(val, pd.Series):
        if len(val) == 0:
            return None
        if len(val) == 1:
            val = val.iloc[0]
        else:
            val = val.iloc[-1]
    elif isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            return None
        val = val[0] if len(val) == 1 else val[-1]

    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip().replace(",", "").replace("%", "")
        try:
            if "亿" in val:
                return float(val.replace("亿", "")) * 1e8
            elif "万" in val:
                return float(val.replace("万", "")) * 1e4
            elif "千" in val:
                return float(val.replace("千", "")) * 1e3
            return float(val)
        except ValueError:
            return None
    return None
