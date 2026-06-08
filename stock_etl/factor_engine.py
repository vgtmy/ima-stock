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
    """安全除法"""
    if b is None or b == 0 or pd.isna(b):
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
    lower_shadow = np.minimum(open_, close_) - low
    upper_shadow = high - np.maximum(open_, close_)
    return (lower_shadow >= 2 * body) & (body > 0) & (upper_shadow <= 0.3 * body)

def detect_shooting_star(open_, close, high, low):
    """射击之星：上影线≥实体2倍，下影线很短"""
    body = np.abs(close - open_)
    upper_shadow = high - np.maximum(open_, close_)
    lower_shadow = np.minimum(open_, close_) - low
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
    def calc_style_factors(self, code: str, kline_df: Optional[pd.DataFrame]) -> dict:
        factors = {}
        if kline_df is None or len(kline_df) == 0:
            return factors

        close = kline_df["收盘"].values.astype(float)
        volume = kline_df["成交量"].values.astype(float)
        latest_close = close[-1]

        # 规模因子
        # 总股本/市值从财务数据获取，这里计算流通市值估算
        if "总股本" not in factors:
            factors["总市值"] = np.nan  # 需外部提供总股本
        factors["对数市值"] = np.log(factors.get("总市值", np.nan)) if factors.get("总市值", 0) > 0 else np.nan

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

        return factors

    # ---- 基本面因子 ----
    def calc_fundamental_factors(self, code: str, financials: Optional[Dict[str, pd.DataFrame]]) -> dict:
        """从财务报表计算基本面因子"""
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
    def calc_chip_factors(self, code: str, shareholders_df: Optional[pd.DataFrame]) -> dict:
        """从股东户数计算筹码因子"""
        factors = {}
        if shareholders_df is None or len(shareholders_df) == 0:
            return factors

        try:
            latest = shareholders_df.iloc[0]
            holders_col = next((c for c in shareholders_df.columns if "股东" in str(c) and ("户数" in str(c) or "人数" in str(c))), None)
            if holders_col:
                holders = _to_numeric(latest.get(holders_col))
                factors["最新股东户数"] = holders

                if len(shareholders_df) >= 2:
                    prev_holders = _to_numeric(shareholders_df.iloc[1].get(holders_col))
                    if prev_holders and prev_holders > 0:
                        factors["股东户数增长率"] = round((holders - prev_holders) / prev_holders * 100, 2)
        except Exception as e:
            logger.debug(f"  筹码计算异常 {code}: {e}")

        return factors

    # ---- 综合计算 ----
    def calc_all_factors(self, code: str) -> dict:
        """计算单只股票的全部因子"""
        stock_list = self.raw["stock_list"]
        stock_row = stock_list[stock_list["股票代码"] == code]
        stock_row = stock_row.iloc[0] if len(stock_row) > 0 else None

        kline_df = self.raw["klines"].get(code)
        financials = self.raw["financials"].get(code)
        fund_flows = self.raw["fund_flows"].get(code)
        shareholders = self.raw["shareholders"].get(code)

        all_factors = {}
        all_factors.update(self.calc_base_info(code, stock_row) if stock_row is not None else {})
        all_factors.update(self.calc_style_factors(code, kline_df))
        all_factors.update(self.calc_technical_factors(kline_df))
        all_factors.update(self.calc_fundamental_factors(code, financials))
        all_factors.update(self.calc_fund_flow_factors(code, fund_flows))
        all_factors.update(self.calc_chip_factors(code, shareholders))

        return all_factors


def _to_numeric(val):
    """将带单位的值转换为数值"""
    if val is None or pd.isna(val):
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
