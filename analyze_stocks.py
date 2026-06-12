#!/usr/bin/env python3
import pickle
import numpy as np

with open('data/checkpoint.pkl', 'rb') as f:
    data = pickle.load(f)

klines = data['klines']

stocks = {
    '600900': '长江电力',
    '600011': '华能国际',
    '601398': '工商银行',
    '600938': '中国海油',
    '600406': '国电南瑞',
}

def calc_macd(close, fast=12, slow=26, signal=9):
    alpha_fast = 2/(fast+1)
    alpha_slow = 2/(slow+1)
    alpha_sig = 2/(signal+1)
    ema_fast = [close[0]]
    ema_slow = [close[0]]
    for i in range(1, len(close)):
        ema_fast.append(alpha_fast * close[i] + (1-alpha_fast) * ema_fast[-1])
        ema_slow.append(alpha_slow * close[i] + (1-alpha_slow) * ema_slow[-1])
    dif = [ema_fast[i] - ema_slow[i] for i in range(len(close))]
    dea = [dif[0]]
    for i in range(1, len(dif)):
        dea.append(alpha_sig * dif[i] + (1-alpha_sig) * dea[-1])
    macd_bar = [2*(dif[i]-dea[i]) for i in range(len(dif))]
    return dif, dea, macd_bar

def calc_kdj(high, low, close, n=9, m1=3, m2=3):
    if len(close) < n:
        return [np.nan]*len(close), [np.nan]*len(close), [np.nan]*len(close)
    lowest = [min(low[max(0,i-n+1):i+1]) for i in range(len(low))]
    highest = [max(high[max(0,i-n+1):i+1]) for i in range(len(high))]
    rsv = []
    for i in range(len(close)):
        if highest[i] != lowest[i]:
            rsv.append((close[i]-lowest[i])/(highest[i]-lowest[i])*100)
        else:
            rsv.append(50)
    k = [50.0]
    d = [50.0]
    for i in range(1, len(rsv)):
        k.append((m1-1)/m1*k[-1] + 1/m1*rsv[i])
        d.append((m2-1)/m2*d[-1] + 1/m2*k[i])
    j = [3*k[i]-2*d[i] for i in range(len(k))]
    return k, d, j

def calc_rsi(close, period=14):
    if len(close) < period+1:
        return [np.nan]*len(close)
    delta = [close[i]-close[i-1] for i in range(1, len(close))]
    gains = [d if d>0 else 0 for d in delta]
    losses = [-d if d<0 else 0 for d in delta]
    avg_gain = sum(gains[:period])/period
    avg_loss = sum(losses[:period])/period
    rsi = [np.nan]*period
    for i in range(period, len(close)):
        avg_gain = (avg_gain*(period-1) + gains[i-1])/period
        avg_loss = (avg_loss*(period-1) + losses[i-1])/period
        rs = avg_gain/avg_loss if avg_loss != 0 else 100
        rsi.append(100 - 100/(1+rs))
    return rsi

def calc_boll(close, period=20, std_mult=2):
    if len(close) < period:
        return [np.nan]*len(close), [np.nan]*len(close), [np.nan]*len(close)
    upper = []
    mid = []
    lower = []
    for i in range(period-1, len(close)):
        segment = close[i-period+1:i+1]
        mid.append(sum(segment)/period)
        std = np.std(segment)
        upper.append(mid[-1] + std_mult*std)
        lower.append(mid[-1] - std_mult*std)
    return upper, mid, lower

results = {}
import pandas as pd

for code, name in stocks.items():
    if code not in klines:
        print(f'{name}({code}): 无K线数据')
        continue
    df = klines[code]
    # 列顺序: 日期,开盘,最高,收盘,最低,成交量,成交额,换手率,涨跌额,涨跌幅,振幅,股票代码
    close_col = df.columns[3]
    open_col = df.columns[1]
    high_col = df.columns[2]
    low_col = df.columns[4]
    vol_col = df.columns[5]

    close = pd.to_numeric(df[close_col], errors='coerce').dropna().values
    open_ = pd.to_numeric(df[open_col], errors='coerce').dropna().values
    high = pd.to_numeric(df[high_col], errors='coerce').dropna().values
    low = pd.to_numeric(df[low_col], errors='coerce').dropna().values
    vol = pd.to_numeric(df[vol_col], errors='coerce').dropna().values

    if len(close) < 60:
        print(f'{name}({code}): K线数据不足({len(close)}天)')
        continue

    # 今日数据
    c_today = close[-1]
    c_yest = close[-2]
    chg_pct = (c_today/c_yest - 1) * 100

    # 近期涨幅
    ret_5d = (close[-1]/close[-6]-1)*100 if len(close)>=6 else np.nan
    ret_10d = (close[-1]/close[-11]-1)*100 if len(close)>=11 else np.nan
    ret_20d = (close[-1]/close[-21]-1)*100 if len(close)>=21 else np.nan

    # MACD
    dif, dea, macd_bar = calc_macd(close)
    if dif[-2] <= dea[-2] and dif[-1] > dea[-1]:
        macd_signal = '金叉'
    elif dif[-2] >= dea[-2] and dif[-1] < dea[-1]:
        macd_signal = '死叉'
    else:
        macd_signal = '中性'

    # KDJ
    k_vals, d_vals, j_vals = calc_kdj(high, low, close)
    k_today = k_vals[-1]
    d_today = d_vals[-1]
    j_today = j_vals[-1]
    if len(k_vals) >=3 and k_vals[-2] <= d_vals[-2] and k_vals[-1] > d_vals[-1]:
        kdj_signal = '金叉'
    elif len(k_vals) >= 3 and k_vals[-2] >= d_vals[-2] and k_vals[-1] < d_vals[-1]:
        kdj_signal = '死叉'
    else:
        kdj_signal = '中性'
    kdj_status = '超买' if j_today > 80 else ('超卖' if j_today < 20 else '中性')

    # RSI
    rsi_vals = calc_rsi(close, 14)
    rsi_today = rsi_vals[-1] if not np.isnan(rsi_vals[-1]) else 50
    rsi_status = '超买' if rsi_today > 70 else ('超卖' if rsi_today < 30 else '中性')

    # BOLL
    boll_upper, boll_mid, boll_lower = calc_boll(close)
    boll_today_upper = boll_upper[-1] if not np.isnan(boll_upper[-1]) else c_today * 1.05
    boll_today_lower = boll_lower[-1] if not np.isnan(boll_lower[-1]) else c_today * 0.95
    if c_today > boll_today_upper:
        boll_status = '突破上轨'
    elif c_today < boll_today_lower:
        boll_status = '跌破下轨'
    else:
        boll_status = '中性'

    # 量能
    vol_avg5 = np.mean(vol[-6:-1]) if len(vol) >= 6 else np.mean(vol)
    vol_ratio = vol[-1]/vol_avg5 if vol_avg5 > 0 else 1
    if vol_ratio > 1.5:
        vol_status = '放量'
    elif vol_ratio < 0.7:
        vol_status = '缩量'
    else:
        vol_status = '正常'

    # 均线多头
    ma5 = np.mean(close[-5:]) if len(close) >= 5 else c_today
    ma10 = np.mean(close[-10:]) if len(close) >= 10 else c_today
    ma20 = np.mean(close[-20:]) if len(close) >= 20 else c_today
    if ma5 > ma10 > ma20:
        ma排列 = '多头'
    elif ma5 < ma10 < ma20:
        ma排列 = '空头'
    else:
        ma排列 = '混乱'

    results[code] = {
        'name': name,
        'price': c_today,
        'chg_pct': chg_pct,
        'ret_5d': ret_5d,
        'ret_10d': ret_10d,
        'ret_20d': ret_20d,
        'macd_signal': macd_signal,
        'kdj_signal': kdj_signal,
        'kdj_status': kdj_status,
        'j_value': j_today,
        'rsi_today': rsi_today,
        'rsi_status': rsi_status,
        'boll_status': boll_status,
        'vol_ratio': vol_ratio,
        'vol_status': vol_status,
        'ma排列': ma排列,
    }

    print(f'{name}({code})')
    print(f'  价格={c_today:.2f}, 涨跌幅={chg_pct:+.2f}%')
    print(f'  5日涨幅={ret_5d:+.2f}%, 10日涨幅={ret_10d:+.2f}%, 20日涨幅={ret_20d:+.2f}%')
    print(f'  MACD={macd_signal}, KDJ={kdj_signal}(J={j_today:.1f},{kdj_status})')
    print(f'  RSI={rsi_today:.1f}({rsi_status}), BOLL={boll_status}')
    print(f'  量比={vol_ratio:.2f}({vol_status}), 均线={ma排列}')
    print()

print('=' * 60)
print('=== 综合评分与建议 ===')
print('=' * 60)

for code, r in results.items():
    score = 0
    notes = []

    # RSI：40-65最优区间
    if 35 <= r['rsi_today'] <= 65:
        score += 20
        notes.append('RSI适中')
    elif r['rsi_today'] < 30:
        score += 15
        notes.append('RSI超卖')
    elif r['rsi_today'] > 80:
        score -= 15
        notes.append('RSI超买')

    # MACD
    if r['macd_signal'] == '金叉':
        score += 20
        notes.append('MACD金叉')
    elif r['macd_signal'] == '死叉':
        score -= 15
        notes.append('MACD死叉')

    # KDJ
    if r['kdj_signal'] == '金叉' and r['j_value'] < 60:
        score += 20
        notes.append('KDJ低位金叉')
    elif r['kdj_signal'] == '金叉' and r['j_value'] >= 60:
        score += 10
        notes.append('KDJ高位金叉')
    elif r['kdj_signal'] == '死叉':
        score -= 10
        notes.append('KDJ死叉')

    # 量能
    if r['vol_ratio'] > 1.5:
        score += 15
        notes.append('放量')
    elif r['vol_ratio'] > 1.1:
        score += 8
        notes.append('轻微放量')

    # 均线多头
    if r['ma排列'] == '多头':
        score += 15
        notes.append('均线多头')
    elif r['ma排列'] == '空头':
        score -= 15
        notes.append('均线空头')

    # 近期涨幅：不是追高阶段
    if r['ret_10d'] is not None and -5 <= r['ret_10d'] <= 10:
        score += 10
        notes.append('涨幅适中')
    elif r['ret_10d'] is not None and r['ret_10d'] > 15:
        score -= 10
        notes.append('短期涨幅过大')

    r['score'] = score
    r['notes'] = notes
    print(f'{r["name"]}({code}): 评分={score}  ({", ".join(notes)})')

print()
best = max(results.items(), key=lambda x: x[1]['score'])
print(f'推荐买入: {best[1]["name"]}({best[0]})  综合评分: {best[1]["score"]}')
print(f'核心理由: {", ".join(best[1]["notes"])}')