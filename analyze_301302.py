#!/usr/bin/env python3
import pickle, numpy as np, pandas as pd

with open('data/checkpoint.pkl', 'rb') as f:
    data = pickle.load(f)
klines = data['klines']

code = '301302'
df = klines[code]
cols = df.columns
close = pd.to_numeric(df[cols[3]], errors='coerce').dropna().values
open_ = pd.to_numeric(df[cols[1]], errors='coerce').dropna().values
high = pd.to_numeric(df[cols[2]], errors='coerce').dropna().values
low = pd.to_numeric(df[cols[4]], errors='coerce').dropna().values
vol = pd.to_numeric(df[cols[5]], errors='coerce').dropna().values

print(f'华如科技({code})')
print(f'数据天数: {len(close)}')
print(f'最新价格: {close[-1]:.2f}')
print(f'上一日价格: {close[-2]:.2f}')
print(f'涨跌幅: {(close[-1]/close[-2]-1)*100:+.2f}%')

ret5 = (close[-1]/close[-6]-1)*100 if len(close)>=6 else np.nan
ret10 = (close[-1]/close[-11]-1)*100 if len(close)>=11 else np.nan
ret20 = (close[-1]/close[-21]-1)*100 if len(close)>=21 else np.nan
print(f'5日涨幅: {ret5:+.2f}%')
print(f'10日涨幅: {ret10:+.2f}%')
print(f'20日涨幅: {ret20:+.2f}%')

def calc_macd(close, fast=12, slow=26, signal=9):
    alpha_fast = 2/(fast+1); alpha_slow = 2/(slow+1); alpha_sig = 2/(signal+1)
    ema_fast = [close[0]]; ema_slow = [close[0]]
    for i in range(1, len(close)):
        ema_fast.append(alpha_fast*close[i]+(1-alpha_fast)*ema_fast[-1])
        ema_slow.append(alpha_slow*close[i]+(1-alpha_slow)*ema_slow[-1])
    dif = [ema_fast[i]-ema_slow[i] for i in range(len(close))]
    dea = [dif[0]]
    for i in range(1, len(dif)):
        dea.append(alpha_sig*dif[i]+(1-alpha_sig)*dea[-1])
    return dif, dea

def calc_kdj(high, low, close, n=9, m1=3, m2=3):
    if len(close)<n: return [np.nan]*len(close),[np.nan]*len(close),[np.nan]*len(close)
    lowest = [min(low[max(0,i-n+1):i+1]) for i in range(len(low))]
    highest = [max(high[max(0,i-n+1):i+1]) for i in range(len(high))]
    rsv = [(close[i]-lowest[i])/(highest[i]-lowest[i])*100 if highest[i]!=lowest[i] else 50 for i in range(len(close))]
    k=[50.0]; d=[50.0]
    for i in range(1,len(rsv)):
        k.append((m1-1)/m1*k[-1]+1/m1*rsv[i]); d.append((m2-1)/m2*d[-1]+1/m2*k[i])
    j=[3*k[i]-2*d[i] for i in range(len(k))]
    return k, d, j

def calc_rsi(close, period=14):
    if len(close)<period+1: return [np.nan]*len(close)
    delta=[close[i]-close[i-1] for i in range(1,len(close))]
    gains=[d if d>0 else 0 for d in delta]; losses=[-d if d<0 else 0 for d in delta]
    avg_gain=sum(gains[:period])/period; avg_loss=sum(losses[:period])/period
    rsi=[np.nan]*period
    for i in range(period,len(close)):
        avg_gain=(avg_gain*(period-1)+gains[i-1])/period
        avg_loss=(avg_loss*(period-1)+losses[i-1])/period
        rs=avg_gain/avg_loss if avg_loss!=0 else 100
        rsi.append(100-100/(1+rs))
    return rsi

def calc_boll(close, period=20, std_mult=2):
    if len(close)<period: return [np.nan],[np.nan],[np.nan]
    upper,mid,lower=[],[],[]
    for i in range(period-1,len(close)):
        seg=close[i-period+1:i+1]
        mid.append(sum(seg)/period)
        std=np.std(seg)
        upper.append(mid[-1]+std_mult*std); lower.append(mid[-1]-std_mult*std)
    return upper,mid,lower

dif, dea = calc_macd(close)
k_vals, d_vals, j_vals = calc_kdj(high, low, close)
rsi_vals = calc_rsi(close, 14)
boll_upper, boll_mid, boll_lower = calc_boll(close)

macd_sig = '金叉' if dif[-2]<=dea[-2] and dif[-1]>dea[-1] else ('死叉' if dif[-2]>=dea[-2] and dif[-1]<dea[-1] else '中性')
kdj_sig = '金叉' if k_vals[-2]<=d_vals[-2] and k_vals[-1]>d_vals[-1] else ('死叉' if k_vals[-2]>=d_vals[-2] and k_vals[-1]<d_vals[-1] else '中性')
kdj_stat = '超买' if j_vals[-1]>80 else ('超卖' if j_vals[-1]<20 else '中性')
rsi_today = rsi_vals[-1] if not np.isnan(rsi_vals[-1]) else 50
rsi_stat = '超买' if rsi_today>70 else ('超卖' if rsi_today<30 else '中性')
boll_stat = '突破上轨' if close[-1]>boll_upper[-1] else ('跌破下轨' if close[-1]<boll_lower[-1] else '中性')
vol_avg5 = np.mean(vol[-6:-1]) if len(vol)>=6 else np.mean(vol)
vol_ratio = vol[-1]/vol_avg5 if vol_avg5>0 else 1
vol_stat = '放量' if vol_ratio>1.5 else ('缩量' if vol_ratio<0.7 else '正常')
ma5=np.mean(close[-5:]); ma10=np.mean(close[-10:]); ma20=np.mean(close[-20:])
ma_arr = '多头' if ma5>ma10>ma20 else ('空头' if ma5<ma10<ma20 else '混乱')

print()
print('=== 技术指标 ===')
print(f'MACD: {macd_sig}')
print(f'KDJ: {kdj_sig}, J={j_vals[-1]:.1f} ({kdj_stat})')
print(f'RSI(14): {rsi_today:.1f} ({rsi_stat})')
print(f'BOLL: {boll_stat}')
print(f'量比: {vol_ratio:.2f} ({vol_stat})')
print(f'均线排列: {ma_arr}')
print(f'MA5={ma5:.2f}, MA10={ma10:.2f}, MA20={ma20:.2f}')

# 评分
score = 0; notes = []
if 35 <= rsi_today <= 65:
    score += 20; notes.append('RSI适中')
elif rsi_today < 30:
    score += 15; notes.append('RSI超卖')
elif rsi_today > 80:
    score -= 15; notes.append('RSI超买')

if macd_sig == '金叉':
    score += 20; notes.append('MACD金叉')
elif macd_sig == '死叉':
    score -= 15; notes.append('MACD死叉')

if kdj_sig == '金叉' and j_vals[-1] < 60:
    score += 20; notes.append('KDJ低位金叉')
elif kdj_sig == '金叉':
    score += 10; notes.append('KDJ金叉')
elif kdj_sig == '死叉':
    score -= 10; notes.append('KDJ死叉')

if vol_ratio > 1.5:
    score += 15; notes.append('放量')
elif vol_ratio > 1.1:
    score += 8; notes.append('轻微放量')

if ma_arr == '多头':
    score += 15; notes.append('均线多头')
elif ma_arr == '空头':
    score -= 15; notes.append('均线空头')

if ret10 is not None and -5 <= ret10 <= 10:
    score += 10; notes.append('涨幅适中')
elif ret10 is not None and ret10 > 15:
    score -= 10; notes.append('短期涨幅过大')

print()
print(f'综合评分: {score}')
print(f'评价: {", ".join(notes)}')