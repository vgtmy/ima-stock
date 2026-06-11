#!/usr/bin/env python3
"""单只股票因子计算诊断脚本"""
import sys
import pickle
import os
import traceback

# 强制清除本目录的 .pyc 缓存
import shutil
cache_dir = os.path.join(os.path.dirname(__file__), '__pycache__')
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"已清除: {cache_dir}")

# 1) 检查文件版本
print("=" * 60)
print("【1】文件版本检查")
print("=" * 60)
for fname in ['data_fetcher.py', 'factor_engine.py', 'config.py', 'main.py']:
    p = os.path.join(os.path.dirname(__file__), fname)
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            content = f.read()
        mtime = os.path.getmtime(p)
        import datetime
        mod_time = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        # 检查关键标记
        marks = {
            'data_fetcher.py': ['_parallel_fetch_stocks', 'CONCURRENT_WORKERS'],
            'factor_engine.py': ['calc_margin_factors', 'calc_dividend_factors', 'calc_industry_factors', 'calc_chip_factors'],
            'config.py': ['CONCURRENT_WORKERS'],
            'main.py': ['run_etl_pipeline'],
        }
        print(f"\n  {fname} (修改: {mod_time}):")
        for mark in marks.get(fname, []):
            present = "✅" if mark in content else "❌"
            print(f"    {present} {mark}")

# 2) 检查快照
print()
print("=" * 60)
print("【2】快照文件检查")
print("=" * 60)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
if os.path.exists(data_dir):
    for f in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, f)
        size = os.path.getsize(p) / 1024 / 1024
        print(f"  {f} ({size:.1f} MB)")

# 3) 尝试用最新快照加载
print()
print("=" * 60)
print("【3】用最新快照跑单只股票因子")
print("=" * 60)
candidates = []
for f in sorted(os.listdir(data_dir)):
    if f.startswith('raw_data_') and f.endswith('.pkl'):
        candidates.append(f)
print(f"  找到候选文件: {candidates}")
if not candidates:
    print("❌ 没有快照文件！")
    sys.exit(1)

latest = candidates[-1]
print(f"  使用快照: {latest}")
try:
    with open(os.path.join(data_dir, latest), 'rb') as f:
        raw = pickle.load(f)
    print(f"  ✅ 快照加载成功")
except Exception as e:
    print(f"  ❌ 加载失败: {e}")
    traceback.print_exc()
    sys.exit(1)

print(f"  股票数: {len(raw['stock_list'])}")
print(f"  raw_data 包含: {list(raw.keys())}")
print(f"  klines: {len(raw.get('klines', {}))} 只")
print(f"  financials: {len(raw.get('financials', {}))} 只")
print(f"  fund_flows: {len(raw.get('fund_flows', {}))} 只")
print(f"  margin_trading: {len(raw.get('margin_trading', {}))} 只")
print(f"  chips: {len(raw.get('chips', {}))} 只")
print(f"  dividends: {len(raw.get('dividends', {}))} 只")
print(f"  industry_map: {len(raw.get('industry_map', {}))} 只")
print(f"  concept_map: {len(raw.get('concept_map', {}))} 只")
print(f"  shareholders: {len(raw.get('shareholders', {}))} 只")

# 4) 跑一只股票
print()
print("=" * 60)
print("【4】运行 calc_all_factors(单只)")
print("=" * 60)
from factor_engine import FactorEngine
engine = FactorEngine(raw)
codes = raw['stock_list']['股票代码'].tolist()
test_code = codes[0]
print(f"  测试股票: {test_code}")
try:
    factors = engine.calc_all_factors(test_code)
    print(f"  ✅ 成功！计算了 {len(factors)} 个因子")
    # 按层分组统计
    for k, v in list(factors.items())[:30]:
        print(f"    {k}: {v}")
    if len(factors) > 30:
        print(f"    ... 还有 {len(factors)-30} 个因子")
except Exception as e:
    print(f"  ❌ 异常: {e}")
    traceback.print_exc()

# 5) 跑50只验证
print()
print("=" * 60)
print("【5】批量跑前50只验证")
print("=" * 60)
success = 0
fail_codes = []
for code in codes[:50]:
    try:
        factors = engine.calc_all_factors(code)
        if factors and len(factors) > 5:
            success += 1
    except Exception as e:
        fail_codes.append((code, str(e)[:100]))
print(f"  成功: {success}/50")
if fail_codes:
    print(f"  失败样例（前5个）:")
    for code, err in fail_codes[:5]:
        print(f"    {code}: {err}")
