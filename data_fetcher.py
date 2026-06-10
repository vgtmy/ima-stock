#!/usr/bin/env python3
"""
数据采集模块 — data_fetcher.py (westock-data 腾讯源版)
======================================================
数据源: westock-data (腾讯自选股行情接口)
优势: 腾讯源稳定无限制，替代 akshare 东方财富被墙问题

采集内容:
- 股票列表 & 基础信息
- 日线K线（前复权）
- 三大财务报表
- A股资金流向（含主力/超大单/大单/中单/小单 + 北向资金）
- 融资融券
- 技术指标（MACD/KDJ/RSI/布林带等）
- 筹码成本
- 股东结构
- 分红数据
- 行业分类
"""
import os
import re
import sys
import json
import time
import pickle
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    DATA_DIR, OUTPUT_DIR, MAX_RETRIES, RETRY_DELAY,
    REQUEST_INTERVAL, LOOKBACK_DAYS, BATCH_SIZE,
    COOLDOWN_AFTER_FAILURES, COOLDOWN_SLEEP, CONCURRENT_WORKERS, logger
)

# ============================================================
# westock-data CLI 封装
# ============================================================
WESTOCK_CMD = "npx -y westock-data-skillhub@1.0.3"

def _run_westock(subcmd: str, args: str = "", timeout: int = 60, max_retries: int = None) -> str:
    """执行 westock-data CLI 命令，返回原始 stdout（UTF-8编码）"""
    if max_retries is None:
        max_retries = MAX_RETRIES
    cmd = f"{WESTOCK_CMD} {subcmd} {args}".strip()
    for attempt in range(max_retries):
        try:
            # Windows 默认用 GBK 解码，westock-data 输出 UTF-8，必须显式指定
            result = subprocess.run(
                cmd, shell=True, capture_output=True, timeout=timeout,
                encoding='utf-8', errors='replace'
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            # 检查是否返回了错误 JSON
            if result.stdout and result.stdout.strip().startswith("{") and '"success": false' in result.stdout:
                logger.debug(f"  [{subcmd}] API 返回错误: {result.stdout[:200]}")
            # stderr 可能有有用信息
            if result.stderr and result.stderr.strip():
                logger.debug(f"  [{subcmd}] stderr: {result.stderr[:200]}")
            raise RuntimeError(f"westock {subcmd} 返回空或失败 (rc={result.returncode})")
        except subprocess.TimeoutExpired:
            logger.warning(f"  [{subcmd}] 超时({timeout}s), 第{attempt+1}次")
        except Exception as e:
            if max_retries <= 1:
                # 单次尝试，不打warning，用debug
                logger.debug(f"  [{subcmd}] 失败: {e}")
            else:
                logger.warning(f"  [{subcmd}] 第{attempt+1}次失败: {e}")
        if attempt < max_retries - 1:
            wait = RETRY_DELAY * (2 ** attempt)
            time.sleep(wait)
    raise RuntimeError(f"westock {subcmd} 重试{max_retries}次后仍失败")


def _parse_markdown_table(text: str) -> Optional[pd.DataFrame]:
    """将 westock-data 返回的 Markdown 表格解析为 DataFrame"""
    if not text or not text.strip():
        return None
    lines = text.strip().split("\n")
    # 找到表头行
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and "---" in lines[i + 1] if i + 1 < len(lines) else False:
            header_idx = i
            break
    if header_idx is None:
        # 可能没有分隔行，尝试第一行
        for i, line in enumerate(lines):
            if line.strip().startswith("|"):
                header_idx = i
                break
    if header_idx is None:
        return None

    def parse_row(line):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        return cells

    headers = parse_row(lines[header_idx])
    # 跳过分隔行
    data_start = header_idx + 1
    if data_start < len(lines) and "---" in lines[data_start]:
        data_start += 1

    rows = []
    for line in lines[data_start:]:
        if line.strip().startswith("|") and not line.strip().startswith("| ---"):
            row = parse_row(line)
            if len(row) == len(headers):
                rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=headers)
    # 尝试数值转换
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    return df


def _parse_batch_markdown(text: str) -> Dict[str, pd.DataFrame]:
    """解析批量查询返回的 Markdown（含 [Batch] 头 + symbol/code 列）"""
    result = {}
    if not text or not text.strip():
        return result

    lines = text.strip().split("\n")
    # 跳过 [Batch] 状态行
    data_lines = []
    for line in lines:
        if line.strip().startswith("[Batch]"):
            continue
        data_lines.append(line)

    merged_text = "\n".join(data_lines)
    df = _parse_markdown_table(merged_text)
    if df is not None:
        # 优先用 symbol 列，其次用 code 列做分组
        group_col = None
        for col in ["symbol", "code"]:
            if col in df.columns:
                group_col = col
                break
        if group_col:
            for sym, group in df.groupby(group_col):
                result[str(sym)] = group.drop(columns=[group_col], errors="ignore").reset_index(drop=True)
        else:
            result["_single"] = df
    return result


# ============================================================
# 通用并发拉取框架
# ============================================================
def _parallel_fetch_stocks(
    codes: List[str],
    fetch_single_fn,
    label: str,
    progress_step: int = 100,
    max_workers: int = None,
) -> Dict:
    """
    通用并发拉取框架 — 对 codes 列表中的每个代码调用 fetch_single_fn(code)，
    用 ThreadPoolExecutor 并发执行，实时报告进度。

    fetch_single_fn 签名: (code: str) -> Optional[Tuple[str, Any]]
        - 成功返回 (code, data)
        - 失败返回 None
    """
    if max_workers is None:
        max_workers = CONCURRENT_WORKERS
    result = {}
    total = len(codes)
    done_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single_fn, code): code for code in codes}
        for future in as_completed(futures):
            done_count += 1
            try:
                res = future.result()
                if res is not None:
                    code, data = res
                    result[code] = data
            except Exception:
                pass
            if done_count % progress_step == 0 or done_count == total:
                logger.info(f"  [{label}] 进度: {done_count}/{total} ({len(result)}只)")

    return result


# ============================================================
# 缓存工具
# ============================================================
def save_cache(data, filename: str):
    """保存缓存到磁盘"""
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"缓存已保存: {filename}")

def load_cache(filename: str):
    """从磁盘加载缓存"""
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


# ============================================================
# 代码格式转换
# ============================================================
def _to_westock_code(code: str) -> str:
    """6位纯数字代码 → westock格式 (sh600000 / sz000001 / bj430047)"""
    if code.startswith(("sh", "sz", "bj", "hk", "us")):
        return code
    if code.startswith(("6", "9")):
        return f"sh{code}"
    elif code.startswith(("0", "3")):
        return f"sz{code}"
    elif code.startswith("4") or code.startswith("8"):
        return f"bj{code}"
    return f"sz{code}"  # fallback


def _from_westock_code(ws_code: str) -> str:
    """westock格式 → 6位纯数字"""
    for prefix in ("sh", "sz", "bj"):
        if ws_code.startswith(prefix):
            return ws_code[2:]
    return ws_code


# ============================================================
# 1. 股票列表（批量 profile 获取基础信息）
# ============================================================
def fetch_stock_list(use_cache: bool = True) -> pd.DataFrame:
    """获取A股全量股票列表 — 多重 fallback 确保一定能拿到"""
    cache_file = "stock_list.pkl"
    if use_cache:
        cached = load_cache(cache_file)
        if cached is not None and len(cached) > 0:
            logger.info(f"[股票列表] 从缓存加载: {len(cached)} 只")
            return cached

    # ---- 方法1: akshare stock_info_a_code_name (SSE+SZSE) ----
    logger.info("[股票列表] 方法1: akshare stock_info_a_code_name...")
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        df = df.rename(columns={"code": "股票代码", "name": "证券名称"})
        df = df[df["股票代码"].str.match(r'^[063]\d{5}$')].copy()
        df = df.drop_duplicates(subset="股票代码").reset_index(drop=True)
        if len(df) > 100:
            logger.info(f"[股票列表] ✅ 方法1成功: {len(df)} 只")
            save_cache(df, cache_file)
            return df
    except Exception as e:
        logger.warning(f"[股票列表] 方法1失败: {e}")

    # ---- 方法2: akshare stock_zh_a_spot_em (东方财富实时行情，含全量代码) ----
    logger.info("[股票列表] 方法2: akshare stock_zh_a_spot_em...")
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        df = df.rename(columns={"代码": "股票代码", "名称": "证券名称"})
        df = df[df["股票代码"].str.match(r'^[063]\d{5}$')].copy()
        df = df.drop_duplicates(subset="股票代码").reset_index(drop=True)
        if len(df) > 100:
            logger.info(f"[股票列表] ✅ 方法2成功: {len(df)} 只")
            save_cache(df, cache_file)
            return df
    except Exception as e:
        logger.warning(f"[股票列表] 方法2失败: {e}")

    # ---- 方法3: westock-data search 扫描各市场 ----
    logger.info("[股票列表] 方法3: westock-data search 扫描...")
    try:
        all_codes = set()
        # 搜索沪市主板、深市主板、创业板、科创板、北交所
        for market_query in ["sh600", "sh601", "sh603", "sh605", "sz000", "sz001", "sz002", "sz003", "sz300", "sz301", "sh688", "bj4", "bj8"]:
            try:
                text = _run_westock("search", market_query, timeout=30)
                df = _parse_markdown_table(text)
                if df is not None and "code" in df.columns:
                    for _, row in df.iterrows():
                        code = str(row.get("code", ""))
                        # 提取6位纯数字
                        pure = _from_westock_code(code) if code.startswith(("sh", "sz", "bj")) else code
                        if len(pure) == 6 and pure[0] in "013684":
                            all_codes.add(pure)
            except Exception:
                pass
            time.sleep(0.3)

        if len(all_codes) > 100:
            df = pd.DataFrame({"股票代码": sorted(all_codes), "证券名称": [""] * len(all_codes)})
            logger.info(f"[股票列表] ✅ 方法3成功: {len(df)} 只")
            save_cache(df, cache_file)
            return df
    except Exception as e:
        logger.warning(f"[股票列表] 方法3失败: {e}")

    # ---- 方法4: 从已知代码范围生成 + westock验证 ----
    logger.info("[股票列表] 方法4: 生成代码范围 + westock验证...")
    try:
        # 生成所有可能的A股代码
        code_ranges = []
        # 沪市主板: 600000-684000
        for prefix in ["600", "601", "603", "605", "688"]:
            for i in range(0, 1000):
                code_ranges.append(f"{prefix}{i:03d}")
        # 深市主板: 000001-004999
        for prefix in ["000", "001", "002", "003"]:
            for i in range(0, 1000):
                code_ranges.append(f"{prefix}{i:03d}")
        # 创业板: 300001-301999
        for prefix in ["300", "301"]:
            for i in range(0, 1000):
                code_ranges.append(f"{prefix}{i:03d}")
        # 北交所: 430001-873999 (简化)
        for prefix in ["430", "830", "870", "871", "872", "873"]:
            for i in range(0, 1000):
                code_ranges.append(f"{prefix}{i:03d}")

        # 用 westock kline 批量验证哪些代码存在（每批20只）
        valid_codes = []
        batch_size = 20
        total = len(code_ranges)
        logger.info(f"  生成 {total} 个候选代码，开始验证...")

        for start in range(0, total, batch_size):
            batch = code_ranges[start:start + batch_size]
            symbols = [_to_westock_code(c) for c in batch]
            batch_str = ",".join(symbols)
            try:
                text = _run_westock("kline", f"{batch_str} --period day --limit 1", timeout=30)
                parsed = _parse_batch_markdown(text)
                for sym in parsed.keys():
                    if sym != "_single":
                        valid_codes.append(_from_westock_code(sym))
            except Exception:
                pass

            if (start + batch_size) % 5000 < batch_size:
                logger.info(f"  验证进度: {min(start+batch_size, total)}/{total} ({len(valid_codes)}只有效)")

            time.sleep(0.1)

        if len(valid_codes) > 100:
            df = pd.DataFrame({"股票代码": sorted(valid_codes), "证券名称": [""] * len(valid_codes)})
            logger.info(f"[股票列表] ✅ 方法4成功: {len(df)} 只")
            save_cache(df, cache_file)
            return df
    except Exception as e:
        logger.warning(f"[股票列表] 方法4失败: {e}")

    # ---- 全部失败 ----
    logger.error("[股票列表] ❌ 所有方法均失败！请检查网络或手动提供 stock_codes.txt")
    df = pd.DataFrame(columns=["股票代码", "证券名称"])
    save_cache(df, cache_file)
    return df


# ============================================================
# 2. 日线K线
# ============================================================
def _fetch_kline_batch(codes: List[str], period: str = "day", limit: int = 300) -> Dict[str, pd.DataFrame]:
    """批量获取K线数据（westock-data 支持逗号分隔批量）"""
    symbols = [_to_westock_code(c) for c in codes]
    # westock 批量查询限制：每次最多 20 只
    result = {}
    batch_size = 20

    for start in range(0, len(symbols), batch_size):
        batch = symbols[start:start + batch_size]
        batch_str = ",".join(batch)
        try:
            text = _run_westock("kline", f"{batch_str} --period {period} --limit {limit} --fq qfq", timeout=120)
            parsed = _parse_batch_markdown(text)
            for sym, df in parsed.items():
                if sym == "_single":
                    continue
                pure_code = _from_westock_code(sym)
                # 统一列名
                col_map = {
                    "date": "日期", "open": "开盘", "last": "收盘",
                    "high": "最高", "low": "最低", "volume": "成交量",
                    "amount": "成交额", "exchange": "换手率",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                # 补算缺失字段
                if "收盘" in df.columns:
                    close = pd.to_numeric(df["收盘"], errors="coerce")
                    df["涨跌额"] = close.diff()
                    df["涨跌幅"] = close.pct_change() * 100
                    if "最高" in df.columns and "最低" in df.columns:
                        high = pd.to_numeric(df["最高"], errors="coerce")
                        low = pd.to_numeric(df["最低"], errors="coerce")
                        df["振幅"] = (high - low) / close.shift(1) * 100
                df["股票代码"] = pure_code
                result[pure_code] = df
        except Exception as e:
            logger.debug(f"  K线批量获取失败 ({batch_str[:30]}...): {e}")
            # 回退：逐只获取
            for sym in batch:
                try:
                    text = _run_westock("kline", f"{sym} --period {period} --limit {limit} --fq qfq", timeout=30)
                    df = _parse_markdown_table(text)
                    if df is not None and len(df) > 0:
                        col_map = {
                            "date": "日期", "open": "开盘", "last": "收盘",
                            "high": "最高", "low": "最低", "volume": "成交量",
                            "amount": "成交额", "exchange": "换手率",
                        }
                        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                        pure_code = _from_westock_code(sym)
                        if "收盘" in df.columns:
                            close = pd.to_numeric(df["收盘"], errors="coerce")
                            df["涨跌额"] = close.diff()
                            df["涨跌幅"] = close.pct_change() * 100
                            if "最高" in df.columns and "最低" in df.columns:
                                high = pd.to_numeric(df["最高"], errors="coerce")
                                low = pd.to_numeric(df["最低"], errors="coerce")
                                df["振幅"] = (high - low) / close.shift(1) * 100
                        df["股票代码"] = pure_code
                        result[pure_code] = df
                except Exception:
                    pass
                time.sleep(0.3)

        time.sleep(REQUEST_INTERVAL)

    return result


def fetch_all_klines(stock_list: pd.DataFrame, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
    """
    批量获取全量股票日线K线 — 支持断点续跑
    每100只自动存缓存，中断后重跑自动跳过已完成的
    返回: {股票代码: DataFrame}
    """
    limit = min(LOOKBACK_DAYS + 50, 2000)  # westock 最大 2000
    codes = stock_list["股票代码"].tolist()

    # ---- 断点续跑：加载已有缓存 ----
    kline_cache_file = "klines_cache.pkl"
    all_data = load_cache(kline_cache_file)
    if all_data is not None and len(all_data) > 0:
        # 过滤掉已完成的代码
        remaining = [c for c in codes if c not in all_data]
        logger.info(f"[日线K线] 🔄 断点续跑: 已有 {len(all_data)} 只, 剩余 {len(remaining)} 只")
        codes = remaining
        if len(codes) == 0:
            logger.info(f"[日线K线] ✅ 全部完成 (缓存恢复)")
            return all_data
    else:
        all_data = {}
        logger.info(f"[日线K线] 批量拉取 {len(codes)} 只, limit={limit}")

    consecutive_failures = 0
    save_counter = 0

    for i in range(0, len(codes), 20):
        batch = codes[i:i + 20]
        try:
            batch_result = _fetch_kline_batch(batch, limit=limit)
            all_data.update(batch_result)
            consecutive_failures = 0
            save_counter += len(batch_result)
        except Exception:
            consecutive_failures += len(batch)

        # 连续失败冷却
        if consecutive_failures >= COOLDOWN_AFTER_FAILURES * 20:
            logger.warning(f"  ⚠️ 连续失败较多，进入冷却 {COOLDOWN_SLEEP:.0f}s...")
            # 先保存当前进度
            save_cache(all_data, kline_cache_file)
            time.sleep(COOLDOWN_SLEEP)
            consecutive_failures = 0

        # 每100只保存一次增量缓存
        if save_counter >= 100:
            save_cache(all_data, kline_cache_file)
            logger.info(f"  💾 增量缓存已保存 ({len(all_data)}只)")
            save_counter = 0

        if (i + 20) % BATCH_SIZE < 20:
            logger.info(f"  进度: {len(all_data)}/{len(stock_list)} ({len(all_data)}只成功)")

    # 最终保存
    save_cache(all_data, kline_cache_file)
    logger.info(f"[日线K线] ✅ 成功拉取 {len(all_data)}/{len(stock_list)} 只")
    return all_data


# ============================================================
# 3. 财务报表
# ============================================================
def _fetch_finance_batch(codes: List[str], num_periods: int = 4) -> Dict[str, Dict[str, pd.DataFrame]]:
    """获取财务报表 — 逐只获取（finance 不支持批量，rc=1通常表示无数据不需重试）"""
    symbols = [_to_westock_code(c) for c in codes]
    result = {}

    for sym in symbols:
        try:
            # finance 通常不支持批量，且 rc=1 多为"无数据"，只试1次
            text = _run_westock("finance", f"{sym} --num {num_periods}", timeout=30, max_retries=1)
            reports = {}
            sections = re.split(r'\n\*\*(lrb|zcfz|xjll)\*\*\n', text)
            pure_code = _from_westock_code(sym)

            for idx in range(1, len(sections), 2):
                rtype_key = sections[idx]  # lrb / zcfz / xjll
                table_text = sections[idx + 1] if idx + 1 < len(sections) else ""
                df = _parse_markdown_table(table_text)
                if df is not None and len(df) > 0:
                    type_map = {"lrb": "利润表", "zcfz": "资产负债表", "xjll": "现金流量表"}
                    reports[type_map.get(rtype_key, rtype_key)] = df

            if reports:
                result[pure_code] = reports
        except Exception:
            pass
        time.sleep(0.15)  # 财报逐只，短间隔即可

    return result


def fetch_all_financials(stock_list: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    批量获取全量股票三大报表 — 支持断点续跑
    每100只自动存缓存，中断后重跑自动跳过已完成的
    返回: {股票代码: {"利润表": df, "资产负债表": df, "现金流量表": df}}
    """
    codes = stock_list["股票代码"].tolist()

    # ---- 断点续跑 ----
    fin_cache_file = "financials_cache.pkl"
    all_data = load_cache(fin_cache_file)
    if all_data is not None and len(all_data) > 0:
        remaining = [c for c in codes if c not in all_data]
        logger.info(f"[财务报表] 🔄 断点续跑: 已有 {len(all_data)} 只, 剩余 {len(remaining)} 只")
        codes = remaining
        if len(codes) == 0:
            logger.info(f"[财务报表] ✅ 全部完成 (缓存恢复)")
            return all_data
    else:
        all_data = {}
        logger.info(f"[财务报表] 批量拉取 {len(codes)} 只")

    save_counter = 0

    for i in range(0, len(codes), 10):
        batch = codes[i:i + 10]
        try:
            batch_result = _fetch_finance_batch(batch)
            all_data.update(batch_result)
            save_counter += len(batch_result)
        except Exception:
            pass

        # 每100只保存一次增量缓存
        if save_counter >= 100:
            save_cache(all_data, fin_cache_file)
            logger.info(f"  💾 增量缓存已保存 ({len(all_data)}只)")
            save_counter = 0

        if (i + 10) % BATCH_SIZE < 10:
            logger.info(f"  进度: {len(all_data)}/{len(stock_list)} ({len(all_data)}只完整)")

    # 最终保存
    save_cache(all_data, fin_cache_file)
    logger.info(f"[财务报表] ✅ 完整获取 {len(all_data)}/{len(stock_list)} 只")
    return all_data


# ============================================================
# 4. A股资金流向（替代原 akshare stock_individual_fund_flow + 北向资金）
# ============================================================
def _fetch_asfund_batch(codes: List[str]) -> Dict[str, pd.DataFrame]:
    """批量获取A股资金流向"""
    symbols = [_to_westock_code(c) for c in codes]
    result = {}
    batch_size = 20

    for start in range(0, len(symbols), batch_size):
        batch = symbols[start:start + batch_size]
        batch_str = ",".join(batch)
        try:
            text = _run_westock("asfund", batch_str, timeout=60)
            parsed = _parse_batch_markdown(text)
            for sym, df in parsed.items():
                if sym == "_single":
                    continue
                pure_code = _from_westock_code(sym)
                # 统一列名映射
                col_map = {
                    "MainNetFlow": "主力净流入",
                    "JumboNetFlow": "超大单净流入",
                    "BlockNetFlow": "大单净流入",
                    "MidNetFlow": "中单净流入",
                    "SmallNetFlow": "小单净流入",
                    "ClosePrice": "收盘价",
                    "EndDate": "日期",
                    "LgtHoldInfo": "北向资金信息",
                    "MarginTradeInfos": "融资融券信息",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                result[pure_code] = df
        except Exception:
            # 逐只回退
            for sym in batch:
                try:
                    text = _run_westock("asfund", sym, timeout=30)
                    df = _parse_markdown_table(text)
                    if df is not None and len(df) > 0:
                        pure_code = _from_westock_code(sym)
                        col_map = {
                            "MainNetFlow": "主力净流入",
                            "JumboNetFlow": "超大单净流入",
                            "BlockNetFlow": "大单净流入",
                            "MidNetFlow": "中单净流入",
                            "SmallNetFlow": "小单净流入",
                            "ClosePrice": "收盘价",
                            "EndDate": "日期",
                            "LgtHoldInfo": "北向资金信息",
                            "MarginTradeInfos": "融资融券信息",
                        }
                        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                        result[pure_code] = df
                except Exception:
                    pass
                time.sleep(0.15)

        time.sleep(REQUEST_INTERVAL)

    return result


def fetch_all_fund_flow(stock_list: pd.DataFrame, days: int = 30) -> Dict[str, pd.DataFrame]:
    """批量获取个股资金流数据（westock asfund，腾讯源，无限流）"""
    logger.info(f"[资金流] 批量拉取 {len(stock_list)} 只 (腾讯源)")
    all_data = {}
    codes = stock_list["股票代码"].tolist()

    for i in range(0, len(codes), 20):
        batch = codes[i:i + 20]
        try:
            batch_result = _fetch_asfund_batch(batch)
            all_data.update(batch_result)
        except Exception:
            pass
        if (i + 20) % BATCH_SIZE < 20:
            logger.info(f"  进度: {min(i+20, len(codes))}/{len(codes)} ({len(all_data)}只)")

    logger.info(f"[资金流] ✅ 获取 {len(all_data)}/{len(codes)} 只")
    return all_data


# ============================================================
# 5. 融资融券
# ============================================================
def _fetch_margintrade_single(code: str):
    """获取单只股票融资融券数据"""
    ws_code = _to_westock_code(code)
    try:
        text = _run_westock("margintrade", ws_code, timeout=30, max_retries=1)
        df = _parse_markdown_table(text)
        if df is not None and len(df) > 0:
            col_map = {
                "FinanceValue": "融资余额",
                "SecurityValue": "融券余额",
                "FinanceBuyValue": "融资买入额",
                "FinanceRefundValue": "融资偿还额",
                "TradingValue": "融资融券余额",
                "TradingValueDif": "融资融券余额差额",
                "FinanceValueDOD": "融资余额日变动",
                "SecurityValueDOD": "融券余额日变动",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            return (code, df)
    except Exception:
        pass
    return None


def fetch_margin_data(date_str: str = None) -> Dict[str, pd.DataFrame]:
    """获取融资融券数据 — westock 版（逐只获取，返回与旧接口兼容的 dict）"""
    # 注意：旧版返回 DataFrame，新版返回 Dict[str, DataFrame]
    # margintrade 需要逐只调用，这里从 stock_list 缓存获取代码列表
    logger.info("[融资融券] 通过 westock margintrade 获取...")
    # 此函数在 fetch_all_data 中被调用时，需要 stock_list
    # 暂时返回空，实际在 fetch_all_data 中统一处理
    return {}


# ============================================================
# 6. 北向资金 — 已合并到 asfund 的 LgtHoldInfo 字段
# ============================================================
def fetch_north_bound(stock_list: pd.DataFrame) -> Dict[str, dict]:
    """
    北向资金数据 — 从 asfund 的 LgtHoldInfo 字段提取
    注意：此函数在 fetch_all_data 中从 fund_flows 数据中提取，
    不再单独调用东方财富接口
    """
    logger.info("[北向资金] 从资金流数据中提取（asfund LgtHoldInfo 字段）")
    # 实际提取逻辑在 fetch_all_data 中统一处理
    return {}


# ============================================================
# 7. 股东结构
# ============================================================
def _fetch_shareholder_single(code: str):
    """获取单只股票股东结构（1次重试），返回 (code, data) 或 None"""
    ws_code = _to_westock_code(code)
    try:
        text = _run_westock("shareholder", ws_code, timeout=30, max_retries=1)
        # shareholder 返回混合格式（标题 + 多个表格），需要分段解析
        result = {"股票代码": code}

        # 解析十大股东
        sections = text.split("十大流通股东")
        if len(sections) >= 1:
            top10_text = sections[0]
            # 提取十大股东表格
            match = re.search(r'十大股东\n+(.*)', top10_text, re.DOTALL)
            if match:
                table_text = match.group(1)
                # 找下一个标题前的内容
                next_title = re.search(r'\n\n[^|]', table_text)
                if next_title:
                    table_text = table_text[:next_title.start()]
                df = _parse_markdown_table(table_text)
                if df is not None:
                    result["十大股东"] = df

        if len(sections) >= 2:
            # 十大流通股东
            float_text = sections[1]
            # 股东户数
            holder_count_match = re.search(r'股东户数[：:]\s*(\d+)', float_text)
            if holder_count_match:
                result["股东户数"] = int(holder_count_match.group(1))
            # 解析表格
            df = _parse_markdown_table(float_text)
            if df is not None:
                result["十大流通股东"] = df

        if len(result) > 1:
            return (code, result)
    except Exception:
        pass
    return None


def fetch_shareholders(stock_list: pd.DataFrame) -> Dict[str, dict]:
    """获取股东结构数据（8线程并发）"""
    codes = stock_list["股票代码"].tolist()
    logger.info(f"[股东结构] 并发拉取 {len(codes)} 只 ({CONCURRENT_WORKERS}线程)...")
    result = _parallel_fetch_stocks(codes, _fetch_shareholder_single, "股东", progress_step=BATCH_SIZE)
    logger.info(f"[股东结构] ✅ 获取 {len(result)}/{len(codes)} 只")
    return result
    return all_data


# ============================================================
# 8. 技术指标
# ============================================================
def _fetch_technical_batch(codes: List[str]) -> Dict[str, pd.DataFrame]:
    """批量获取技术指标"""
    symbols = [_to_westock_code(c) for c in codes]
    result = {}
    batch_size = 10  # technical 数据量大

    for start in range(0, len(symbols), batch_size):
        batch = symbols[start:start + batch_size]
        batch_str = ",".join(batch)
        try:
            text = _run_westock("technical", f"{batch_str} --group all", timeout=60)
            parsed = _parse_batch_markdown(text)
            for sym, df in parsed.items():
                if sym == "_single":
                    continue
                pure_code = _from_westock_code(sym)
                result[pure_code] = df
        except Exception:
            for sym in batch:
                try:
                    text = _run_westock("technical", f"{sym} --group all", timeout=30, max_retries=1)
                    df = _parse_markdown_table(text)
                    if df is not None and len(df) > 0:
                        pure_code = _from_westock_code(sym)
                        result[pure_code] = df
                except Exception:
                    pass
                time.sleep(0.15)
        time.sleep(REQUEST_INTERVAL)

    return result


# ============================================================
# 9. 筹码成本
# ============================================================
def _fetch_chip_single(code: str):
    """获取单只股票筹码成本，返回 (code, df) 或 None"""
    ws_code = _to_westock_code(code)
    try:
        text = _run_westock("chip", ws_code, timeout=30, max_retries=1)
        df = _parse_markdown_table(text)
        if df is not None and len(df) > 0:
            col_map = {
                "chipProfitRate": "盈利率",
                "chipAvgCost": "平均成本",
                "chipConcentration90": "集中度90",
                "chipConcentration70": "集中度70",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            return (code, df)
    except Exception:
        pass
    return None


# ============================================================
# 10. 分红数据
# ============================================================
def _fetch_dividend_single(code: str, years: int = 5):
    """获取单只股票分红数据，返回 (code, df) 或 None"""
    ws_code = _to_westock_code(code)
    try:
        text = _run_westock("dividend", f"{ws_code} --years {years}", timeout=30, max_retries=1)
        df = _parse_markdown_table(text)
        if df is not None and len(df) > 0:
            return (code, df)
    except Exception:
        pass
    return None


# ============================================================
# 11. 行业分类（从 profile 提取）
# ============================================================
def fetch_industry_map() -> Dict[str, str]:
    """获取股票→行业映射 — 从 profile 批量提取 industry 字段"""
    logger.info("[行业分类] 从 profile 提取行业信息...")
    mapping = {}

    # 先从缓存获取股票列表
    stock_list = load_cache("stock_list.pkl")
    if stock_list is None:
        logger.warning("[行业分类] 无股票列表缓存，跳过")
        return mapping

    codes = stock_list["股票代码"].tolist()
    batch_size = 20

    for start in range(0, len(codes), batch_size):
        batch = codes[start:start + batch_size]
        symbols = [_to_westock_code(c) for c in batch]
        batch_str = ",".join(symbols)
        try:
            text = _run_westock("profile", batch_str, timeout=60)
            df = _parse_markdown_table(text)
            if df is not None and "code" in df.columns and "industry" in df.columns:
                for _, row in df.iterrows():
                    code = str(row.get("code", ""))
                    industry = str(row.get("industry", ""))
                    pure_code = _from_westock_code(code)
                    if industry and industry != "nan":
                        mapping[pure_code] = industry
        except Exception:
            pass

        if (start + batch_size) % 500 < batch_size:
            logger.info(f"  进度: {min(start+batch_size, len(codes))}/{len(codes)} ({len(mapping)}只映射)")
        time.sleep(REQUEST_INTERVAL)

    logger.info(f"[行业分类] ✅ 映射 {len(mapping)} 只 → 行业")
    return mapping


# ============================================================
# 12. 概念板块（暂从 profile 提取 sector）
# ============================================================
def fetch_concept_map() -> Dict[str, List[str]]:
    """获取股票→概念标签映射 — 从 profile 的 sector 字段提取"""
    logger.info("[概念板块] 从 profile 提取板块信息...")
    mapping = {}

    stock_list = load_cache("stock_list.pkl")
    if stock_list is None:
        logger.warning("[概念板块] 无股票列表缓存，跳过")
        return mapping

    codes = stock_list["股票代码"].tolist()
    batch_size = 20

    for start in range(0, len(codes), batch_size):
        batch = codes[start:start + batch_size]
        symbols = [_to_westock_code(c) for c in batch]
        batch_str = ",".join(symbols)
        try:
            text = _run_westock("profile", batch_str, timeout=60)
            df = _parse_markdown_table(text)
            if df is not None and "code" in df.columns and "sector" in df.columns:
                for _, row in df.iterrows():
                    code = str(row.get("code", ""))
                    sector = str(row.get("sector", ""))
                    pure_code = _from_westock_code(code)
                    if sector and sector != "nan":
                        mapping.setdefault(pure_code, []).append(sector)
        except Exception:
            pass
        time.sleep(REQUEST_INTERVAL)

    logger.info(f"[概念板块] ✅ 映射 {len(mapping)} 只")
    return mapping


# ============================================================
# 主采集函数
# ============================================================
def fetch_all_data(target_date: str = None, use_cache: bool = True) -> dict:
    """
    一键采集所有原始数据（westock-data 腾讯源版）
    target_date: YYYY-MM-DD 格式，默认最新交易日
    use_cache: 是否使用各模块的增量缓存（K线/财报等）
    返回包含所有原始数据的字典
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    # ---- 尝试从完整快照恢复（如果存在则直接跳过整个采集）----
    snapshot_file = f"raw_data_{target_date}.pkl"
    snapshot_path = os.path.join(DATA_DIR, snapshot_file)
    if os.path.exists(snapshot_path):
        try:
            with open(snapshot_path, "rb") as f:
                raw_data = pickle.load(f)
            logger.info(f"📦 从快照恢复原始数据: {snapshot_file} ({len(raw_data.get('stock_list', []))} 只股票)")
            return raw_data
        except Exception as e:
            logger.warning(f"⚠️ 快照加载失败，重新采集: {e}")

    start_date = (datetime.strptime(target_date, "%Y-%m-%d")
                  - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    end_date = target_date.replace("-", "")

    logger.info("=" * 60)
    logger.info(f"🚀 开始全量数据采集 (westock-data 腾讯源), 目标日期: {target_date}")
    logger.info("=" * 60)

    # 1. 股票列表
    stock_list = fetch_stock_list(use_cache=use_cache)

    # 2. 日线K线（自带增量缓存 klines_cache.pkl，断点续跑）
    klines = fetch_all_klines(stock_list, start_date, end_date)

    # 3. 财务报表（自带增量缓存 financials_cache.pkl，断点续跑）
    financials = fetch_all_financials(stock_list)

    # 4. 资金流向（含北向资金 LgtHoldInfo）
    fund_flows = fetch_all_fund_flow(stock_list)

    # 5. 融资融券（从 asfund 的 MarginTradeInfos 或单独 margintrade）
    codes = stock_list["股票代码"].tolist()
    logger.info(f"[融资融券] 并发拉取 {len(codes)} 只 ({CONCURRENT_WORKERS}线程)...")
    margin = _parallel_fetch_stocks(codes, _fetch_margintrade_single, "融资融券", progress_step=BATCH_SIZE)
    logger.info(f"[融资融券] ✅ 获取 {len(margin)}/{len(codes)} 只")

    # 6. 北向资金 — 从 fund_flows 的 LgtHoldInfo 提取
    north_bound = {}
    logger.info("[北向资金] 从资金流数据提取 LgtHoldInfo...")
    for code, ff_df in fund_flows.items():
        if isinstance(ff_df, pd.DataFrame) and "北向资金信息" in ff_df.columns:
            info = ff_df.iloc[0].get("北向资金信息")
            if info and str(info) != "nan":
                try:
                    # LgtHoldInfo 是 JSON 字符串
                    if isinstance(info, str):
                        north_bound[code] = json.loads(info)
                    elif isinstance(info, dict):
                        north_bound[code] = info
                except Exception:
                    pass
    logger.info(f"[北向资金] ✅ 提取 {len(north_bound)} 只")

    # 7. 股东结构
    shareholders = fetch_shareholders(stock_list)

    # 8. 技术指标
    logger.info("[技术指标] 批量拉取...")
    technicals = {}
    for i in range(0, len(codes), 10):
        batch = codes[i:i + 10]
        try:
            batch_result = _fetch_technical_batch(batch)
            technicals.update(batch_result)
        except Exception:
            pass
        if (i + 10) % BATCH_SIZE < 10:
            logger.info(f"  进度: {min(i+10, len(codes))}/{len(codes)} ({len(technicals)}只)")
    logger.info(f"[技术指标] ✅ 获取 {len(technicals)}/{len(codes)} 只")

    # 9. 筹码成本
    logger.info(f"[筹码成本] 并发拉取 {len(codes)} 只 ({CONCURRENT_WORKERS}线程)...")
    chips = _parallel_fetch_stocks(codes, _fetch_chip_single, "筹码", progress_step=BATCH_SIZE)
    logger.info(f"[筹码成本] ✅ 获取 {len(chips)}/{len(codes)} 只")

    # 10. 分红数据
    logger.info(f"[分红数据] 并发拉取 {len(codes)} 只 ({CONCURRENT_WORKERS}线程)...")
    dividends = _parallel_fetch_stocks(codes, _fetch_dividend_single, "分红", progress_step=BATCH_SIZE)
    logger.info(f"[分红数据] ✅ 获取 {len(dividends)}/{len(codes)} 只")

    # 11. 行业分类
    industry_map = fetch_industry_map()

    # 12. 概念板块
    concept_map = fetch_concept_map()

    logger.info("=" * 60)
    logger.info("✅ 数据采集完成")
    logger.info("=" * 60)

    # 全部完成，清除增量缓存（可选，保留下次增量更新时仍可用）
    for cache_name in ["klines_cache.pkl", "financials_cache.pkl"]:
        cp = os.path.join(DATA_DIR, cache_name)
        if os.path.exists(cp):
            # 不删除，保留用于下次增量更新
            logger.info(f"[缓存] 保留 {cache_name} 用于下次增量更新")

    # ---- 保存完整快照到磁盘（下次直接加载，跳过整个采集过程）----
    raw_data = {
        "target_date": target_date,
        "stock_list": stock_list,
        "klines": klines,
        "financials": financials,
        "fund_flows": fund_flows,
        "margin": margin,
        "north_bound": north_bound,
        "shareholders": shareholders,
        "technicals": technicals,
        "chips": chips,
        "dividends": dividends,
        "industry_map": industry_map,
        "concept_map": concept_map,
    }
    try:
        sp = os.path.join(DATA_DIR, f"raw_data_{target_date}.pkl")
        with open(sp, "wb") as f:
            pickle.dump(raw_data, f)
        logger.info(f"📦 原始数据快照已保存: raw_data_{target_date}.pkl ({len(stock_list)} 只股票)")
    except Exception as e:
        logger.warning(f"⚠️ 快照保存失败（不影响本次执行）: {e}")

    return raw_data
