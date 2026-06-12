#!/usr/bin/env python3
"""
InStock 整合桥接模块 — instock_bridge.py
=========================================
将 stock_etl 的 raw_data 写入 InStock 的 MySQL 数据库，使 InStock Web UI
(localhost:9988) 能够展示 westock-data 腾讯源数据。

同时利用 InStock 的 TA-Lib 指标计算引擎为 factor_engine 提供更精准的
技术指标（74个），替代 factor_engine 中的 pandas 手算实现。

依赖：
  - MySQL 运行中（默认 localhost:3306，instockdb 数据库）
  - stock/ 目录已克隆到 /d/stock_etl/stock/
  - pip install PyMySQL SQLAlchemy（已在 requirements.txt 中）
  - TA-Lib C 库已安装（可选，未安装时技术指标 fallback 到 pandas）

写入的表：
  - cn_stock_spot        每日行情（行情 + 基本面数据）
  - cn_stock_indicators  技术指标（74个 TA-Lib 指标）
  - cn_stock_fund_flow   资金流向（主力/超大/大/中/小单）
"""
import sys
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger("stock_etl")

# ============================================================
# TA-Lib 可用性检测（可选依赖）
# ============================================================
_TALIB_AVAILABLE = False
_get_indicators = None

try:
    import talib  # noqa: F401
    from config import INSTOCK_ROOT
    _instock_indicator_path = os.path.join(INSTOCK_ROOT, "instock", "core", "indicator")
    if _instock_indicator_path not in sys.path:
        sys.path.insert(0, INSTOCK_ROOT)
    from instock.core.indicator.calculate_indicator import get_indicators as _get_indicators
    _TALIB_AVAILABLE = True
    logger.info("[InStock] TA-Lib 已加载，将使用 InStock 技术指标引擎")
except ImportError as e:
    logger.warning(f"[InStock] TA-Lib 未安装或 InStock 路径不可用：{e}，技术指标将使用 pandas fallback")

# ============================================================
# MySQL 连接
# ============================================================
_engine_instance = None


def _get_engine():
    global _engine_instance
    if _engine_instance is not None:
        return _engine_instance
    try:
        from sqlalchemy import create_engine
        from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT
        url = (f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
               f"@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4")
        _engine_instance = create_engine(url, pool_recycle=3600, pool_pre_ping=True)
        return _engine_instance
    except Exception as e:
        logger.warning(f"[InStock] 创建 MySQL engine 失败：{e}")
        return None


def check_connection() -> bool:
    """检测 MySQL 是否可达，返回 True/False"""
    try:
        from sqlalchemy import text
        eng = _get_engine()
        if eng is None:
            return False
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning(f"[InStock] MySQL 连接失败：{e}")
        return False


# ============================================================
# 列名映射：stock_etl（中文）→ InStock（英文）
# ============================================================
KLINE_COL_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "p_change",
    "换手率": "turnover",
    "股票代码": "code",
}


def _prepare_kline_df(code: str, kline_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """将 stock_etl 的 kline_df 转换为 InStock 期望的格式"""
    if kline_df is None or len(kline_df) < 10:
        return None
    df = kline_df.copy()
    df = df.rename(columns={k: v for k, v in KLINE_COL_MAP.items() if k in df.columns})
    df["code"] = code

    for col in ["open", "close", "high", "low", "volume", "amount", "p_change"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 确保 date 为字符串格式 YYYY-MM-DD
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)

    return df


# ============================================================
# 技术指标计算（TA-Lib）
# ============================================================

def calc_talib_indicators(code: str, kline_df: pd.DataFrame) -> Optional[pd.Series]:
    """
    用 InStock 的 TA-Lib 引擎计算单只股票指标，返回最后一行的 Series。
    若 TA-Lib 不可用或数据不足则返回 None。
    """
    if not _TALIB_AVAILABLE or _get_indicators is None:
        return None
    df = _prepare_kline_df(code, kline_df)
    if df is None or len(df) < 90:
        return None
    try:
        result = _get_indicators(df, threshold=1)
        if result is None or len(result) == 0:
            return None
        return result.iloc[-1]
    except Exception as e:
        logger.debug(f"[InStock] TA-Lib 指标计算失败 {code}：{e}")
        return None


def calc_and_write_indicators(raw_data: dict, date_str: str) -> Dict[str, pd.Series]:
    """
    批量计算所有股票的 TA-Lib 技术指标，写入 cn_stock_indicators，
    返回 {code: Series} 供 factor_engine 直接使用（避免重算）。
    """
    if not _TALIB_AVAILABLE:
        return {}

    klines: dict = raw_data.get("klines", {})
    stock_list: pd.DataFrame = raw_data.get("stock_list", pd.DataFrame())

    indicators: Dict[str, pd.Series] = {}
    rows = []

    total = len(klines)
    done = 0
    for code, kline_df in klines.items():
        ind = calc_talib_indicators(code, kline_df)
        if ind is not None:
            indicators[code] = ind
            row = {"date": date_str, "code": code}
            # 尝试获取股票名称
            if not stock_list.empty and "股票代码" in stock_list.columns:
                name_rows = stock_list[stock_list["股票代码"] == code]
                if len(name_rows) > 0 and "证券名称" in stock_list.columns:
                    row["name"] = name_rows.iloc[0]["证券名称"]
            row.update({k: float(v) if isinstance(v, (int, float, np.floating, np.integer))
                        and not (isinstance(v, float) and np.isnan(v)) else 0.0
                        for k, v in ind.items()
                        if k not in ("date", "code", "name") and not isinstance(v, str)})
            rows.append(row)
        done += 1
        if done % 500 == 0:
            logger.info(f"[InStock] 指标计算进度：{done}/{total}")

    if rows:
        _write_indicators_to_db(rows, date_str)

    logger.info(f"[InStock] TA-Lib 指标计算完成：{len(indicators)}/{total} 只")
    return indicators


def _write_indicators_to_db(rows: list, date_str: str):
    """将指标行写入 cn_stock_indicators 表（先删当天数据，再插入）"""
    eng = _get_engine()
    if eng is None or not rows:
        return
    try:
        df = pd.DataFrame(rows)
        # 先清除同日数据
        with eng.connect() as conn:
            from sqlalchemy import text
            conn.execute(text(f"DELETE FROM cn_stock_indicators WHERE date=:d"), {"d": date_str})
            conn.commit()
        df.to_sql("cn_stock_indicators", con=eng, if_exists="append", index=False,
                  chunksize=500, method="multi")
        logger.info(f"[InStock] cn_stock_indicators 写入 {len(df)} 行（{date_str}）")
    except Exception as e:
        logger.warning(f"[InStock] cn_stock_indicators 写入失败：{e}，尝试自动建表后重试")
        try:
            # 首次建表
            df.to_sql("cn_stock_indicators", con=eng, if_exists="replace", index=False,
                      chunksize=500, method="multi")
            logger.info(f"[InStock] cn_stock_indicators 建表并写入 {len(df)} 行")
        except Exception as e2:
            logger.error(f"[InStock] cn_stock_indicators 建表失败：{e2}")


# ============================================================
# 写入 cn_stock_spot（每日行情）
# ============================================================

def write_spot(raw_data: dict, date_str: str):
    """
    从 klines 最后一行 + financials 拼出 cn_stock_spot 格式，写入 MySQL。
    """
    eng = _get_engine()
    if eng is None:
        return

    klines: dict = raw_data.get("klines", {})
    financials: dict = raw_data.get("financials", {})
    stock_list: pd.DataFrame = raw_data.get("stock_list", pd.DataFrame())

    rows = []
    for code, kline_df in klines.items():
        if kline_df is None or len(kline_df) < 2:
            continue
        try:
            last = kline_df.iloc[-1]
            prev = kline_df.iloc[-2]

            def _f(row, key, default=None):
                val = row.get(key, default)
                if val is None:
                    return default
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default

            close = _f(last, "收盘")
            if close is None or close <= 0:
                continue

            row = {
                "date": date_str,
                "code": code,
                "new_price": close,
                "change_rate": _f(last, "涨跌幅", 0.0),
                "ups_downs": _f(last, "涨跌额", 0.0),
                "volume": _f(last, "成交量", 0),
                "deal_amount": _f(last, "成交额", 0),
                "amplitude": _f(last, "振幅", 0.0),
                "turnoverrate": _f(last, "换手率", 0.0),
                "open_price": _f(last, "开盘", close),
                "high_price": _f(last, "最高", close),
                "low_price": _f(last, "最低", close),
                "pre_close_price": _f(prev, "收盘", close),
            }

            # 股票名称
            if not stock_list.empty and "股票代码" in stock_list.columns:
                name_rows = stock_list[stock_list["股票代码"] == code]
                if len(name_rows) > 0 and "证券名称" in stock_list.columns:
                    row["name"] = str(name_rows.iloc[0]["证券名称"])

            # 近期涨幅
            if len(kline_df) >= 60:
                close_vals = pd.to_numeric(kline_df["收盘"], errors="coerce").values
                row["speed_increase_60"] = round((close_vals[-1] / close_vals[-61] - 1) * 100, 2) \
                    if len(close_vals) >= 61 and close_vals[-61] > 0 else 0.0

            # 财务数据（从 financials 提取）
            fin = financials.get(code) if financials else None
            if fin and isinstance(fin, dict):
                income = fin.get("income") or fin.get("profit")
                balance = fin.get("balance")

                def _get_fin(df, col_hint):
                    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                        return None
                    for col in df.columns:
                        if col_hint.lower() in col.lower():
                            val = df.iloc[0][col]
                            try:
                                return float(val)
                            except (ValueError, TypeError):
                                return None
                    return None

                if income is not None:
                    row["total_operate_income"] = _get_fin(income, "营业总收入") or \
                                                   _get_fin(income, "营业收入")
                    row["netprofit_yoy_ratio"] = _get_fin(income, "净利润增长率")
                    row["toi_yoy_ratio"] = _get_fin(income, "营收增长率")
                    row["parent_netprofit"] = _get_fin(income, "归母净利润") or \
                                               _get_fin(income, "归属净利润")
                    row["sale_gpr"] = _get_fin(income, "毛利率")
                    row["basic_eps"] = _get_fin(income, "每股收益")

                if balance is not None:
                    row["debt_asset_ratio"] = _get_fin(balance, "资产负债率")
                    row["bvps"] = _get_fin(balance, "每股净资产")
                    row["per_capital_reserve"] = _get_fin(balance, "每股资本公积") or \
                                                  _get_fin(balance, "每股公积金")
                    row["per_unassign_profit"] = _get_fin(balance, "每股未分配利润")
                    row["total_shares"] = _get_fin(balance, "总股本")
                    row["free_shares"] = _get_fin(balance, "流通股本")

            rows.append(row)
        except Exception as e:
            logger.debug(f"[InStock] write_spot 组装行失败 {code}：{e}")

    if not rows:
        logger.warning("[InStock] write_spot：无有效数据，跳过写入")
        return

    df = pd.DataFrame(rows)
    try:
        with eng.connect() as conn:
            from sqlalchemy import text
            conn.execute(text(f"DELETE FROM cn_stock_spot WHERE date=:d"), {"d": date_str})
            conn.commit()
        df.to_sql("cn_stock_spot", con=eng, if_exists="append", index=False,
                  chunksize=500, method="multi")
        logger.info(f"[InStock] cn_stock_spot 写入 {len(df)} 行（{date_str}）")
    except Exception as e:
        logger.warning(f"[InStock] cn_stock_spot 写入失败：{e}，尝试自动建表")
        try:
            df.to_sql("cn_stock_spot", con=eng, if_exists="replace", index=False,
                      chunksize=500, method="multi")
            logger.info(f"[InStock] cn_stock_spot 建表并写入 {len(df)} 行")
        except Exception as e2:
            logger.error(f"[InStock] cn_stock_spot 写入最终失败：{e2}")


# ============================================================
# 写入 cn_stock_fund_flow（资金流向）
# ============================================================

def write_fund_flow(raw_data: dict, date_str: str):
    """
    将 westock 资金流向数据写入 cn_stock_fund_flow。
    映射字段：主力净流入→fund_amount，超大单→fund_amount_super，以此类推。
    """
    eng = _get_engine()
    if eng is None:
        return

    fund_flows: dict = raw_data.get("fund_flows", {})
    stock_list: pd.DataFrame = raw_data.get("stock_list", pd.DataFrame())

    rows = []
    for code, ff_df in fund_flows.items():
        if ff_df is None or not isinstance(ff_df, pd.DataFrame) or ff_df.empty:
            continue
        try:
            latest = ff_df.iloc[-1] if "日期" not in ff_df.columns else \
                ff_df.sort_values("日期").iloc[-1]

            def _fv(key, default=None):
                val = latest.get(key, default)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return default
                try:
                    return float(str(val).replace(",", "").replace("亿", "e8").replace("万", "e4"))
                except (ValueError, TypeError):
                    return default

            row = {
                "date": date_str,
                "code": code,
                "new_price": _fv("收盘价", 0.0),
                "change_rate": _fv("涨跌幅", 0.0),
                "fund_amount": _fv("主力净流入", 0),
                "fund_amount_super": _fv("超大单净流入", 0),
                "fund_amount_large": _fv("大单净流入", 0),
                "fund_amount_medium": _fv("中单净流入", 0),
                "fund_amount_small": _fv("小单净流入", 0),
            }

            # 股票名称
            if not stock_list.empty and "股票代码" in stock_list.columns:
                name_rows = stock_list[stock_list["股票代码"] == code]
                if len(name_rows) > 0 and "证券名称" in stock_list.columns:
                    row["name"] = str(name_rows.iloc[0]["证券名称"])

            rows.append(row)
        except Exception as e:
            logger.debug(f"[InStock] write_fund_flow 组装行失败 {code}：{e}")

    if not rows:
        logger.warning("[InStock] write_fund_flow：无有效数据，跳过写入")
        return

    df = pd.DataFrame(rows)
    try:
        with eng.connect() as conn:
            from sqlalchemy import text
            conn.execute(text(f"DELETE FROM cn_stock_fund_flow WHERE date=:d"), {"d": date_str})
            conn.commit()
        df.to_sql("cn_stock_fund_flow", con=eng, if_exists="append", index=False,
                  chunksize=500, method="multi")
        logger.info(f"[InStock] cn_stock_fund_flow 写入 {len(df)} 行（{date_str}）")
    except Exception as e:
        logger.warning(f"[InStock] cn_stock_fund_flow 写入失败：{e}，尝试自动建表")
        try:
            df.to_sql("cn_stock_fund_flow", con=eng, if_exists="replace", index=False,
                      chunksize=500, method="multi")
            logger.info(f"[InStock] cn_stock_fund_flow 建表并写入 {len(df)} 行")
        except Exception as e2:
            logger.error(f"[InStock] cn_stock_fund_flow 写入最终失败：{e2}")


# ============================================================
# 便捷入口：一键写入所有表
# ============================================================

def write_all(raw_data: dict, date_str: str) -> Dict[str, pd.Series]:
    """
    一键执行：计算 TA-Lib 指标、写入三张表。
    返回 indicators dict 供 factor_engine 使用。

    Returns:
        indicators: {code: Series}，TA-Lib 不可用时返回 {}
    """
    if not check_connection():
        logger.warning("[InStock] MySQL 不可用，跳过所有 DB 写入")
        return {}

    logger.info(f"[InStock] 开始写入 MySQL（{date_str}）")

    indicators = calc_and_write_indicators(raw_data, date_str)
    write_spot(raw_data, date_str)
    write_fund_flow(raw_data, date_str)

    logger.info(f"[InStock] MySQL 写入完成（{date_str}）")
    return indicators
