#!/usr/bin/env python3
"""
定时调度模块 — scheduler.py
===========================
支持两种运行模式:
1. 定时模式: 每个交易日 15:30 自动执行
2. 单次模式: 手动运行一次

用法:
  python scheduler.py              # 定时模式（持续运行）
  python scheduler.py --once       # 单次执行
  python scheduler.py --once 2026-06-06  # 指定日期执行
"""
import sys
import time
import schedule
from datetime import datetime, date

from config import SCHEDULE_TIME, logger
from main import run_etl_pipeline


def is_trading_day(d: date) -> bool:
    """简单判断是否为交易日（排除周末）"""
    # 完整的交易日历判断需要额外数据，这里先排除周末
    return d.weekday() < 5


def job():
    """单次 ETL 执行任务"""
    today = datetime.now()
    if not is_trading_day(today.date()):
        logger.info(f"[调度] {today:%Y-%m-%d} 非交易日，跳过")
        return

    logger.info(f"[调度] {today:%Y-%m-%d} 开始执行 ETL...")
    try:
        run_etl_pipeline(target_date=today.strftime("%Y-%m-%d"))
        logger.info("[调度] ✅ ETL 执行完成")
    except Exception as e:
        logger.error(f"[调度] ❌ ETL 执行失败: {e}", exc_info=True)


def run_once(target_date: str = None):
    """单次运行"""
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"[单次] 目标日期: {target_date}")
    run_etl_pipeline(target_date=target_date)
    logger.info("[单次] ✅ 完成")


def run_scheduled():
    """定时调度模式"""
    # 注册任务：每个交易日 SCHEDULE_TIME 执行
    schedule.every().day.at(SCHEDULE_TIME).do(job)

    logger.info(f"[调度] 定时任务已注册: 每日 {SCHEDULE_TIME}")
    logger.info("[调度] 等待执行... (按 Ctrl+C 停止)")

    while True:
        schedule.run_pending()
        time.sleep(60)  # 每分钟检查一次


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        target = sys.argv[2] if len(sys.argv) > 2 else None
        run_once(target)
    else:
        run_scheduled()
