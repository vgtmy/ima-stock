#!/usr/bin/env python3
"""
ETL 主调度器 — main.py
======================
整合全部模块:
1. 数据采集 (data_fetcher)
1.5 InStock MySQL 写入 + TA-Lib 指标计算 (instock_bridge)
2. 因子计算 (factor_engine)
3. 输出生成 (output_generator)
4. 知识库上传 (uploader)

用法:
  python main.py                  # 全量执行（最新交易日）
  python main.py 2026-06-06       # 指定日期执行
  python main.py --no-upload      # 仅本地生成，不上传
  python main.py --no-db          # 不写入 MySQL（跳过 InStock 整合）
"""
import sys
import time
from datetime import datetime

from config import logger, BATCH_SIZE, WRITE_TO_DB
from data_fetcher import fetch_all_data
from factor_engine import FactorEngine
from output_generator import generate_output
from uploader import upload_data_files


def run_etl_pipeline(target_date: str = None, upload: bool = True,
                     force_refetch: bool = False, write_db: bool = None):
    """
    执行完整 ETL 管道

    Args:
        target_date: YYYY-MM-DD，默认今天
        upload: 是否上传到知识库
        force_refetch: 强制重新拉取数据（忽略已有快照）
        write_db: 是否写入 MySQL（None 则读 config.WRITE_TO_DB）
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")
    if write_db is None:
        write_db = WRITE_TO_DB

    t0 = time.time()

    # =============================================
    # Phase 1: 数据采集
    # =============================================
    logger.info("=" * 60)
    logger.info(f"Phase 1: 数据采集 ({target_date})")
    logger.info("=" * 60)
    t1 = time.time()
    raw_data = fetch_all_data(target_date=target_date, use_cache=not force_refetch)
    t1_elapsed = time.time() - t1
    logger.info(f"  耗时: {t1_elapsed:.0f}s")

    # =============================================
    # Phase 1.5: InStock MySQL 写入 + TA-Lib 指标计算
    # =============================================
    if write_db:
        logger.info("=" * 60)
        logger.info(f"Phase 1.5: InStock MySQL 写入 + TA-Lib 指标计算")
        logger.info("=" * 60)
        t15 = time.time()
        try:
            import instock_bridge
            indicators = instock_bridge.write_all(raw_data, target_date)
            if indicators:
                raw_data["indicators"] = indicators
                logger.info(f"  TA-Lib 指标已注入 raw_data ({len(indicators)} 只)")
        except Exception as e:
            logger.warning(f"  InStock 整合失败（不影响 ETL 主流程）：{e}")
        t15_elapsed = time.time() - t15
        logger.info(f"  耗时: {t15_elapsed:.0f}s")
    else:
        logger.info("Phase 1.5: 跳过 MySQL 写入 (--no-db)")

    # =============================================
    # Phase 2: 因子计算
    # =============================================
    logger.info("=" * 60)
    logger.info(f"Phase 2: 因子计算 ({len(raw_data['stock_list'])} 只股票)")
    logger.info("=" * 60)
    t2 = time.time()
    engine = FactorEngine(raw_data)
    all_factors = {}
    codes = raw_data["stock_list"]["股票代码"].tolist()

    for i, code in enumerate(codes):
        try:
            factors = engine.calc_all_factors(code)
            if factors:
                all_factors[code] = factors
        except Exception as e:
            logger.debug(f"  {code} 因子计算失败: {e}")

        if (i + 1) % BATCH_SIZE == 0:
            logger.info(f"  进度: {i+1}/{len(codes)} ({len(all_factors)} 只完成)")

    t2_elapsed = time.time() - t2
    logger.info(f"  耗时: {t2_elapsed:.0f}s | 成功: {len(all_factors)}/{len(codes)}")

    # =============================================
    # Phase 3: 输出生成
    # =============================================
    logger.info("=" * 60)
    logger.info("Phase 3: 输出生成")
    logger.info("=" * 60)
    t3 = time.time()
    fi_path, sd_path = generate_output(all_factors, target_date)
    t3_elapsed = time.time() - t3
    logger.info(f"  耗时: {t3_elapsed:.0f}s")

    # =============================================
    # Phase 4: 知识库上传（可选）
    # =============================================
    if upload:
        logger.info("=" * 60)
        logger.info("Phase 4: 知识库上传")
        logger.info("=" * 60)
        t4 = time.time()
        fi_ok, sd_ok = upload_data_files(fi_path, sd_path)
        t4_elapsed = time.time() - t4
        logger.info(f"  耗时: {t4_elapsed:.0f}s")
        logger.info(f"  field_index: {'OK' if fi_ok else 'FAIL'}")
        logger.info(f"  stock_data:  {'OK' if sd_ok else 'FAIL'}")
    else:
        logger.info("=" * 60)
        logger.info("Phase 4: 跳过上传 (--no-upload)")
        logger.info("=" * 60)

    # =============================================
    # 总结
    # =============================================
    total_elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"ETL 管道执行完毕")
    logger.info(f"  股票数: {len(all_factors)}")
    logger.info(f"  总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    logger.info(f"  输出目录: {fi_path}")
    logger.info("=" * 60)

    return {
        "stock_count": len(all_factors),
        "elapsed": total_elapsed,
        "field_index_path": fi_path,
        "stock_data_path": sd_path,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A股2200+因子ETL管道")
    parser.add_argument("date", nargs="?", default=None, help="目标日期 YYYY-MM-DD")
    parser.add_argument("--no-upload", action="store_true", help="仅本地生成，不上传知识库")
    parser.add_argument("--force-refetch", action="store_true", help="强制重新拉取数据（忽略已有快照）")
    parser.add_argument("--no-db", action="store_true", help="跳过 MySQL 写入（不使用 InStock 整合）")
    args = parser.parse_args()

    run_etl_pipeline(
        target_date=args.date,
        upload=not args.no_upload,
        force_refetch=args.force_refetch,
        write_db=not args.no_db,
    )
