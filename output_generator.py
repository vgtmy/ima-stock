#!/usr/bin/env python3
"""
输出生成器 — output_generator.py
=================================
生成标准格式的:
- field_index_{date}.json  (字段索引)
- stock_data_{date}.json.gz (全量股票因子数据)
"""
import json
import gzip
import os
from datetime import datetime
from typing import Dict

from config import OUTPUT_DIR, generate_field_index, logger


def generate_output(all_factors: Dict[str, dict], target_date: str) -> tuple:
    """
    生成标准输出文件

    Args:
        all_factors: {股票代码: {因子名: 因子值}}
        target_date: YYYY-MM-DD

    Returns:
        (field_index_path, stock_data_path)
    """
    date_tag = target_date.replace("-", "")
    logger.info(f"[输出] 生成 {date_tag} 数据文件...")

    # --- 1. 生成 field_index.json ---
    field_list = generate_field_index()

    # 建立因子名→ID的映射
    name_to_id = {f["name"]: f["id"] for f in field_list}

    field_index = {
        "metadata": {
            "data_date": target_date,
            "field_count": len(field_list),
            "stock_count": len(all_factors),
            "generated_at": datetime.now().isoformat(),
        },
        "fields": field_list,
        "layer_info": {
            "layers": _build_layer_info(field_list)
        }
    }

    field_index_path = os.path.join(OUTPUT_DIR, f"field_index_{target_date}.json")
    with open(field_index_path, "w", encoding="utf-8") as f:
        json.dump(field_index, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✅ field_index: {field_index_path} ({len(field_list)} 字段)")

    # --- 2. 生成 stock_data.json.gz ---
    stock_data = []
    for code, factors in all_factors.items():
        # 按 field_index 中的 ID 映射因子值
        f_list = []
        for name, value in factors.items():
            fid = name_to_id.get(name)
            if fid and value is not None and not (isinstance(value, float) and (value != value)):  # exclude NaN
                # 统一格式: 数值 → 字符串
                if isinstance(value, bool):
                    val_str = "1" if value else "0"
                elif isinstance(value, float):
                    val_str = f"{value:.6g}"
                else:
                    val_str = str(value)
                f_list.append([fid, val_str])

        if f_list:
            stock_data.append({
                "c": code,
                "n": factors.get("证券名称", ""),
                "f": f_list,
            })

    stock_data_path = os.path.join(OUTPUT_DIR, f"stock_data_{target_date}.json.gz")
    with gzip.open(stock_data_path, "wt", encoding="utf-8") as f:
        json.dump(stock_data, f, ensure_ascii=False)
    logger.info(f"  ✅ stock_data: {stock_data_path} ({len(stock_data)} 只股票)")

    return field_index_path, stock_data_path


def _build_layer_info(field_list: list) -> dict:
    """构建层级信息"""
    layers = {}
    for f in field_list:
        l1 = f["layer1"]
        l2 = f["layer2"]
        if l1 not in layers:
            layers[l1] = {}
        if l2 not in layers[l1]:
            layers[l1][l2] = 0
        layers[l1][l2] += 1

    # 转换格式: {layer1: [layer2_list]}
    result = {}
    for l1, l2_dict in layers.items():
        result[l1] = list(l2_dict.keys())
    return result


def generate_markdown_for_stock(code: str, factors: dict, field_list: list) -> str:
    """
    根据因子数据生成单只股票的Markdown文件（对齐现有格式）
    """
    lines = []
    lines.append(f"# 股票数据_{factors.get('证券名称', code)}_{code}")
    lines.append("")
    lines.append(f"> 数据日期: {factors.get('_data_date', '')}")
    lines.append("> 本数据由 ETL 管道自动生成，基于 akshare 数据源")
    lines.append("")

    # 按层级组织
    current_l1 = None
    current_l2 = None
    for f in field_list:
        name = f["name"]
        l1 = f["layer1"]
        l2 = f["layer2"]
        value = factors.get(name)

        if value is None:
            continue

        if l1 != current_l1:
            lines.append(f"## {l1}")
            current_l1 = l1
            current_l2 = None

        if l2 != current_l2:
            lines.append(f"### {l2}")
            current_l2 = l2

        lines.append(f"- **{name}**: {value}")

    return "\n".join(lines)
