#!/usr/bin/env python3
"""
知识库上传模块 — uploader.py
=============================
将生成的 factor_index 和 stock_data 文件上传到知识库。

通过 IMA OpenAPI 实现:
- 查询 markdown 文件夹 ID
- 上传文件到指定文件夹
- 支持覆盖更新
"""
import os
import json
import subprocess
import requests
from typing import Optional, Tuple

from config import (
    KB_ID, KB_NAME, KB_API_BASE, MARKDOWN_FOLDER_KEYWORD, logger
)

# ============================================================
# IMA API 凭证（从环境变量获取）
# ============================================================
CLIENT_ID = os.environ.get("IMA_OPENAPI_CLIENTID", "")
API_KEY = os.environ.get("IMA_OPENAPI_APIKEY", "")


def _api_post(endpoint: str, body: dict) -> dict:
    """调用 IMA OpenAPI"""
    url = f"{KB_API_BASE}/{endpoint}"
    headers = {
        "ima-openapi-clientid": CLIENT_ID,
        "ima-openapi-apikey": API_KEY,
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_markdown_folder_id() -> Optional[str]:
    """
    动态获取 markdown 文件夹 ID
    规则：使用 search_knowledge 接口，关键词 "markdown" + media_type=99
    """
    if not CLIENT_ID or not API_KEY:
        logger.warning("[上传] 未配置 API 凭证，跳过文件夹查询")
        return None

    try:
        result = _api_post("openapi/wiki/v1/search_knowledge", {
            "query": MARKDOWN_FOLDER_KEYWORD,
            "knowledge_base_id": KB_ID,
        })
        info_list = result.get("data", {}).get("info_list", [])
        for item in info_list:
            if item.get("media_type") == 99:  # 文件夹
                folder_id = item.get("media_id")
                logger.info(f"[上传] markdown 文件夹 ID: {folder_id}")
                return folder_id
        logger.warning("[上传] 未找到 markdown 文件夹")
        return None
    except Exception as e:
        logger.warning(f"[上传] 查询文件夹失败: {e}")
        return None


def upload_file_to_kb(file_path: str, folder_id: Optional[str] = None) -> bool:
    """
    上传文件到知识库

    Args:
        file_path: 本地文件路径
        folder_id: 目标文件夹 ID（可选）

    Returns:
        是否成功
    """
    if not CLIENT_ID or not API_KEY:
        logger.warning("[上传] 未配置 API 凭证，跳过上传")
        logger.info(f"  文件位置: {file_path}")
        return False

    try:
        # 使用平台命令上传到 COS，获取 cosKey
        result = subprocess.run(
            ["ima_cos_util", "-f", file_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            logger.error(f"[上传] COS 上传失败: {result.stderr}")
            return False

        cos_key = result.stdout.strip()
        logger.info(f"[上传] COS Key: {cos_key}")

        # 调用知识库 add_knowledge API
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()
        media_type_map = {".md": 7, ".json": 13, ".gz": 13, ".txt": 13}
        media_type = media_type_map.get(ext, 13)

        body = {
            "media_type": media_type,
            "media_id": cos_key,
            "title": filename,
            "knowledge_base_id": KB_ID,
        }
        if folder_id:
            body["parent_folder_id"] = folder_id

        result = _api_post("openapi/wiki/v1/add_knowledge", body)
        logger.info(f"[上传] ✅ {filename} → 知识库")
        return True

    except Exception as e:
        logger.error(f"[上传] 失败: {e}")
        return False


def upload_data_files(field_index_path: str, stock_data_path: str) -> Tuple[bool, bool]:
    """
    上传 field_index 和 stock_data 到知识库

    返回: (field_index成功, stock_data成功)
    """
    logger.info("[上传] 开始上传数据文件到知识库...")

    folder_id = get_markdown_folder_id()

    fi_ok = upload_file_to_kb(field_index_path, folder_id)
    sd_ok = upload_file_to_kb(stock_data_path, folder_id)

    if fi_ok and sd_ok:
        logger.info("[上传] ✅ 全部上传成功")
    else:
        logger.warning("[上传] ⚠️ 部分上传失败")

    return fi_ok, sd_ok
