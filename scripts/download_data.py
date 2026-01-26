#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据下载脚本
下载A股历史数据
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from loguru import logger

from data import DataDownloader, DataProcessor, QlibConverter


def setup_logging(log_dir: Path):
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "download_{time}.log",
        rotation="10 MB",
        retention="30 days",
        level="INFO"
    )


def main():
    parser = argparse.ArgumentParser(description="下载A股数据")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "update"],
        default="full",
        help="下载模式: full-全量下载, update-增量更新"
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="跳过数据处理步骤"
    )
    parser.add_argument(
        "--skip-qlib",
        action="store_true",
        help="跳过Qlib格式转换"
    )

    args = parser.parse_args()

    # 加载配置
    config_path = project_root / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 设置日志
    log_dir = Path(config.get("paths", {}).get("log_dir", "./logs"))
    setup_logging(log_dir)

    logger.info("=" * 50)
    logger.info("A股数据下载脚本")
    logger.info("=" * 50)

    try:
        # 1. 下载数据
        logger.info("Step 1: Downloading data...")
        downloader = DataDownloader(config)

        if args.mode == "full":
            data = downloader.download_all()
        else:
            data = downloader.update_data()

        logger.info("Data download completed!")

        # 2. 处理数据
        if not args.skip_process:
            logger.info("Step 2: Processing data...")
            processor = DataProcessor(config)
            processed_data = processor.process()
            logger.info("Data processing completed!")
        else:
            logger.info("Step 2: Skipped data processing")

        # 3. 转换为Qlib格式
        if not args.skip_qlib:
            logger.info("Step 3: Converting to Qlib format...")
            converter = QlibConverter(config)
            converter.convert()
            logger.info("Qlib conversion completed!")
        else:
            logger.info("Step 3: Skipped Qlib conversion")

        logger.info("=" * 50)
        logger.info("All tasks completed successfully!")
        logger.info("=" * 50)

    except Exception as e:
        logger.exception(f"Error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
