"""
病理报告特征提取与编码模块
将JSON格式的病理报告转换为向量表示
支持多部分分别编码并拼接
支持多种Ollama嵌入模型
所有键名和值均转换为小写
可以直接在Python脚本中执行
"""

import json
import glob
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import logging
import time
import argparse

import pandas as pd
import numpy as np
import ollama
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """配置类"""
    # Ollama配置
    model_name: str = 'qwen3-embedding:latest'

    # 模型维度映射
    MODEL_DIMENSIONS: Dict[str, int] = field(default_factory=lambda: {
        'qwen3-embedding:latest': 4096,
        'embeddinggemma': 768,
        'qwen3-embedding:4b': 2560,
        'nomic-embed-text-v2-moe': 768,
        'qwen3-embedding:0.6b': 1024
    })

    # 并行处理配置
    max_workers: int = 36
    batch_size: int = 20

    # 病理报告部分数
    num_sections: int = 4

    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0

    # 文件路径
    input_dir: Optional[str] = None
    output_csv: Optional[str] = None
    flattened_csv: Optional[str] = None

    # 模型基础名称（用于文件路径）
    model_base: Optional[str] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.model_base is None:
            # 从model_name提取基础名称
            if ':' in self.model_name:
                self.model_base = self.model_name.split(':')[0]
            else:
                self.model_base = self.model_name

    @property
    def single_embedding_dim(self) -> int:
        """获取单个嵌入向量的维度"""
        return self.MODEL_DIMENSIONS.get(self.model_name, 4096)

    @property
    def total_embedding_dim(self) -> int:
        """总嵌入向量维度"""
        return self.single_embedding_dim * self.num_sections


class SectionProcessor:
    """病理报告部分处理器"""

    SECTION_NAMES = [
        'glomerular_lesions',
        'tubulointerstitial_lesions',
        'vascular_lesions',
        'immunofluorescence'
    ]

    @staticmethod
    def to_lowercase(value: Any) -> Any:
        """将值转换为小写（如果是字符串）"""
        if isinstance(value, str):
            return value.lower()
        return value

    @staticmethod
    def flatten_dict(data: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """递归扁平化嵌套字典，所有键和字符串值都转换为小写"""
        items = {}
        for key, value in data.items():
            # 将当前键转换为小写
            lowercase_key = key.lower() if isinstance(key, str) else key
            new_key = f"{parent_key}{sep}{lowercase_key}" if parent_key else lowercase_key

            if isinstance(value, dict):
                # 递归处理嵌套字典
                items.update(SectionProcessor.flatten_dict(value, new_key, sep))
            elif isinstance(value, list):
                # 处理列表
                if value and isinstance(value[0], dict):
                    # 如果是字典列表，合并处理
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            items.update(
                                SectionProcessor.flatten_dict(
                                    item,
                                    f"{new_key}_{i}",
                                    sep
                                )
                            )
                else:
                    # 简单列表转换为字符串，并将所有字符串元素转换为小写
                    if value:
                        # 处理列表中的每个元素
                        processed_values = []
                        for item in value:
                            if isinstance(item, str):
                                processed_values.append(item.lower())
                            else:
                                processed_values.append(str(item))
                        items[new_key] = ', '.join(processed_values)
                    else:
                        items[new_key] = ''
            else:
                # 将字符串值转换为小写
                items[new_key] = SectionProcessor.to_lowercase(value)
        return items

    @staticmethod
    def extract_section_text(data: Dict, section_name: str) -> str:
        """提取指定部分的文本内容"""
        if section_name not in data:
            return ""

        section_data = data[section_name]
        flat_data = SectionProcessor.flatten_dict(section_data)

        # 转换为键值对字符串
        section_text = '; '.join(
            f"{k}: {v}" for k, v in flat_data.items()
            if v not in [None, '', []]  # 过滤空值
        )

        return section_text

    @staticmethod
    def extract_all_section_texts(data: Dict) -> Dict[str, str]:
        """提取所有部分的文本内容"""
        result = {}
        for section in SectionProcessor.SECTION_NAMES:
            result[section] = SectionProcessor.extract_section_text(data, section)
        return result


class EmbeddingGenerator:
    """嵌入向量生成器"""

    def __init__(self, config: ReportConfig):
        self.config = config
        self.model_name = config.model_name

    def generate_embedding_with_retry(self, text: str, section_name: str = "") -> Optional[List[float]]:
        """带重试机制的嵌入生成"""
        if not text:
            # 返回零向量
            return [0.0] * self.config.single_embedding_dim

        for attempt in range(self.config.max_retries):
            try:
                response = ollama.embed(
                    model=self.model_name,
                    input=text
                )
                embeddings = response.get("embeddings", [])
                if embeddings and len(embeddings) > 0:
                    embedding_vector = embeddings[0]

                    # 验证维度
                    expected_dim = self.config.single_embedding_dim
                    actual_dim = len(embedding_vector)

                    if actual_dim != expected_dim:
                        logger.warning(
                            f"嵌入向量维度异常: 期望{expected_dim}, "
                            f"实际{actual_dim}"
                        )
                        # 调整维度
                        if actual_dim > expected_dim:
                            embedding_vector = embedding_vector[:expected_dim]
                        else:
                            embedding_vector.extend(
                                [0.0] * (expected_dim - actual_dim)
                            )

                    return embedding_vector
                else:
                    logger.error(f"嵌入生成返回空结果: {text[:100]}...")
                    return None

            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"嵌入生成失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}"
                        f"，等待 {wait_time} 秒后重试..."
                    )
                    time.sleep(wait_time)
                else:
                    section_info = f" for section '{section_name}'" if section_name else ""
                    logger.error(f"嵌入生成失败，已达到最大重试次数{section_info}: {e}")
                    return None

        return None

    def generate_section_embeddings(self, section_texts: Dict[str, str]) -> Dict[str, List[float]]:
        """为每个部分生成嵌入向量"""
        embeddings = {}

        for section_name, text in section_texts.items():
            embedding = self.generate_embedding_with_retry(text, section_name)
            if embedding is not None:
                embeddings[section_name] = embedding
            else:
                # 如果生成失败，使用零向量
                embeddings[section_name] = [0.0] * self.config.single_embedding_dim
                logger.warning(f"部分 '{section_name}' 使用零向量代替")

        return embeddings

    def concatenate_embeddings(self, section_embeddings: Dict[str, List[float]]) -> List[float]:
        """拼接所有部分的嵌入向量"""
        concatenated = []

        # 按照固定的顺序拼接
        for section_name in SectionProcessor.SECTION_NAMES:
            if section_name in section_embeddings:
                concatenated.extend(section_embeddings[section_name])
            else:
                # 如果缺少某个部分，用零向量填充
                concatenated.extend([0.0] * self.config.single_embedding_dim)
                logger.warning(f"缺少部分 '{section_name}'，使用零向量填充")

        # 验证总维度
        expected_dim = self.config.total_embedding_dim
        actual_dim = len(concatenated)

        if actual_dim != expected_dim:
            logger.error(f"拼接后向量维度错误: 期望{expected_dim}, 实际{actual_dim}")
            # 调整维度
            if actual_dim > expected_dim:
                concatenated = concatenated[:expected_dim]
            else:
                concatenated.extend([0.0] * (expected_dim - actual_dim))

        return concatenated


class FileIDExtractor:
    """文件ID提取器"""

    @staticmethod
    def extract_ids_from_filename(filename: str) -> Tuple[str, str]:
        """从文件名中提取phID和bio_id"""
        try:
            basename = os.path.basename(filename)
            # 移除扩展名
            name_without_ext = os.path.splitext(basename)[0]

            # 移除前缀
            name_without_prefix = name_without_ext.replace('pathology_features_', '')

            # 分割ID
            parts = name_without_prefix.split('_')

            if len(parts) >= 2:
                phID = parts[0]
                bio_id = parts[1]
            elif len(parts) == 1:
                phID = parts[0]
                bio_id = ""
            else:
                phID = ""
                bio_id = ""

            return phID, bio_id

        except Exception as e:
            logger.error(f"提取文件名ID时出错: {filename} - {e}")
            return "", ""


class ReportProcessor:
    """病理报告处理器"""

    def __init__(self, config: ReportConfig):
        self.config = config
        self.section_processor = SectionProcessor()
        self.embedding_generator = EmbeddingGenerator(config)
        self.id_extractor = FileIDExtractor()
        self._validate_paths()

    def _validate_paths(self):
        """验证文件路径"""
        if self.config.input_dir:
            input_path = Path(self.config.input_dir)
            if not input_path.exists():
                raise FileNotFoundError(f"输入目录不存在: {input_path}")

    def load_json_files(self) -> List[str]:
        """加载所有JSON文件路径"""
        if not self.config.input_dir:
            raise ValueError("未设置输入目录")

        pattern = os.path.join(self.config.input_dir, "*.json")
        files = glob.glob(pattern)

        if not files:
            logger.warning(f"在 {self.config.input_dir} 中未找到JSON文件")
            return []

        logger.info(f"找到 {len(files)} 个JSON文件")
        return files

    def create_flattened_dataframe(self) -> pd.DataFrame:
        """创建扁平化的DataFrame"""
        files = self.load_json_files()
        all_reports = []

        for file in tqdm(files, desc="扁平化JSON文件"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 扁平化整个数据结构（自动转换为小写）
                flattened_data = self.section_processor.flatten_dict(data)

                # 提取ID
                phID, bio_id = self.id_extractor.extract_ids_from_filename(file)

                # 添加元数据（ID保持不变，不转换为小写）
                flattened_data['phID'] = phID
                flattened_data['bio_id'] = bio_id
                flattened_data['source_file'] = Path(file).name

                all_reports.append(flattened_data)

            except Exception as e:
                logger.error(f"处理文件 {file} 时出错: {e}")
                continue

        if not all_reports:
            return pd.DataFrame()

        df = pd.DataFrame(all_reports)

        # 保存到CSV
        if self.config.flattened_csv:
            df.to_csv(self.config.flattened_csv, index=False, encoding='utf-8')
            logger.info(f"扁平化数据已保存到: {self.config.flattened_csv}")

        return df

    def process_single_file(self, file_path: str) -> Optional[Dict]:
        """处理单个文件，生成拼接的嵌入向量"""
        try:
            # 加载JSON数据
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取ID
            phID, bio_id = self.id_extractor.extract_ids_from_filename(file_path)

            # 提取所有部分的文本（自动转换为小写）
            section_texts = self.section_processor.extract_all_section_texts(data)

            # 为每个部分生成嵌入向量
            section_embeddings = self.embedding_generator.generate_section_embeddings(section_texts)

            # 拼接所有嵌入向量
            concatenated_embedding = self.embedding_generator.concatenate_embeddings(section_embeddings)

            # 创建结果字典
            result = {
                'phID': phID,
                'bio_id': bio_id,
                'source_file': Path(file_path).name,
                'total_embedding_dim': len(concatenated_embedding),
                'timestamp': pd.Timestamp.now()
            }

            # 添加各部分的文本摘要（前100字符）
            for section_name, text in section_texts.items():
                result[f'{section_name}_text_preview'] = text[:100] + "..." if len(text) > 100 else text

            # 添加各部分的嵌入向量维度信息
            for section_name, embedding in section_embeddings.items():
                result[f'{section_name}_embedding_dim'] = len(embedding)

            # 添加拼接后的嵌入向量
            result['concatenated_embedding'] = concatenated_embedding

            return result

        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}")
            return None

    def process_files_parallel(self) -> pd.DataFrame:
        """并行处理所有文件"""
        files = self.load_json_files()
        if not files:
            return pd.DataFrame()

        results = []
        failed_files = []

        logger.info(f"开始并行处理 {len(files)} 个文件，使用 {self.config.max_workers} 个线程...")

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self.process_single_file, file_path): file_path
                for file_path in files
            }

            # 处理完成的任务
            with tqdm(total=len(files), desc="生成嵌入向量") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                        else:
                            failed_files.append(file_path)
                    except Exception as e:
                        logger.error(f"处理文件 {file_path} 时发生异常: {e}")
                        failed_files.append(file_path)
                    finally:
                        pbar.update(1)

        # 统计信息
        success_count = len(results)
        total_count = len(files)
        logger.info(f"处理完成: 成功 {success_count}/{total_count}, 失败 {len(failed_files)}")

        if failed_files:
            logger.warning(f"失败的文件 (前10个): {failed_files[:10]}")

        # 转换为DataFrame
        if not results:
            logger.warning("没有成功处理任何文件")
            return pd.DataFrame()

        # 创建元数据DataFrame
        metadata_list = []
        embedding_list = []

        for result in results:
            # 提取元数据
            metadata = {
                'phID': result['phID'],
                'bio_id': result['bio_id'],
                'source_file': result['source_file'],
                'total_embedding_dim': result['total_embedding_dim'],
                'timestamp': result['timestamp']
            }

            # 添加各部分的文本摘要
            for section_name in SectionProcessor.SECTION_NAMES:
                text_key = f'{section_name}_text_preview'
                if text_key in result:
                    metadata[text_key] = result[text_key]

            metadata_list.append(metadata)

            # 提取拼接的嵌入向量
            embedding_list.append(result['concatenated_embedding'])

        # 创建元数据DataFrame
        metadata_df = pd.DataFrame(metadata_list)

        # 创建嵌入向量DataFrame
        total_dim = self.config.total_embedding_dim
        embedding_columns = [f'emb_{i}' for i in range(total_dim)]
        embedding_df = pd.DataFrame(embedding_list, columns=embedding_columns)

        # 合并数据
        final_df = pd.concat([metadata_df, embedding_df], axis=1)

        # 保存到CSV
        if self.config.output_csv:
            final_df.to_csv(self.config.output_csv, index=False, encoding='utf-8')
            logger.info(f"嵌入数据已保存到: {self.config.output_csv}")

            # 验证保存的数据
            saved_df = pd.read_csv(self.config.output_csv)
            logger.info(f"保存的数据形状: {saved_df.shape}")
            logger.info(f"嵌入向量列数: {len([col for col in saved_df.columns if col.startswith('emb_')])}")

        return final_df


def validate_embeddings(df: pd.DataFrame, expected_dim: int) -> None:
    """验证嵌入向量"""
    if df.empty:
        logger.warning("DataFrame为空，无法验证")
        return

    # 提取嵌入向量列
    embedding_cols = [col for col in df.columns if col.startswith('emb_')]

    logger.info(f"嵌入向量列数: {len(embedding_cols)}")
    logger.info(f"期望维度: {expected_dim}")

    if len(embedding_cols) != expected_dim:
        logger.error(f"维度不匹配: 期望{expected_dim}, 实际{len(embedding_cols)}")

    # 检查缺失值
    missing_values = df[embedding_cols].isnull().sum().sum()
    if missing_values > 0:
        logger.warning(f"嵌入向量中存在 {missing_values} 个缺失值")

    # 检查零向量
    zero_embeddings = (df[embedding_cols] == 0).all(axis=1).sum()
    if zero_embeddings > 0:
        logger.warning(f"发现 {zero_embeddings} 个全零向量")

    # 检查重复
    duplicates = df.duplicated(subset=['phID', 'bio_id']).sum()
    if duplicates > 0:
        logger.warning(f"发现 {duplicates} 个重复的ID")

    # 检查向量范围
    for col in embedding_cols[:5]:  # 只检查前5列
        col_min = df[col].min()
        col_max = df[col].max()
        logger.info(f"列 {col}: 范围 [{col_min:.4f}, {col_max:.4f}]")


def process_reports(
        model_name: str = 'qwen3-embedding:latest',
        input_dir: str = None,
        output_csv: str = None,
        flattened_csv: str = None,
        max_workers: int = 36,
        num_sections: int = 4,
        model_base: str = None
) -> pd.DataFrame:
    """
    处理病理报告的主要函数

    Args:
        model_name: Ollama模型名称
        input_dir: 输入JSON文件目录
        output_csv: 输出嵌入向量CSV文件路径
        flattened_csv: 扁平化数据CSV文件路径（可选）
        max_workers: 最大工作线程数
        num_sections: 病理报告部分数量
        model_base: 模型基础名称（用于文件路径，可选）

    Returns:
        pd.DataFrame: 包含嵌入向量的DataFrame
    """
    # 创建配置
    config = ReportConfig(
        model_name=model_name,
        input_dir=input_dir,
        output_csv=output_csv,
        flattened_csv=flattened_csv,
        max_workers=max_workers,
        num_sections=num_sections,
        model_base=model_base
    )

    logger.info(f"配置信息:")
    logger.info(f"  模型名称: {config.model_name}")
    logger.info(f"  模型基础名称: {config.model_base}")
    logger.info(f"  输入目录: {config.input_dir}")
    logger.info(f"  输出文件: {config.output_csv}")
    logger.info(f"  扁平化文件: {config.flattened_csv}")
    logger.info(f"  线程数: {config.max_workers}")
    logger.info(f"  单嵌入维度: {config.single_embedding_dim}")
    logger.info(f"  部分数量: {config.num_sections}")
    logger.info(f"  总嵌入维度: {config.total_embedding_dim}")
    logger.info(f"  所有键和值均转换为小写")

    # 创建处理器
    processor = ReportProcessor(config)

    try:
        # 第一步：创建扁平化数据（可选）
        if config.flattened_csv:
            logger.info("开始创建扁平化数据...")
            flattened_df = processor.create_flattened_dataframe()
            if not flattened_df.empty:
                logger.info(f"扁平化数据形状: {flattened_df.shape}")
            else:
                logger.warning("未能创建扁平化数据")

        # 第二步：生成拼接的嵌入向量
        logger.info("开始生成拼接的嵌入向量...")
        embedding_df = processor.process_files_parallel()

        if not embedding_df.empty:
            logger.info(f"嵌入数据形状: {embedding_df.shape}")

            # 验证数据
            logger.info("开始验证嵌入向量...")
            validate_embeddings(embedding_df, config.total_embedding_dim)

            # 显示前几行
            logger.info("数据前几行:")
            print(embedding_df[['phID', 'bio_id', 'total_embedding_dim']].head())

            # 显示嵌入向量维度
            embedding_cols = [col for col in embedding_df.columns if col.startswith('emb_')]
            logger.info(f"实际嵌入向量列数: {len(embedding_cols)}")

            return embedding_df
        else:
            logger.error("未能生成嵌入向量数据")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}", exc_info=True)
        return pd.DataFrame()


def create_paths(model_base: str, model_name: str = None) -> tuple:
    """
    创建标准化的文件路径

    Args:
        model_base: 模型基础名称（如qwen, deepseek等）
        model_name: 完整模型名称（可选）

    Returns:
        tuple: (input_dir, output_csv, flattened_csv)
    """
    base_dir = r"E:\igan_nephropathy_research2"

    # 如果提供了完整模型名称但未提供model_base，则从model_name提取
    if model_name and not model_base:
        if ':' in model_name:
            model_base = model_name.split(':')[0]
        else:
            model_base = model_name

    input_dir = os.path.join(base_dir, f'pathology_feature_{model_base}_cleaned')
    output_csv = os.path.join(base_dir, 'data', f'pathology_ollama_embed_{model_base}.csv')
    flattened_csv = os.path.join(base_dir, 'data', f'pathology_lower_flattened_{model_base}.csv')

    return input_dir, output_csv, flattened_csv


def main(model_base: str = None, model_name: str = None):
    """主函数示例 - 可以直接在Python脚本中调用"""

    # ==================== 配置参数 ====================

    # 如果没有提供model_base，使用默认值
    if model_base is None:
        model_base = "qwen"

    # 如果没有提供model_name，根据model_base设置默认模型
    if model_name is None:
        model_mapping = {
            "qwen": "qwen3-embedding:latest",
            "deepseek": "deepseek-r1:latest",
            "bge": "bge-m3:latest",
            "gemma": "embeddinggemma",
            "nomic": "nomic-embed-text-v2-moe"
        }
        model_name = model_mapping.get(model_base, "qwen3-embedding:latest")

    # 创建标准化的文件路径
    input_dir, output_csv, flattened_csv = create_paths(model_base, model_name)

    # 处理配置
    max_workers = 48  # 根据CPU核心数调整
    num_sections = 4  # 病理报告部分数量

    # ==================== 执行处理 ====================

    logger.info("=" * 60)
    logger.info(f"开始处理病理报告，模型: {model_name}")
    logger.info("=" * 60)

    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
        # 尝试创建目录或使用备用路径
        alternative_input_dir = os.path.join(r"E:\igan_nephropathy_research2", "pathology_feature_json")
        if os.path.exists(alternative_input_dir):
            logger.info(f"使用备用目录: {alternative_input_dir}")
            input_dir = alternative_input_dir
        else:
            logger.error("无法找到有效的输入目录")
            return pd.DataFrame()

    # 处理报告并获取嵌入向量
    embedding_df = process_reports(
        model_name=model_name,
        model_base=model_base,
        input_dir=input_dir,
        output_csv=output_csv,
        flattened_csv=flattened_csv,
        max_workers=max_workers,
        num_sections=num_sections
    )

    if not embedding_df.empty:
        logger.info(f"处理完成！共处理 {len(embedding_df)} 个报告")
        logger.info(f"数据已保存到: {output_csv}")
    else:
        logger.error("处理失败，未生成有效数据")

    return embedding_df


def batch_process_models(models: List[str] = None):
    """
    批量处理多个模型

    Args:
        models: 模型基础名称列表，如 ['qwen', 'deepseek', 'bge']
    """
    if models is None:
        models = ['qwen', 'deepseek', 'bge']

    results = {}

    for model_base in models:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"开始处理模型: {model_base}")
        logger.info(f"{'=' * 60}")

        try:
            embedding_df = main(model_base=model_base)
            results[model_base] = embedding_df

            if not embedding_df.empty:
                logger.info(f"模型 {model_base} 处理完成，共 {len(embedding_df)} 条记录")
            else:
                logger.warning(f"模型 {model_base} 处理失败")

        except Exception as e:
            logger.error(f"处理模型 {model_base} 时发生错误: {e}")
            results[model_base] = None

    return results


if __name__ == "__main__":
    # 方法3：批量处理多个模型
    results = batch_process_models(models=['qwen', 'deepseek'])

