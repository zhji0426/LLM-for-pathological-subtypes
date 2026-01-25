import requests
import json
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, Any, Optional, Literal

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from threading import Lock

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMAPIError(Exception):
    """LLM API自定义异常"""
    pass


class JSONParseError(Exception):
    """JSON解析异常"""
    pass


# 模型配置
MODEL_CONFIGS = {
    "deepseek": {
        "api_base": "https://api.deepseek.com/v1/chat/completions",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat"
    },
    "qwen": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "api_key_env": "QWEN_API_KEY",
        "default_model": "qwen-plus"
    }
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((LLMAPIError, JSONParseError))
)
def extract_structured_features_safe(
        report_text: str,
        llm_provider: Literal["deepseek", "qwen"] = "deepseek",
        llm_model: Optional[str] = None,
        prompt_config_path: Optional[str] = None,
        prompt_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    安全版本的LLM特征提取

    Args:
        report_text: 病理报告文本
        llm_provider: LLM提供商，可选 "deepseek" 或 "qwen"
        llm_model: 模型名称，如果为None则使用默认模型
        prompt_config_path: prompt配置文件路径
        prompt_config: prompt配置字典（如果提供，则忽略prompt_config_path）

    Returns:
        结构化的病理特征字典

    Raises:
        LLMAPIError: API调用失败
        JSONParseError: JSON解析失败
    """

    # 验证输入
    if not report_text or not report_text.strip():
        raise ValueError("报告文本不能为空")

    if len(report_text.strip()) < 10:
        raise ValueError("报告文本过短，可能不包含有效信息")

    # 验证LLM提供商
    if llm_provider not in MODEL_CONFIGS:
        raise ValueError(f"不支持的LLM提供商: {llm_provider}，可选: {list(MODEL_CONFIGS.keys())}")

    # 获取模型配置
    model_config = MODEL_CONFIGS[llm_provider]

    # 加载prompt配置
    if prompt_config is None:
        if prompt_config_path is None:
            # 使用默认路径
            if llm_provider == "deepseek":
                prompt_config_path = r"E:\igan_nephropathy_research2\prompts\prompt_2025-12-12.json"
            else:
                prompt_config_path = r"E:\igan_nephropathy_research2\prompts\prompt_2025-12-10.json"

        if not os.path.exists(prompt_config_path):
            raise FileNotFoundError(f"Prompt配置文件不存在: {prompt_config_path}")

        with open(prompt_config_path, 'r', encoding='utf-8') as f:
            prompt_config = json.load(f)

    # 构建prompt
    prompt = build_original_prompt(prompt_config=prompt_config, report_text=report_text)

    # 获取API密钥
    api_key = os.getenv(model_config["api_key_env"])
    if not api_key:
        raise ValueError(f"请设置环境变量 {model_config['api_key_env']}")

    # 使用指定的模型或默认模型
    model_name = llm_model or model_config["default_model"]

    # 准备API请求
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 根据提供商调整参数
    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    # 提供商特定的参数
    if llm_provider == "deepseek":
        data["max_tokens"] = 6000
        timeout = 120
    else:  # qwen
        data["max_tokens"] = 8000
        timeout = 120

    try:
        logger.info(f"调用{llm_provider.upper()} API提取病理特征，使用模型: {model_name}")
        response = requests.post(model_config["api_base"], headers=headers, json=data, timeout=timeout)
        response.raise_for_status()

        result = response.json()

        if 'choices' not in result or not result['choices']:
            raise LLMAPIError("API响应格式异常")

        content = result['choices'][0]['message']['content']
        logger.info(f"成功获取{llm_provider.upper()} API响应")

    except requests.exceptions.RequestException as e:
        logger.error(f"{llm_provider.upper()} API请求失败: {e}")
        raise LLMAPIError(f"{llm_provider.upper()} API请求失败: {e}")
    except KeyError as e:
        logger.error(f"{llm_provider.upper()} API响应格式错误: {e}")
        raise LLMAPIError(f"{llm_provider.upper()} API响应格式错误")

    # 安全解析JSON
    try:
        parsed_data = json.loads(content)
        logger.info("JSON解析成功")
        return parsed_data

    except json.JSONDecodeError as e:
        logger.warning(f"直接JSON解析失败，尝试提取JSON内容: {e}")
        # 尝试从响应中提取JSON部分
        cleaned_content = _extract_json_from_text(content)
        if cleaned_content:
            try:
                parsed_data = json.loads(cleaned_content)
                logger.info("从文本中提取JSON成功")
                return parsed_data
            except json.JSONDecodeError:
                pass

        logger.error(f"JSON解析失败，原始内容: {content[:200]}...")
        raise JSONParseError(f"无法解析API返回的JSON内容: {e}")


# 构建完整的prompt（还原原始格式）
def build_original_prompt(prompt_config, report_text):
    prompt = f"""{prompt_config["role_instruction"]}

    报告内容：
    {report_text}

    请严格按照以下JSON格式返回结果，不要添加任何额外说明、注释或Markdown标记：

    {json.dumps(prompt_config["output_format"], indent=4)}

    提取与标准化规则：

    """

    # 添加提取规则（保持原始编号格式）
    for i, rule in enumerate(prompt_config["extraction_rules"], 1):
        prompt += f"    {i}. {rule['title']}：\n       - {rule['description']}\n"

    # 添加额外换行以匹配原始格式
    prompt += "    "

    return prompt


def _extract_json_from_text(text: str) -> Optional[str]:
    """从文本中提取JSON内容"""
    import re
    # 尝试匹配 {...} 格式的JSON
    json_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}'  # 匹配嵌套的JSON对象
    matches = re.findall(json_pattern, text, re.DOTALL)

    if matches:
        # 返回最长的匹配项（最可能是完整的JSON）
        return max(matches, key=len)

    # 如果没有找到嵌套JSON，尝试简单匹配
    simple_match = re.search(r'\{.*\}', text, re.DOTALL)
    return simple_match.group() if simple_match else None


def validate_extracted_features(features: Dict[str, Any]) -> bool:
    """
    验证提取的特征结构是否完整

    Args:
        features: 提取的特征字典

    Returns:
        bool: 验证是否通过
    """
    required_sections = [
        "MEST_C_score",
        "glomerular_lesions",
        "tubulointerstitial_lesions",
        "vascular_lesions",
        "immunofluorescence",
        "key_pathology_terms"
    ]

    for section in required_sections:
        if section not in features:
            logger.error(f"缺失必要部分: {section}")
            return False

    # 检查MEST-C评分结构
    mestc_keys = ["M", "E", "S", "T", "C"]
    if not all(key in features["MEST_C_score"] for key in mestc_keys):
        logger.error("MEST-C评分结构不完整")
        return False

    return True


# 使用示例和测试函数
def extraction(
        report_text: str,
        llm_provider: Literal["deepseek", "qwen"] = "deepseek",
        llm_model: Optional[str] = None,
        prompt_config_path: Optional[str] = None,
        prompt_config: Optional[Dict] = None
) -> Optional[Dict[str, Any]]:
    try:
        features = extract_structured_features_safe(
            report_text=report_text,
            llm_provider=llm_provider,
            llm_model=llm_model,
            prompt_config_path=prompt_config_path,
            prompt_config=prompt_config
        )

        if validate_extracted_features(features):
            return features
        else:
            return None

    except Exception as e:
        logger.error(f"特征提取失败: {e}")
        return None


print_lock = Lock()


def safe_print(message):
    """线程安全的打印函数"""
    with print_lock:
        print(message)


def process_reports_parallel(
        pathology_report,
        save_path,
        llm_provider: Literal["deepseek", "qwen"] = "deepseek",
        llm_model: Optional[str] = None,
        prompt_config_path: Optional[str] = None,
        prompt_config: Optional[Dict] = None,
        max_workers: int = 5,
        overwrite: bool = False
):
    """多线程处理报告

    Args:
        pathology_report: 病理报告数据
        save_path: 保存路径
        llm_provider: LLM提供商，可选 "deepseek" 或 "qwen"
        llm_model: 模型名称
        prompt_config_path: prompt配置文件路径
        prompt_config: prompt配置字典
        max_workers: 最大线程数
        overwrite: 是否覆盖已存在的文件，默认为False（跳过已存在的文件）
    """
    # 确保输出目录存在
    os.makedirs(save_path, exist_ok=True)

    # 验证LLM提供商
    if llm_provider not in MODEL_CONFIGS:
        safe_print(f"错误: 不支持的LLM提供商: {llm_provider}")
        return

    total_reports = len(pathology_report)
    successful = 0
    failed = 0
    skipped = 0

    safe_print(f"开始处理 {total_reports} 个报告，使用 {llm_provider.upper()} API")
    safe_print(f"模型: {llm_model or MODEL_CONFIGS[llm_provider]['default_model']}")
    safe_print(f"覆盖模式: {'是' if overwrite else '否'} (已存在文件将被跳过)")
    start_time = time.time()

    # 创建待处理任务列表
    tasks = []
    for i, row in pathology_report.iterrows():
        try:
            # 获取phid和pathology_id用于构建文件名
            phid = str(row['phID']) if pd.notna(row['phID']) else ""
            pathology_id = str(row['pathology_number']) if pd.notna(row['pathology_number']) else ""

            # 构建预期的输出文件名
            output_file = os.path.join(save_path, f"pathology_features_{phid}_{pathology_id}.json")

            # 检查是否应该跳过已处理的文件
            if not overwrite and os.path.exists(output_file):
                safe_print(f"报告 {i} (PHID: {phid}) 已存在，跳过处理")
                skipped += 1
                continue

            # 添加到任务列表
            tasks.append((row, i))
        except Exception as e:
            safe_print(f"准备报告 {i} 时发生错误: {str(e)}")
            failed += 1

    safe_print(f"实际需要处理的报告: {len(tasks)} (跳过: {skipped}, 准备失败: {failed})")

    # 重置失败计数，因为上面的失败是准备阶段的失败
    preparation_failed = failed
    failed = 0
    successful = 0

    if tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_index = {}
            for row, index in tasks:
                # 为每个任务传递参数
                future = executor.submit(
                    process_single_report_with_overwrite,
                    row, index, save_path, llm_provider, llm_model,
                    prompt_config_path, prompt_config, overwrite
                )
                future_to_index[future] = index

            # 处理完成的任务
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if result:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    safe_print(f"任务 {index} 执行异常: {str(e)}")
                    failed += 1
    else:
        safe_print("没有需要处理的报告")

    end_time = time.time()
    safe_print(f"\n处理完成！")
    safe_print(f"总报告数: {total_reports}")
    safe_print(f"成功: {successful}")
    safe_print(f"处理失败: {failed}")
    safe_print(f"准备失败: {preparation_failed}")
    safe_print(f"跳过: {skipped}")
    safe_print(f"总耗时: {end_time - start_time:.2f} 秒")


def process_single_report_with_overwrite(
        row,
        index,
        save_path,
        llm_provider: Literal["deepseek", "qwen"] = "deepseek",
        llm_model: Optional[str] = None,
        prompt_config_path: Optional[str] = None,
        prompt_config: Optional[Dict] = None,
        overwrite: bool = False
):
    """处理单个报告的函数（支持覆盖选项）

    Args:
        row: 报告数据
        index: 报告索引
        save_path: 保存路径
        llm_provider: LLM提供商
        llm_model: 模型名称
        prompt_config_path: prompt配置文件路径
        prompt_config: prompt配置字典
        overwrite: 是否覆盖已存在的文件
    """
    try:
        report_text = str(row['gross_description']) if pd.notna(row['gross_description']) else ""
        phid = str(row['phID']) if pd.notna(row['phID']) else ""
        pathology_id = str(row['pathology_number']) if pd.notna(row['pathology_number']) else ""

        # 构建输出文件名
        output_file = os.path.join(save_path, f"pathology_features_{phid}_{pathology_id}.json")

        # 如果文件已存在且不覆盖，则直接返回True（表示已存在）
        if not overwrite and os.path.exists(output_file):
            safe_print(f"报告 {index} (PHID: {phid}) 已存在，跳过处理")
            return True

        # 调用特征提取函数
        features = extraction(
            report_text=report_text,
            llm_provider=llm_provider,
            llm_model=llm_model,
            prompt_config_path=prompt_config_path,
            prompt_config=prompt_config
        )

        # 保存结果
        if features:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(features, f, ensure_ascii=False, indent=2)
            safe_print(f"报告 {index} (PHID: {phid}) 特征提取成功，已保存到 {output_file}")
            return True
        else:
            safe_print(f"报告 {index} (PHID: {phid}) 特征提取失败")
            # 如果特征提取失败但文件已存在，不删除原文件（保留上次成功的结果）
            return False

    except Exception as e:
        safe_print(f"处理报告 {index} 时发生错误: {str(e)}")
        return False


def process_single_report(row, index, save_path, llm_provider="deepseek", llm_model=None, prompt_config_path=None):
    """处理单个报告的函数（兼容原接口）"""
    return process_single_report_with_overwrite(
        row, index, save_path, llm_provider, llm_model,
        prompt_config_path, overwrite=False
    )


def process_reports_batch(
        pathology_report,
        save_path,
        llm_provider: Literal["deepseek", "qwen"] = "deepseek",
        llm_model: Optional[str] = None,
        prompt_config_path: Optional[str] = None,
        prompt_config: Optional[Dict] = None,
        batch_size: int = 100,
        max_workers: int = 5,
        overwrite: bool = False
):
    """分批处理报告，避免内存溢出"""
    total_reports = len(pathology_report)
    safe_print(f"开始分批处理 {total_reports} 个报告，每批 {batch_size} 个...")
    safe_print(f"使用 {llm_provider.upper()} API")

    for start_idx in range(0, total_reports, batch_size):
        end_idx = min(start_idx + batch_size, total_reports)
        batch = pathology_report.iloc[start_idx:end_idx]

        safe_print(f"处理批次 {start_idx // batch_size + 1}: 报告 {start_idx} 到 {end_idx - 1}")
        process_reports_parallel(
            batch, save_path, llm_provider, llm_model,
            prompt_config_path, prompt_config, max_workers, overwrite
        )

        # 可选：批次间延迟，避免资源竞争
        time.sleep(1)


def main(
        data_path: str = r"E:\igan_nephropathy_research2\data\iga_pathology.csv",
        llm_provider: Literal["deepseek", "qwen"] = "deepseek",
        llm_model: Optional[str] = None,
        prompt_config_path: Optional[str] = None,
        save_path: Optional[str] = None,
        max_workers: int = 40,
        overwrite: bool = False,
        batch_mode: bool = False,
        batch_size: int = 100
):
    """主函数，直接在Python中调用

    Args:
        data_path: 病理报告数据CSV文件路径
        llm_provider: LLM提供商，可选 "deepseek" 或 "qwen"
        llm_model: 模型名称
        prompt_config_path: prompt配置文件路径
        save_path: 特征保存路径，如果为None则根据提供商自动生成
        max_workers: 最大线程数
        overwrite: 是否覆盖已存在的文件
        batch_mode: 是否使用分批处理模式
        batch_size: 分批处理的大小
    """
    # 验证LLM提供商
    if llm_provider not in MODEL_CONFIGS:
        print(f"错误: 不支持的LLM提供商: {llm_provider}")
        return

    # 验证文件路径
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        return

    # 设置默认保存路径
    if save_path is None:
        if llm_provider == "deepseek":
            save_path = r"E:\igan_nephropathy_research2\pathology_feature_deepseek"
        else:
            save_path = r"E:\igan_nephropathy_research2\pathology_feature_qwen"

    # 设置默认模型
    if llm_model is None:
        llm_model = MODEL_CONFIGS[llm_provider]["default_model"]

    # 读取数据
    pathology_report = pd.read_csv(data_path,
                                   dtype={"phID": str, "pathology_number": str})

    print(f"已加载 {len(pathology_report)} 个病理报告")
    print(f"使用 {llm_provider.upper()} API")
    print(f"模型: {llm_model}")
    print(f"保存路径: {save_path}")
    print(f"最大线程数: {max_workers}")
    print(f"覆盖模式: {'是' if overwrite else '否'}")
    print(f"处理模式: {'分批处理' if batch_mode else '直接处理'}")
    if batch_mode:
        print(f"批次大小: {batch_size}")

    # 选择处理模式
    if batch_mode:
        print("使用分批处理模式...")
        process_reports_batch(pathology_report, save_path, llm_provider, llm_model,
                              prompt_config_path, None, batch_size, max_workers, overwrite)
    else:
        print("使用直接多线程处理模式...")
        process_reports_parallel(pathology_report, save_path, llm_provider, llm_model,
                                 prompt_config_path, None, max_workers, overwrite)


if __name__ == "__main__":
    # 示例1：使用DeepSeek（默认）
    main(
        data_path=r"E:\igan_nephropathy_research2\data\iga_pathology.csv",
        llm_provider="deepseek",
        llm_model="deepseek-chat",  # 可选: deepseek-coder, deepseek-reasoner等
        prompt_config_path=r"E:\igan_nephropathy_research2\prompts\prompt_2026-01-07.json",
        max_workers=40,
        overwrite=False,
        batch_mode=False
    )

    # 示例2：使用Qwen
    # main(
    #     data_path=r"E:\igan_nephropathy_research2\data\iga_pathology.csv",
    #     llm_provider="qwen",
    #     llm_model="qwen-plus",  # 可选: qwen-turbo, qwen-max等
    #     prompt_config_path=r"E:\igan_nephropathy_research2\prompts\prompt_2025-12-10.json",
    #     max_workers=30,
    #     overwrite=True,
    #     batch_mode=True,
    #     batch_size=50
    # )

    # 示例3：使用不同的prompt配置文件
    # main(
    #     data_path=r"E:\igan_nephropathy_research2\data\iga_pathology.csv",
    #     llm_provider="deepseek",
    #     llm_model="deepseek-chat",
    #     prompt_config_path=r"E:\igan_nephropathy_research2\prompts\prompt_new_version.json",
    #     save_path=r"E:\igan_nephropathy_research2\pathology_feature_deepseek_v2",
    #     max_workers=20,
    #     overwrite=True
    # )