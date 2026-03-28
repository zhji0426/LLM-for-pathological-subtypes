import json
from typing import Dict, Any, List
import copy
import os


class TemplateDrivenMapper:
    """模板驱动的数据映射器"""

    def __init__(self, template: Dict = None):
        """初始化模板"""
        # 先加载模板，然后移除指定键
        if template:
            self.template = self._remove_complement_complex(template)
        else:
            self.template = self._get_default_template()
            # 从默认模板中也移除
            self.template = self._remove_complement_complex(self.template)

    def _remove_complement_complex(self, template: Dict) -> Dict:
        """移除 complement_membrane_attack_complex_C5b-9 键"""
        # 深拷贝以避免修改原始模板
        cleaned_template = copy.deepcopy(template)

        # 检查并移除指定键
        if "immunofluorescence" in cleaned_template:
            immunofluorescence = cleaned_template["immunofluorescence"]
            if "complement_membrane_attack_complex_C5b-9" in immunofluorescence:
                del immunofluorescence["complement_membrane_attack_complex_C5b-9"]

        return cleaned_template

    def _get_default_template(self) -> Dict:
        """获取默认模板"""
        return {
            "MEST_C_score": {"M": "", "E": "", "S": "", "T": "", "C": ""},
            "glomerular_lesions": {
                "quantitative": {
                    "total_glomeruli": 0,
                    "global_sclerosis_ratio": "",
                    "segmental_sclerosis_ratio": "",
                    "cellular_crescents_ratio": "",
                    "fibrocellular_crescents_ratio": "",
                    "fibrous_crescents_ratio": ""
                },
                "qualitative": {
                    "endocapillary_hypercellularity_degree": "",
                    "mesangial_hypercellularity_degree": "",
                    "neutrophil_infiltration_present": False,
                    "fibrinoid_necrosis_present": False,
                    "capillary_loop_abnormalities": ""
                }
            },
            "tubulointerstitial_lesions": {
                "chronic_features": {
                    "tubular_atrophy_degree": "",
                    "tubular_atrophy_percentage": "",
                    "interstitial_fibrosis_degree": "",
                    "interstitial_fibrosis_percentage": ""
                },
                "active_inflammatory_features": {
                    "inflammatory_infiltration_present": False,
                    "inflammatory_infiltration_degree": "",
                    "inflammatory_infiltration_distribution": "",
                    "inflammatory_cell_composition": {
                        "lymphocytes": "",
                        "plasma_cells": "",
                        "neutrophils": "",
                        "eosinophils": ""
                    },
                    "tubulitis_present": False,
                    "tubulitis_degree": "",
                    "edema_present": False
                },
                "tubular_dilatation_degree": "",
                "tubular_regeneration_present": False
            },
            "vascular_lesions": {
                "chronic_changes": {
                    "arteriolosclerosis_severity": "",
                    "intimal_fibrosis_severity": "",
                    "medial_hypertrophy_severity": ""
                },
                "active_vascular_lesions": {
                    "vasculitis_present": False,
                    "vasculitis_type": "",
                    "thrombotic_microangiopathy_acute_present": False,
                    "thrombotic_microangiopathy_acute_features": "",
                    "vascular_inflammatory_infiltrate_present": False,
                    "vascular_inflammatory_infiltrate_cell_type": ""
                },
                "hyaline_arteriolosclerosis_description": ""
            },
            "immunofluorescence": {
                "IgA_deposition": {
                    "intensity": "",
                    "distribution_pattern": "",
                    "location": ""
                },
                "co_deposits": {
                    "IgG": {"present": False, "intensity": "", "distribution": ""},
                    "IgM": {"present": False, "intensity": "", "distribution": ""},
                    "C3": {"present": False, "intensity": "", "distribution": ""},
                    "C1q": {"present": False, "intensity": "", "distribution": ""},
                    "kappa_light_chain": {"present": False, "intensity": "", "distribution": ""},
                    "lambda_light_chain": {"present": False, "intensity": "", "distribution": ""}
                },
                "fibrinogen_deposition": {"present": False, "intensity": "", "distribution": ""}
                # 注意：这里已经移除了 complement_membrane_attack_complex_C5b-9
            },
            "inflammation_activity_summary": {
                "glomerular_activity_score": "",
                "tubulointerstitial_activity_score": "",
                "vascular_activity_score": "",
                "overall_inflammatory_burden": ""
            },
            "key_pathology_terms": []
        }

    def _get_all_template_keys(self, data: Dict, prefix: str = "") -> List[str]:
        """获取模板中所有叶子节点的路径"""
        keys = []
        for key, value in data.items():
            current_path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                keys.extend(self._get_all_template_keys(value, current_path))
            else:
                keys.append(current_path)
        return keys

    def _extract_value_by_path(self, source_data: Dict, path: str) -> Any:
        """从源数据中按路径提取值"""
        keys = path.split('.')
        current = source_data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def _set_value_by_path(self, target_data: Dict, path: str, value: Any) -> None:
        """在目标数据中按路径设置值"""
        keys = path.split('.')
        current = target_data

        for i, key in enumerate(keys[:-1]):
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _find_matching_source_keys(self, template_path: str, source_data: Dict, prefix: str = "") -> List[Any]:
        """在源数据中查找匹配模板路径的值"""
        matches = []

        def search_in_dict(data: Dict, current_prefix: str = ""):
            for key, value in data.items():
                full_key = f"{current_prefix}.{key}" if current_prefix else key

                # 检查是否匹配模板路径
                template_key = template_path.split('.')[-1].lower()
                source_key = key.lower()

                # 简单的关键词匹配
                if template_key in source_key or source_key in template_key:
                    matches.append(value)

                # 递归搜索
                if isinstance(value, dict):
                    search_in_dict(value, full_key)

        search_in_dict(source_data)
        return matches

    def map_data_to_template(self, source_data: Dict) -> Dict:
        """
        将源数据映射到模板
        只映射模板中存在的字段，不会添加新字段
        """
        # 深拷贝模板
        mapped_result = copy.deepcopy(self.template)

        # 1. 获取模板中所有叶子节点的路径
        template_paths = self._get_all_template_keys(mapped_result)

        # 2. 对于每个模板路径，尝试从源数据中提取值
        for template_path in template_paths:
            # 尝试直接按相同路径提取
            value = self._extract_value_by_path(source_data, template_path)

            # 如果直接提取失败，尝试查找匹配的键
            if value is None:
                matches = self._find_matching_source_keys(template_path, source_data)
                if matches:
                    # 取第一个匹配的值
                    value = matches[0]

            # 如果找到值，设置到结果中
            if value is not None:
                self._set_value_by_path(mapped_result, template_path, value)

        # 3. 应用规则：未提及字段填充NA
        self._apply_na_rule(mapped_result)

        # 4. 标准化MEST-C评分格式
        self._standardize_mestc_scores(mapped_result)

        return mapped_result

    def _apply_na_rule(self, data: Dict) -> None:
        """应用规则1: 未提及字段填充NA"""
        for key, value in data.items():
            if isinstance(value, dict):
                self._apply_na_rule(value)
            elif isinstance(value, list):
                # 列表类型保持原样
                if not value:
                    data[key] = []
            elif value in ["", None]:
                # 空字符串或None填充为"NA"
                data[key] = "NA"

    def _standardize_mestc_scores(self, data: Dict) -> None:
        """标准化MEST-C评分格式"""
        if "MEST_C_score" in data:
            mestc = data["MEST_C_score"]
            for key in ["M", "E", "S", "T", "C"]:
                if key in mestc:
                    value = mestc[key]
                    if isinstance(value, (int, float)):
                        # 数字转换为M0/E1等格式
                        prefix = "M" if key == "M" else key
                        mestc[key] = f"{prefix}{int(value)}"
                    elif isinstance(value, str) and value not in ["", "NA"]:
                        # 已经是字符串，确保格式正确
                        if not value.startswith(('M', 'E', 'S', 'T', 'C')):
                            try:
                                num_val = int(value)
                                prefix = "M" if key == "M" else key
                                mestc[key] = f"{prefix}{num_val}"
                            except:
                                # 无法转换，保持原样
                                pass


def load_template_from_prompt(prompt_file: str) -> Dict:
    """从prompt文件中加载模板并移除指定键"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)

    # 从prompt数据中提取output_format作为模板
    template = prompt_data.get("output_format", {})

    # 在加载后立即移除指定键
    def remove_key_recursive(data: Dict):
        if "complement_membrane_attack_complex_C5b-9" in data:
            del data["complement_membrane_attack_complex_C5b-9"]

        for value in data.values():
            if isinstance(value, dict):
                remove_key_recursive(value)

    remove_key_recursive(template)
    return template


def process_single_file(data_file: str, prompt_file: str = None) -> Dict:
    """处理单个文件"""
    # 加载源数据
    with open(data_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    # 加载模板
    if prompt_file:
        template = load_template_from_prompt(prompt_file)
        mapper = TemplateDrivenMapper(template)
    else:
        mapper = TemplateDrivenMapper()

    # 映射数据
    result = mapper.map_data_to_template(source_data)
    return result


def batch_process_files(data_files: List[str], prompt_file: str = None,
                        output_base_dir: str = "output") -> None:
    """批量处理文件"""
    from pathlib import Path

    # 加载模板（只加载一次）
    template = None
    if prompt_file:
        template = load_template_from_prompt(prompt_file)

    mapper = TemplateDrivenMapper(template)

    # 确保输出目录存在
    output_dir = Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, data_file in enumerate(data_files, 1):
        try:
            file_name = os.path.basename(data_file)
            print(f"[{i}/{len(data_files)}] 处理: {file_name}")

            # 加载源数据
            with open(data_file, 'r', encoding='utf-8') as f:
                source_data = json.load(f)

            # 映射数据
            result = mapper.map_data_to_template(source_data)

            # 验证结果中是否包含被移除的键
            if "immunofluorescence" in result:
                immunofluorescence = result["immunofluorescence"]
                if "complement_membrane_attack_complex_C5b-9" in immunofluorescence:
                    print(f"  警告: {file_name} 结果中仍包含 complement_membrane_attack_complex_C5b-9")

            # 保存结果
            output_file = output_dir / file_name
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        except Exception as e:
            file_name = os.path.basename(data_file)
            print(f"  处理失败 {file_name}: {e}")

def get_base_dir() -> str:
    """获取当前脚本所在目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 测试代码
if __name__ == "__main__":
    base_dir = get_base_dir()


    prompt_file = os.path.join(base_dir, "prompts", "prompt_2026-01-07.json")

    # 测试模板是否正确移除指定键
    print("=== 测试模板加载 ===")
    test_template = load_template_from_prompt(prompt_file)

    for i in ['deepseek']:
        org_path = os.path.join(base_dir, f"pathology_feature_{i}")
        if os.path.exists(org_path):
            data_files = [os.path.join(org_path, f) for f in os.listdir(org_path) if f.endswith('.json')]
            if data_files:
                print(f"\n=== 处理 {i} 目录 ===")
                print(f"找到 {len(data_files)} 个文件")
                batch_process_files(data_files, prompt_file=prompt_file,
                                    output_base_dir=os.path.join(base_dir, f"pathology_feature_{i}_cleaned"))
            else:
                print(f"在 {org_path} 中没有找到JSON文件")
        else:
            print(f"目录不存在: {org_path}")