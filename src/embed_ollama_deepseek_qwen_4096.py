# 然后运行下面的代码
import sys
import os

# 添加当前目录到Python路径

from embed_ollama import process_reports, create_paths, main

# 现在可以使用这些函数了

if __name__ == "__main__":
    for i in ["qwen", "deepseek"]:
        input_dir, output_csv, flattened_csv = create_paths(i)
        _ = process_reports(
            model_name='qwen3-embedding:latest',
            input_dir=input_dir,
            output_csv=output_csv,
            flattened_csv=flattened_csv,
            max_workers=48
        )
