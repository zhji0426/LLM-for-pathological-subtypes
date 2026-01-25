from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from datetime import datetime
import numpy as np
import traceback
from sklearn.base import clone
import pickle
import joblib
import json

import gc
import traceback
import time
import psutil
import os
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
# 分割数据
from sklearn.model_selection import train_test_split

# 结合Top3模型的集成方法
from sklearn.ensemble import VotingClassifier


def train_and_evaluate_spec(name, spec, X_train, y_train, X_test, y_test):
    """在 worker 内部创建/克隆模型并训练，返回 (trained_model, y_pred, result)。
    支持两种 spec 形式：
    - (ModelClass, params_dict)
    - 已实例化的模型对象（例如 LogisticRegression(...)）
    """
    try:
        # 如果 spec 是可迭代且长度为2，按 (Class, params) 处理
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            ModelClass, params = spec
            params = params.copy()
            for k in ('n_jobs', 'nthread', 'nthreads'):
                if k in params:
                    params[k] = 1
            model = ModelClass(**params)
        else:
            # 否则当作已实例化模型，使用 clone() 复制，并尝试将内部并行参数设为1
            orig = spec
            model = clone(orig)
            # 如果需要强制并行参数为1，则修改参数字典并重设（对多数 sklearn 模型有效）
            try:
                p = model.get_params()
                changed = False
                for k in ('n_jobs', 'nthread', 'nthreads'):
                    if k in p and p[k] != 1:
                        p[k] = 1
                        changed = True
                if changed:
                    model.set_params(**p)
            except Exception:
                pass

        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        y_pred = model.predict(X_test)
        try:
            y_pred_proba = model.predict_proba(X_test)
        except Exception:
            y_pred_proba = None

        accuracy = accuracy_score(y_test, y_pred)
        if y_pred_proba is not None:
            if len(np.unique(y_train)) > 2:
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            else:
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            auc_score = np.nan

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=1)

        result = {
            'accuracy': accuracy,
            'auc': auc_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time
        }
        return model, y_pred, result

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[{name}] 训练出错: {e}\n{tb}")
        return None, None, None


def print_memory_usage():
    """打印当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return f"内存使用: {memory_info.rss / 1024 / 1024:.2f} MB"


if __name__ == "__main__":
    # 创建模型保存目录
    model_dir = r"E:\igan_nephropathy_research2\models"
    os.makedirs(model_dir, exist_ok=True)

    models = {
        'Logistic': LogisticRegression(
            random_state=42,
            max_iter=1000,
            n_jobs=-1
        ),
        'RandomForest': RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            n_estimators=100
        ),
        'SVM': SVC(
            random_state=42,
            probability=True,
            cache_size=1000
        ),
        'KN': KNeighborsClassifier(
            n_jobs=-1
        ),

        'LightGBM': LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            n_estimators=200,
            verbose=-1
        ),

        'NN': MLPClassifier(
            random_state=42,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=10
        ),

        'XGBoost': XGBClassifier(
            random_state=42,
            tree_method='hist',
            predictor='cpu_predictor',
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='mlogloss'
        ),
        'NaiveBayes': BernoulliNB()
    }

    # 创建集成模型
    ensemble_model = VotingClassifier(
        estimators=[
            ('NN', models['NN']),
            ('Logistic', models['Logistic']),
            ('SVM', models['SVM'])
        ],
        voting='soft',  # 使用概率加权
        n_jobs=-1
    )

    # 将集成模型添加到模型字典中
    models['Ensemble'] = ensemble_model

    try:
        # 示例用法
        print("加载数据...")
        reports_df = pd.read_csv(r"E:\igan_nephropathy_research2\data\pathology_ollama_embed_deepseek.csv",
                                 ## phID as string
                                 dtype={'phID': str}, low_memory=False)
        cluster_qwen3 = pd.read_csv(
            r"E:\igan_nephropathy_research2\deepseek_clustering_results\20251215_022047\clustered_data_simplified.csv")
        print(f"数据加载完成，{print_memory_usage()}")
        cluster_qwen3 = cluster_qwen3.rename(columns={'best_cluster_label': 'kmeans_cluster'})
        ### 标签反转，0->1, 1->0，因为 qwen3 的聚类结果与实际标签相反，0 low， 1 high
        # cluster_qwen3['kmeans_cluster'] = cluster_qwen3['kmeans_cluster'].map({0: 1, 1: 0})
        reports_df = reports_df.merge(cluster_qwen3[["kmeans_cluster", "bio_id"]], on='bio_id', how='left')

        # 1. 转为数值矩阵并清理 NaN
        print("预处理数据...")
        y = reports_df['kmeans_cluster'].to_numpy()
        ### drop 非数值列，'phID' "bio_id"
        reports_df.drop(columns=['phID', 'bio_id', 'source_file', 'total_embedding_dim', 'timestamp',
                                 'glomerular_lesions_text_preview', 'kmeans_cluster',
                                 'tubulointerstitial_lesions_text_preview',
                                 'vascular_lesions_text_preview', 'immunofluorescence_text_preview'],
                        inplace=True)
        X = reports_df.astype(float).to_numpy()
        print(f"原始标签分布: {np.bincount(y)}")
        # 清理 NaN
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"数据预处理完成，{print_memory_usage()}")

        trained_models = {}
        results = {}

        # 创建进度条
        model_names = list(models.keys())
        print(f"开始训练 {len(model_names)} 个模型...")

        # 使用更简单的进度条，避免可能的冲突
        for i, name in enumerate(model_names, 1):
            print(f"\n{'=' * 50}")
            print(f"训练模型 [{i}/{len(model_names)}]: {name}")
            print(f"{'=' * 50}")
            print(f"开始训练前，{print_memory_usage()}")

            try:
                start_time = time.time()
                model, y_pred, result = train_and_evaluate_spec(name, models[name],
                                                                X_train, y_train, X_test,
                                                                y_test)

                training_time = time.time() - start_time

                if model is not None:
                    accuracy = result.get('accuracy', 'N/A')
                    print(f"✓ {name} 训练成功!")
                    print(f"  准确率: {accuracy}")
                    print(f"  训练时间: {training_time:.2f}s")

                    # 保存模型
                    model_path = os.path.join(model_dir, f"{name}_model.pkl")
                    try:
                        with open(model_path, 'wb') as f:
                            pickle.dump(model, f)
                        print(f"✓ 模型已保存: {model_path}")
                    except Exception as e:
                        print(f"✗ 保存模型失败: {str(e)}")

                    results[name] = result
                    trained_models[name] = model

                    # 清理内存
                    del model, y_pred

                else:
                    print(f"✗ {name} 训练失败 - 返回了 None")

            except Exception as e:
                print(f"✗ {name} 训练过程中出现异常: {str(e)}")
                print("异常详情:")
                traceback.print_exc()

            # 强制垃圾回收
            gc.collect()
            print(f"训练完成后，{print_memory_usage()}")

            # 简单的进度显示
            progress = f"[{i}/{len(model_names)}]"
            print(f"进度: {progress}")

        # 显示总体训练结果
        print("\n" + "=" * 60)
        print("训练完成！汇总结果:")
        print("=" * 60)
        successful_models = list(results.keys())
        print(f"成功训练的模型: {len(successful_models)}/{len(model_names)}")

        for name in successful_models:
            result = results[name]
            accuracy = result.get('accuracy', 'N/A')
            print(f"  {name}: 准确率 = {accuracy}")

        # 保存训练结果
        results_path = os.path.join(model_dir, "training_results.pkl")
        try:
            with open(results_path, 'wb') as f:
                pickle.dump({
                    'results': results,
                    'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'successful_models': successful_models
                }, f)
            print(f"\n✓ 所有结果已保存到 {results_path}")
        except Exception as e:
            print(f"\n✗ 保存训练结果失败: {str(e)}")

        training_results = joblib.load(r"E:\igan_nephropathy_research2\models\training_results.pkl")
        with open(r"E:\igan_nephropathy_research2\models\training_results.json", 'w') as f:
            json.dump(training_results, f, indent=4)

    except Exception as e:
        print(f"主程序执行失败: {str(e)}")
        print("异常详情:")
        traceback.print_exc()
