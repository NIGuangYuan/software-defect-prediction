"""
SHAP 可解释性分析模块

使用 SHAP (SHapley Additive exPlanations) 分析模型预测的可解释性。

SHAP 值解释每个特征对预测结果的贡献程度：
- 正值（红色）→ 推动预测为"有缺陷"
- 负值（蓝色）→ 推动预测为"无缺陷"

参考：
- Lundberg, S. M., & Lee, S. I. (2017).
  "A Unified Approach to Interpreting Model Predictions." NeurIPS.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from typing import Any

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def shap_summary_plot(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    save_path: str,
    model_name: str = "",
    dataset_name: str = "",
    max_display: int = 15,
):
    """
    生成 SHAP 摘要图（蜂群图）—— 展示每个特征对预测的贡献分布

    Args:
        model: 训练好的模型（需要有 predict_proba 方法或为树模型）
        X_train: 训练数据（用于构建 explainer 的背景数据）
        X_test: 测试数据（用于计算 SHAP 值）
        save_path: 保存路径
        model_name: 模型名称
        dataset_name: 数据集名称
        max_display: 最多显示的特征数
    """
    print(f"  [SHAP] 计算 {model_name} 在 {dataset_name} 上的 SHAP 值...")

    # 根据模型类型选择 explainer
    model_type = type(model).__name__

    try:
        # 对于树模型，使用 TreeExplainer（快）
        if any(kw in model_type.lower() for kw in ("forest", "tree", "xgb", "boost")):
            explainer = shap.TreeExplainer(model)
            # 使用部分测试数据以加速
            sample_size = min(200, len(X_test))
            X_sample = X_test.iloc[:sample_size]
            shap_values = explainer.shap_values(X_sample)

            # TreeExplainer 可能返回 list（多类）或 array
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # 取正类（有缺陷）的 SHAP 值
        else:
            # 对于线性模型/SVM，使用 KernelExplainer
            train_sample = X_train.iloc[:min(100, len(X_train))]
            test_sample = X_test.iloc[:min(100, len(X_test))]
            X_sample = test_sample
            explainer = shap.KernelExplainer(
                model.predict_proba, train_sample
            )
            shap_values = explainer.shap_values(test_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        # 绘制
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=X_test.columns.tolist(),
            max_display=max_display,
            show=False,
        )
        plt.title(
            f"SHAP Summary - {model_name} on {dataset_name}",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [保存] SHAP 摘要图 → {save_path}")

    except Exception as e:
        print(f"  [SHAP 警告] {model_name}: {e}")


def shap_bar_plot(
    model: Any,
    X_test: pd.DataFrame,
    save_path: str,
    model_name: str = "",
    dataset_name: str = "",
    max_display: int = 15,
):
    """
    生成 SHAP 条形图 —— 按重要性排序的特征平均 |SHAP|

    Args:
        model: 训练好的模型
        X_test: 测试数据
        save_path: 保存路径
        model_name: 模型名称
        dataset_name: 数据集名称
        max_display: 最多显示的特征数
    """
    model_type = type(model).__name__

    try:
        if any(kw in model_type.lower() for kw in ("forest", "tree", "xgb", "boost")):
            explainer = shap.TreeExplainer(model)
            sample_size = min(200, len(X_test))
            X_sample = X_test.iloc[:sample_size]
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            return  # 跳过非树模型（KernelExplainer 太慢）

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=X_test.columns.tolist(),
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.title(
            f"SHAP Feature Importance - {model_name} on {dataset_name}",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [保存] SHAP 条形图 → {save_path}")

    except Exception as e:
        print(f"  [SHAP 警告] {model_name}: {e}")


def run_shap_analysis(
    models: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    dataset_name: str,
    figures_dir: str,
):
    """
    对数据集的树模型（RF, XGBoost）进行 SHAP 分析

    注意：仅对树模型使用 TreeExplainer（速度快）。
    对于 SVM 和 Logistic Regression，KernelExplainer 计算过于耗时，跳过。

    Args:
        models: {名称: 模型} 字典
        X_train: 训练特征
        X_test: 测试特征
        dataset_name: 数据集名称
        figures_dir: 图表保存目录
    """
    print(f"\n--- SHAP 可解释性分析: {dataset_name} ---")

    # 仅对树模型做 SHAP（TreeExplainer 速度快）
    tree_models = {
        k: v for k, v in models.items()
        if any(kw in type(v).__name__.lower() for kw in ("forest", "tree", "xgb", "boost"))
    }

    if not tree_models:
        print("  [SHAP] 该数据集没有树模型，跳过 SHAP 分析")
        return

    for name, model in tree_models.items():
        safe_name = name.replace(" ", "_")

        # SHAP 摘要图
        shap_summary_plot(
            model, X_train, X_test,
            save_path=os.path.join(figures_dir, f"{dataset_name}_{safe_name}_shap_summary.png"),
            model_name=name,
            dataset_name=dataset_name,
        )

        # SHAP 条形图
        shap_bar_plot(
            model, X_test,
            save_path=os.path.join(figures_dir, f"{dataset_name}_{safe_name}_shap_bar.png"),
            model_name=name,
            dataset_name=dataset_name,
        )


if __name__ == "__main__":
    # 测试 SHAP 分析
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_all_datasets
    from preprocess import prepare_data
    from models import train_all_models

    datasets = load_all_datasets()
    df = datasets["KC1"]
    X_tr, X_te, y_tr, y_te, _ = prepare_data(df)
    models = train_all_models(X_tr, y_tr)

    figures_dir = "../results/figures"
    os.makedirs(figures_dir, exist_ok=True)

    run_shap_analysis(models, X_tr, X_te, "KC1", figures_dir)
