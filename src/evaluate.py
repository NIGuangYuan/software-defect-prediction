"""
评估与可视化模块

负责模型评估、结果汇总、图表生成。

评估指标（参考 2026 综述论文的标准做法）：
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- MCC (Matthews Correlation Coefficient，适合不平衡数据)
- 混淆矩阵

可视化：
- 混淆矩阵热力图
- ROC 曲线
- 模型对比条形图
- 特征重要性图
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    roc_curve, classification_report,
)
from typing import Dict, Any, Optional

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 统一配色
COLORS = {
    "Logistic Regression": "#3498db",
    "SVM": "#e74c3c",
    "Random Forest": "#2ecc71",
    "XGBoost": "#f39c12",
}


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    对单个模型进行全面评估

    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签

    Returns:
        评估指标字典
    """
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    if y_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba)

    metrics["Confusion_Matrix"] = confusion_matrix(y_test, y_pred)

    return metrics


def evaluate_all_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    评估所有模型，返回对比表格

    Args:
        models: {名称: 模型} 字典
        X_test: 测试特征
        y_test: 测试标签

    Returns:
        评估结果 DataFrame
    """
    results = []

    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        row = {"模型": name}
        for k, v in metrics.items():
            if k != "Confusion_Matrix":
                row[k] = round(v, 4)
            else:
                tn, fp, fn, tp = v.ravel()
                row["TN"] = tn
                row["FP"] = fp
                row["FN"] = fn
                row["TP"] = tp
        results.append(row)

    return pd.DataFrame(results)


def plot_confusion_matrices(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str,
    dataset_name: str = "",
):
    """
    绘制所有模型的混淆矩阵

    Args:
        models: 模型字典
        X_test: 测试特征
        y_test: 测试标签
        save_path: 保存路径
        dataset_name: 数据集名称
    """
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Defect", "Defect"],
            yticklabels=["No Defect", "Defect"],
            ax=ax, cbar=False,
        )
        ax.set_title(f"{name}\n{dataset_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [保存] 混淆矩阵 → {save_path}")


def plot_roc_curves(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str,
    dataset_name: str = "",
):
    """
    绘制所有模型的 ROC 曲线

    Args:
        models: 模型字典
        X_test: 测试特征
        y_test: 测试标签
        save_path: 保存路径
        dataset_name: 数据集名称
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, linewidth=2, color=COLORS.get(name),
                    label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves - {dataset_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [保存] ROC曲线 → {save_path}")


def plot_metrics_comparison(
    results_df: pd.DataFrame,
    save_path: str,
    dataset_name: str = "",
):
    """
    绘制模型指标对比条形图

    Args:
        results_df: evaluate_all_models 返回的结果
        save_path: 保存路径
        dataset_name: 数据集名称
    """
    metric_cols = [c for c in results_df.columns if c not in ("模型", "TN", "FP", "FN", "TP")]
    n_metrics = len(metric_cols)
    n_models = len(results_df)

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 2), 5))

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [row[c] for c in metric_cols]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=row["模型"],
                      color=COLORS.get(row["模型"], "#888"))

        # 在柱子上标注数值
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Model Comparison - {dataset_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [保存] 指标对比 → {save_path}")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    save_path: str,
    model_name: str = "",
    top_n: int = 15,
):
    """
    绘制特征重要性条形图

    Args:
        importance_df: 特征重要性 DataFrame
        save_path: 保存路径
        model_name: 模型名称
        top_n: 显示前 N 个特征
    """
    df = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(df)))
    bars = ax.barh(range(len(df)), df["importance"].values, color=colors[::-1])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"].values, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Feature Importance - {model_name}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    for bar, val in zip(bars, df["importance"].values):
        ax.text(bar.get_width() + bar.get_width() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [保存] 特征重要性 → {save_path}")


def generate_classification_report(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> str:
    """
    生成 sklearn 分类报告文本

    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        model_name: 模型名称

    Returns:
        分类报告字符串
    """
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=["No Defect", "Defect"],
        zero_division=0,
    )
    return f"=== {model_name} ===\n{report}"


def save_results_table(
    results_df: pd.DataFrame,
    save_path: str,
    dataset_name: str = "",
):
    """
    保存评估结果为 CSV

    Args:
        results_df: 评估结果
        save_path: 保存路径
        dataset_name: 数据集名称
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  [保存] 评估表格 → {save_path}")


if __name__ == "__main__":
    # 测试评估流程
    from data_loader import load_all_datasets
    from preprocess import prepare_data
    from models import train_all_models, get_feature_importance

    datasets = load_all_datasets()
    df = datasets["KC1"]
    X_tr, X_te, y_tr, y_te, _ = prepare_data(df)
    models = train_all_models(X_tr, y_tr)

    results = evaluate_all_models(models, X_te, y_te)
    print(results.to_string(index=False))
