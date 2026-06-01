"""
主流水线入口

完整执行流程：
1. 下载并加载 NASA MDP 数据集 (JM1, KC1, PC1)
2. 数据预处理（清洗、特征工程、SMOTE）
3. 训练多种 ML 模型（LR, SVM, RF, XGBoost）
4. 模型评估与对比
5. 生成实验结果图表
6. SHAP 可解释性分析

用法:
    python main.py                          # 默认运行所有数据集
    python main.py --datasets KC1           # 仅运行 KC1
    python main.py --no-smote               # 不使用 SMOTE
    python main.py --tune                   # 启用超参数调优

参考来源:
    - 数据集: NASA MDP / PROMISE Repository
    - 论文: "Machine Learning Approaches in Software Fault Prediction: A Review" (2026)
    - 参考项目: GitHub 开源缺陷预测项目
"""

import os
import sys
import argparse
import warnings
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_all_datasets
from preprocess import prepare_data
from models import train_all_models, get_feature_importance
from evaluate import (
    evaluate_all_models,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_metrics_comparison,
    plot_feature_importance,
    save_results_table,
    generate_classification_report,
)

warnings.filterwarnings("ignore")

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# 数据集列表
DATASET_NAMES = ["JM1", "KC1", "PC1"]


def run_pipeline(
    dataset_name: str,
    df: pd.DataFrame,
    use_smote: bool = True,
    tune_hyperparams: bool = False,
):
    """
    在单个数据集上运行完整流水线

    Args:
        dataset_name: 数据集名称
        df: 原始 DataFrame
        use_smote: 是否使用 SMOTE
        tune_hyperparams: 是否超参数调优

    Returns:
        (评估结果 DataFrame, 训练好的模型字典)
    """
    header = f" {dataset_name} "
    print(f"\n{'#'*60}")
    print(f"#{header:=^58}#")
    print(f"{'#'*60}")

    # 1. 预处理
    print(f"\n[步骤1] 数据预处理...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        df, use_smote=use_smote
    )

    # 2. 训练模型
    print(f"\n[步骤2] 训练模型...")
    t0 = time.time()
    models = train_all_models(
        X_train, y_train,
        tune_hyperparams=tune_hyperparams,
    )
    elapsed = time.time() - t0
    print(f"  训练完成，耗时: {elapsed:.1f}s")

    # 3. 评估
    print(f"\n[步骤3] 模型评估...")
    results_df = evaluate_all_models(models, X_test, y_test)

    # 打印结果表格
    display_cols = ["模型", "Accuracy", "Precision", "Recall", "F1-Score", "MCC"]
    if "ROC-AUC" in results_df.columns:
        display_cols.append("ROC-AUC")
    print(
        results_df[display_cols]
        .to_string(index=False)
    )

    # 4. 保存图表
    print(f"\n[步骤4] 生成可视化...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    # 混淆矩阵
    plot_confusion_matrices(
        models, X_test, y_test,
        save_path=os.path.join(FIGURES_DIR, f"{dataset_name}_confusion_matrix.png"),
        dataset_name=dataset_name,
    )

    # ROC 曲线
    plot_roc_curves(
        models, X_test, y_test,
        save_path=os.path.join(FIGURES_DIR, f"{dataset_name}_roc_curves.png"),
        dataset_name=dataset_name,
    )

    # 指标对比
    plot_metrics_comparison(
        results_df,
        save_path=os.path.join(FIGURES_DIR, f"{dataset_name}_metrics_comparison.png"),
        dataset_name=dataset_name,
    )

    # 保存评估表格
    save_results_table(
        results_df,
        save_path=os.path.join(TABLES_DIR, f"{dataset_name}_results.csv"),
        dataset_name=dataset_name,
    )

    # 5. 特征重要性
    print(f"\n[步骤5] 特征重要性分析...")
    for name, model in models.items():
        imp = get_feature_importance(model, X_train.columns.tolist(), name)
        if imp is not None:
            print(f"\n  [{name}] Top-10 重要特征:")
            for _, row in imp.head(10).iterrows():
                print(f"    {row['feature']:25s} {row['importance']:.4f}")

            plot_feature_importance(
                imp,
                save_path=os.path.join(
                    FIGURES_DIR, f"{dataset_name}_{name.replace(' ', '_')}_importance.png"
                ),
                model_name=f"{name} on {dataset_name}",
            )

    # 6. 分类报告
    print(f"\n[步骤6] 分类报告...")
    for name, model in models.items():
        report = generate_classification_report(model, X_test, y_test, name)
        print(report)

    return results_df, models


def run_all_datasets(
    datasets: dict,
    use_smote: bool = True,
    tune_hyperparams: bool = False,
) -> dict:
    """
    在所有数据集上运行流水线

    Args:
        datasets: {名称: DataFrame}
        use_smote: 是否使用 SMOTE
        tune_hyperparams: 是否超参数调优

    Returns:
        {数据集名称: (结果DataFrame, 模型字典)}
    """
    all_results = {}

    for name in DATASET_NAMES:
        if name not in datasets:
            print(f"警告: 数据集 {name} 不可用，跳过")
            continue
        all_results[name] = run_pipeline(
            name, datasets[name],
            use_smote=use_smote,
            tune_hyperparams=tune_hyperparams,
        )

    return all_results


def generate_summary_table(all_results: dict):
    """
    生成跨数据集汇总对比表（最佳F1模型）
    """
    print(f"\n\n{'#'*80}")
    print(f"#{' 综合汇总 ':=^78}#")
    print(f"{'#'*80}")

    summary_rows = []
    for dataset_name, (results_df, models) in all_results.items():
        # 找 F1 最高的模型
        best_idx = results_df["F1-Score"].idxmax()
        best_row = results_df.iloc[best_idx]
        summary_rows.append({
            "数据集": dataset_name,
            "最佳模型": best_row["模型"],
            "Accuracy": best_row["Accuracy"],
            "Precision": best_row["Precision"],
            "Recall": best_row["Recall"],
            "F1-Score": best_row["F1-Score"],
            "MCC": best_row["MCC"],
            "ROC-AUC": best_row.get("ROC-AUC", "N/A"),
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n各数据集最佳F1-Score模型:")
    print(summary_df.to_string(index=False))

    # 保存汇总表
    os.makedirs(TABLES_DIR, exist_ok=True)
    summary_path = os.path.join(TABLES_DIR, "summary_all_datasets.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n汇总表已保存至: {summary_path}")

    return summary_df


def print_setup_info(args):
    """打印运行配置信息"""
    print(f"\n{'='*60}")
    print(f"  软件缺陷预测 - 实验流水线")
    print(f"{'='*60}")
    print(f"  数据集: {args.datasets}")
    print(f"  SMOTE: {'开启' if not args.no_smote else '关闭'}")
    print(f"  超参数调优: {'开启' if args.tune else '关闭'}")
    print(f"  结果目录: {RESULTS_DIR}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Software Defect Prediction Pipeline"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASET_NAMES,
        help=f"要运行的数据集 (默认: {DATASET_NAMES})"
    )
    parser.add_argument(
        "--no-smote", action="store_true",
        help="禁用 SMOTE 过采样"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="启用超参数调优 (耗时较长)"
    )
    args = parser.parse_args()

    print_setup_info(args)

    # 加载数据
    print(f"\n[初始化] 下载并加载数据集...")
    datasets = load_all_datasets(DATA_DIR)

    # 仅保留用户指定的数据集
    selected = {k: v for k, v in datasets.items() if k in args.datasets}

    if not selected:
        print("错误: 没有可用的数据集")
        sys.exit(1)

    # 运行流水线
    all_results = run_all_datasets(
        selected,
        use_smote=not args.no_smote,
        tune_hyperparams=args.tune,
    )

    # 生成汇总
    generate_summary_table(all_results)

    print(f"\n{'='*60}")
    print(f"  实验完成! 所有结果已保存至 {RESULTS_DIR}")
    print(f"  - 图表: {FIGURES_DIR}")
    print(f"  - 表格: {TABLES_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
