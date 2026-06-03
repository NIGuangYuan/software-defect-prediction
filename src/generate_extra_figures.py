"""
生成额外的可视化图表，用于完善项目报告

生成图表：
1. 各数据集类别分布（缺陷 vs 无缺陷）
2. 特征相关性热力图
3. 跨数据集模型性能对比
4. 各数据集样本量与缺陷率概览
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_all_datasets
from preprocess import clean_data

# 配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 配色
COLORS = {
    "Logistic Regression": "#3498db",
    "SVM": "#e74c3c",
    "Random Forest": "#2ecc71",
    "XGBoost": "#f39c12",
}
DATASET_COLORS = {"JM1": "#3498db", "KC1": "#e74c3c", "PC1": "#2ecc71"}


def plot_class_distribution(datasets: dict):
    """绘制各数据集的类别分布饼图和柱状图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, df) in zip(axes, datasets.items()):
        df_clean = clean_data(df.copy())
        defect_count = int(df_clean["label"].sum())
        no_defect_count = int(len(df_clean) - defect_count)

        sizes = [no_defect_count, defect_count]
        labels = [f"No Defect\n({no_defect_count})", f"Defect\n({defect_count})"]
        colors_pie = ["#2ecc71", "#e74c3c"]
        explode = (0, 0.05)

        wedges, texts, autotexts = ax.pie(
            sizes, explode=explode, labels=labels, colors=colors_pie,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11}
        )
        for at in autotexts:
            at.set_fontweight('bold')
        ax.set_title(f"{name}\nDefect Rate: {defect_count/len(df_clean):.1%}",
                     fontsize=14, fontweight="bold")

    plt.suptitle("Class Distribution Across Datasets", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "class_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[保存] 类别分布 → {save_path}")


def plot_correlation_heatmap(datasets: dict):
    """为每个数据集绘制特征相关性热力图"""
    for name, df in datasets.items():
        df_clean = clean_data(df.copy())
        feature_cols = [c for c in df_clean.columns if c != "label"]
        # 取前15个特征（或全部，如果少于15）
        cols = feature_cols[:min(15, len(feature_cols))]
        corr = df_clean[cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True,
            linewidths=0.5, ax=ax,
            annot_kws={"fontsize": 7}
        )
        ax.set_title(f"Feature Correlation Heatmap - {name}", fontsize=14, fontweight="bold")

        plt.tight_layout()
        save_path = os.path.join(FIGURES_DIR, f"{name}_correlation_heatmap.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[保存] 相关性热力图 → {save_path}")


def plot_cross_dataset_comparison():
    """绘制跨数据集模型性能对比图"""
    summary_path = os.path.join(TABLES_DIR, "summary_all_datasets.csv")
    summary_df = pd.read_csv(summary_path)

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "MCC", "ROC-AUC"]
    datasets_list = summary_df["数据集"].tolist()

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(metrics))
    width = 0.25

    for i, (_, row) in enumerate(summary_df.iterrows()):
        values = [row[m] for m in metrics]
        offset = (i - len(datasets_list) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width,
                      label=f"{row['数据集']} ({row['最佳模型']})",
                      color=DATASET_COLORS.get(row["数据集"], "#888"))

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Best Model Performance Comparison Across Datasets",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "cross_dataset_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[保存] 跨数据集对比 → {save_path}")


def plot_dataset_overview(datasets: dict):
    """绘制数据集概览图：样本量、特征数、缺陷率"""
    info = []
    for name, df in datasets.items():
        df_clean = clean_data(df.copy())
        feature_cols = [c for c in df_clean.columns if c != "label"]
        info.append({
            "Dataset": name,
            "Samples": len(df_clean),
            "Features": len(feature_cols),
            "Defect Rate": df_clean["label"].mean(),
            "Defects": int(df_clean["label"].sum()),
            "No Defects": int(len(df_clean) - df_clean["label"].sum()),
        })

    info_df = pd.DataFrame(info)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 子图1: 样本量
    colors_bar = [DATASET_COLORS.get(d, "#888") for d in info_df["Dataset"]]
    axes[0].bar(info_df["Dataset"], info_df["Samples"], color=colors_bar, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Sample Size per Dataset", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Samples", fontsize=11)
    for i, v in enumerate(info_df["Samples"]):
        axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold", fontsize=11)

    # 子图2: 缺陷/无缺陷堆叠
    axes[1].bar(info_df["Dataset"], info_df["No Defects"], color="#2ecc71",
                label="No Defect", edgecolor="white", linewidth=1.5)
    axes[1].bar(info_df["Dataset"], info_df["Defects"], bottom=info_df["No Defects"],
                color="#e74c3c", label="Defect", edgecolor="white", linewidth=1.5)
    axes[1].set_title("Defect Distribution per Dataset", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Number of Samples", fontsize=11)
    axes[1].legend(fontsize=10)

    # 子图3: 缺陷率
    axes[2].bar(info_df["Dataset"], info_df["Defect Rate"] * 100, color=colors_bar,
                edgecolor="white", linewidth=1.5)
    axes[2].set_title("Defect Rate per Dataset", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Defect Rate (%)", fontsize=11)
    for i, v in enumerate(info_df["Defect Rate"]):
        axes[2].text(i, v * 100 + 0.5, f"{v:.1%}", ha="center", fontweight="bold", fontsize=11)

    plt.suptitle("Dataset Overview", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "dataset_overview.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[保存] 数据集概览 → {save_path}")

    # 保存数据集信息表
    info_path = os.path.join(TABLES_DIR, "dataset_info.csv")
    info_df.to_csv(info_path, index=False, encoding="utf-8-sig")
    print(f"[保存] 数据集信息表 → {info_path}")


def plot_full_metrics_heatmap():
    """绘制所有模型在所有数据集上的完整性能热力图"""
    all_data = []
    for ds in ["JM1", "KC1", "PC1"]:
        csv_path = os.path.join(TABLES_DIR, f"{ds}_results.csv")
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            all_data.append({
                "Dataset": ds,
                "Model": row["模型"],
                "Accuracy": row["Accuracy"],
                "Precision": row["Precision"],
                "Recall": row["Recall"],
                "F1-Score": row["F1-Score"],
                "MCC": row["MCC"],
                "ROC-AUC": row["ROC-AUC"],
            })

    all_df = pd.DataFrame(all_data)

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "MCC", "ROC-AUC"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        pivot = all_df.pivot(index="Model", columns="Dataset", values=metric)
        sns.heatmap(
            pivot, annot=True, fmt=".4f", cmap="YlOrRd",
            linewidths=1, ax=ax, cbar_kws={"shrink": 0.8},
            vmin=0, vmax=1
        )
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.suptitle("Complete Performance Heatmap: All Models × All Datasets",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "full_metrics_heatmap.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[保存] 完整性能热力图 → {save_path}")


def plot_model_ranking():
    """绘制模型排名图（按F1-Score平均排名）"""
    all_data = []
    for ds in ["JM1", "KC1", "PC1"]:
        csv_path = os.path.join(TABLES_DIR, f"{ds}_results.csv")
        df = pd.read_csv(csv_path)
        df["Dataset"] = ds
        all_data.append(df)

    all_df = pd.concat(all_data, ignore_index=True)

    # 计算平均指标
    avg_metrics = all_df.groupby("模型").agg({
        "Accuracy": "mean", "Precision": "mean", "Recall": "mean",
        "F1-Score": "mean", "MCC": "mean", "ROC-AUC": "mean"
    }).round(4)

    # 排序
    avg_metrics = avg_metrics.sort_values("F1-Score", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(avg_metrics))
    width = 0.12

    plot_metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "MCC", "ROC-AUC"]
    colors_plot = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    for j, (metric, color) in enumerate(zip(plot_metrics, colors_plot)):
        offset = (j - len(plot_metrics) / 2 + 0.5) * width
        values = [avg_metrics.loc[model, metric] for model in avg_metrics.index]
        bars = ax.bar(x + offset, values, width, label=metric, color=color, alpha=0.85)

        for bar, val in zip(bars, values):
            if val == max(values) or val == min(values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(avg_metrics.index, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Average Score", fontsize=12)
    ax.set_title("Average Model Performance Across All Datasets", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", ncol=3)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "model_ranking.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[保存] 模型排名 → {save_path}")

    # 保存平均指标表
    avg_path = os.path.join(TABLES_DIR, "average_model_metrics.csv")
    avg_metrics.to_csv(avg_path, encoding="utf-8-sig")
    print(f"[保存] 平均指标表 → {avg_path}")


def main():
    print("=" * 60)
    print("生成额外可视化图表")
    print("=" * 60)

    # 加载数据
    print("\n[1/6] 加载数据集...")
    datasets = load_all_datasets(DATA_DIR)

    # 生成图表
    print("\n[2/6] 类别分布图...")
    plot_class_distribution(datasets)

    print("\n[3/6] 相关性热力图...")
    plot_correlation_heatmap(datasets)

    print("\n[4/6] 跨数据集对比图...")
    plot_cross_dataset_comparison()

    print("\n[5/6] 数据集概览图...")
    plot_dataset_overview(datasets)

    print("\n[6/6] 完整性能热力图与模型排名...")
    plot_full_metrics_heatmap()
    plot_model_ranking()

    print("\n" + "=" * 60)
    print("所有额外图表已生成完毕!")
    print(f"图表目录: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
