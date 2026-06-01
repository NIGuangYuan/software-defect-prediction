"""
数据加载模块

负责加载 NASA MDP 数据集（JM1, KC1, PC1），
支持 CSV 和 ARFF 两种格式。

数据来源: PROMISE Repository
参考: NASA MDP (Metrics Data Program) 软件缺陷数据集
      ApoorvaKrisna/NASA-promise-dataset-repository (GitHub)
"""

import os
import pandas as pd
import numpy as np
from typing import Optional


# 数据集信息
DATASET_INFO = {
    "JM1": "C语言实时预测地面系统",
    "KC1": "C++地面数据存储管理系统",
    "PC1": "C语言地球轨道卫星飞行软件",
}

# 特征列标准名称映射（处理不同数据集中的命名不一致）
FEATURE_NAME_MAP = {
    # Halstead 度量（PC1 中为大写）
    "n": "n", "N": "n",
    "v": "v", "V": "v",
    "l": "l", "L": "l",
    "d": "d", "D": "d",
    "i": "i", "I": "i",
    "e": "e", "E": "e",
    "b": "b", "B": "b",
    "t": "t", "T": "t",
    # McCabe 度量（PC1 中 iv(G) 带括号）
    "iv(g)": "iv(g)", "iv(G)": "iv(g)",
    "iv_g": "iv(g)",
    # 其他
    "loccodeandcomment": "loccodeandcomment",
}

# 标签列候选名
LABEL_CANDIDATES = ["defects", "problems", "label", "bug", "has_defect"]


def _find_label_column(df: pd.DataFrame) -> Optional[str]:
    """在 DataFrame 中查找标签列"""
    for cand in LABEL_CANDIDATES:
        if cand in df.columns:
            return cand
    return None


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一列名格式：全部转为小写，去除空格和特殊字符

    Args:
        df: 原始 DataFrame

    Returns:
        列名标准化后的 DataFrame
    """
    rename_map = {}
    for col in df.columns:
        clean = col.strip().lower().replace(" ", "_").replace("-", "_")
        if clean != col:
            rename_map[col] = clean
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _clean_nasa_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗 NASA MDP 数据中的已知问题：
    1. 移除重复行
    2. 确保数值列为合适的数值类型

    注意：NASA MDP 数据集已知存在数据质量问题（Shepperd et al. 2013）。
    此处仅做基本的去重处理，保留所有有效数据用于模型训练。

    参考：Shepperd, M., Song, Q., Sun, Z., & Mair, C. (2013).
    "Data quality: Some comments on the NASA software defect datasets."
    IEEE Transactions on Software Engineering, 39(9), 1208-1215.

    Args:
        df: 原始 DataFrame

    Returns:
        清洗后的 DataFrame
    """
    original_len = len(df)

    # 移除重复行
    df = df.drop_duplicates()

    if len(df) < original_len:
        print(f"  [清洗] 移除 {original_len - len(df)} 条重复记录 ({original_len} → {len(df)})")

    return df


def load_csv(filepath: str) -> pd.DataFrame:
    """
    加载 CSV 格式的数据集

    Args:
        filepath: CSV 文件路径

    Returns:
        标准化后的 DataFrame
    """
    df = pd.read_csv(filepath)

    # 标准化列名
    df = _standardize_columns(df)

    # 处理特定命名不一致
    # PC1 可能有 iv(g) 写作 iv(g) 或 iv_g
    for col in list(df.columns):
        clean = col.lower().replace("(", "").replace(")", "").replace(" ", "_")
        if clean != col and clean not in df.columns:
            df = df.rename(columns={col: clean})

    return df


def load_arff(filepath: str) -> pd.DataFrame:
    """
    解析 ARFF 文件为 pandas DataFrame

    Args:
        filepath: ARFF 文件路径

    Returns:
        标准化后的 DataFrame
    """
    import arff

    with open(filepath, "r", encoding="utf-8") as f:
        data = arff.load(f)

    df = pd.DataFrame(data["data"], columns=[attr[0] for attr in data["attributes"]])
    df = _standardize_columns(df)

    return df


def load_dataset(
    name: str,
    data_dir: str = "../data/raw",
) -> pd.DataFrame:
    """
    加载单个数据集（自动检测 CSV 或 ARFF 格式）

    优先使用 CSV（来自 GitHub 仓库），备选 ARFF（来自 OpenML）

    Args:
        name: 数据集名称 (JM1, KC1, PC1)
        data_dir: 数据存储目录

    Returns:
        清洗并标准化的 DataFrame
    """
    name_lower = name.lower()

    # 优先 CSV
    csv_path = os.path.join(data_dir, f"{name_lower}.csv")
    arff_path = os.path.join(data_dir, f"{name_lower}.arff")

    if os.path.exists(csv_path):
        print(f"  [加载] CSV: {csv_path}")
        df = load_csv(csv_path)
    elif os.path.exists(arff_path):
        print(f"  [加载] ARFF: {arff_path}")
        df = load_arff(arff_path)
    else:
        raise FileNotFoundError(
            f"数据集 {name} 未找到。请将 {name_lower}.csv 放入 {data_dir}/"
        )

    # 查找并统一标签列
    label_col = _find_label_column(df)
    if label_col is None:
        raise ValueError(f"无法找到标签列，可用列: {list(df.columns)}")

    df = df.rename(columns={label_col: "label"})

    # 标签二值化
    if df["label"].dtype == object:
        df["label"] = df["label"].apply(
            lambda x: 0 if str(x).lower() in ("false", "n", "no", "0") else 1
        )
    df["label"] = df["label"].astype(int)

    # 数据清洗
    df = _clean_nasa_data(df)

    # 确保所有特征列是数值类型
    feature_cols = [c for c in df.columns if c != "label"]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除非数值特征列（如有）
    numeric_cols = feature_cols + ["label"]
    df = df[numeric_cols].dropna()

    return df


def load_all_datasets(data_dir: str = "../data/raw") -> dict:
    """
    加载所有 NASA MDP 数据集

    Args:
        data_dir: 数据存储目录

    Returns:
        {名称: DataFrame} 字典
    """
    datasets = {}

    for name in ["JM1", "KC1", "PC1"]:
        print(f"\n{'='*50}")
        desc = DATASET_INFO.get(name, "")
        print(f"加载数据集: {name} - {desc}")
        print(f"{'='*50}")

        try:
            df = load_dataset(name, data_dir)
            datasets[name] = df
            print(f"  样本数: {len(df)}, 特征数: {len(df.columns) - 1}")
            print(f"  缺陷率: {df['label'].mean():.2%} "
                  f"(缺陷: {df['label'].sum()}, 无缺陷: {(1-df['label']).sum()})")
        except FileNotFoundError as e:
            print(f"  [警告] {e}")

    return datasets


if __name__ == "__main__":
    datasets = load_all_datasets()
    for name, df in datasets.items():
        print(f"\n{name}: shape={df.shape}")
        print(f"  特征列: {[c for c in df.columns if c != 'label']}")
        print(f"  标签分布:\n{df['label'].value_counts()}")
