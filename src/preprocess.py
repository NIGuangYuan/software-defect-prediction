"""
数据预处理模块

负责数据清洗、特征工程、标准化、类别不平衡处理等。

技术栈：
- StandardScaler: 特征标准化
- SMOTE: 合成少数类过采样（处理缺陷数据不平衡）
- 特征选择: 移除低方差和共线性特征
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from typing import Tuple, Optional


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗：
    1. 移除全空列
    2. 填充缺失值（中位数）
    3. 移除重复行

    Args:
        df: 原始 DataFrame

    Returns:
        清洗后的 DataFrame
    """
    original_shape = df.shape

    # 移除全空列
    df = df.dropna(axis=1, how="all")

    # 分离特征和标签
    feature_cols = [c for c in df.columns if c != "label"]
    label = df["label"] if "label" in df.columns else None

    # 填充缺失值
    for col in feature_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # 移除重复行
    df = df.drop_duplicates()

    if original_shape != df.shape:
        print(f"  [清洗] {original_shape} → {df.shape}")

    return df


def remove_low_variance_features(
    X: pd.DataFrame, threshold: float = 0.01
) -> Tuple[pd.DataFrame, list]:
    """
    移除低方差特征

    Args:
        X: 特征矩阵
        threshold: 方差阈值

    Returns:
        (过滤后的特征, 保留的特征名列表)
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)

    kept_mask = selector.get_support()
    kept_cols = X.columns[kept_mask].tolist()
    removed_cols = X.columns[~kept_mask].tolist()

    if removed_cols:
        print(f"  [特征选择] 移除了 {len(removed_cols)} 个低方差特征: {removed_cols}")

    X_filtered = X[kept_cols]
    return X_filtered, kept_cols


def remove_highly_correlated_features(
    X: pd.DataFrame, threshold: float = 0.95
) -> Tuple[pd.DataFrame, list]:
    """
    移除高度相关的特征（保留每对中的第一个）

    Args:
        X: 特征矩阵
        threshold: 相关系数阈值

    Returns:
        (过滤后的特征, 保留的特征名列表)
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [
        column for column in upper.columns if any(upper[column] > threshold)
    ]

    if to_drop:
        print(f"  [共线性] 移除了 {len(to_drop)} 个高相关特征 (r>{threshold}): {to_drop}")

    kept_cols = [c for c in X.columns if c not in to_drop]
    return X[kept_cols], kept_cols


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
    use_smote: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    完整的数据预处理流水线

    步骤:
    1. 清洗数据
    2. 分离特征与标签
    3. 移除低方差特征
    4. 移除高相关特征
    5. 划分训练/测试集
    6. 标准化特征
    7. SMOTE 过采样（仅对训练集）

    Args:
        df: 原始数据
        test_size: 测试集比例
        random_state: 随机种子
        use_smote: 是否使用 SMOTE

    Returns:
        (X_train, X_test, y_train, y_test, scaler)
    """
    # 1. 清洗
    df = clean_data(df)

    # 2. 分离
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols]
    y = df["label"]

    print(f"  原始: {X.shape[1]} 个特征, {len(X)} 个样本")
    print(f"  正例(缺陷): {y.sum()}, 负例(无缺陷): {(1-y).sum()}")

    # 3. 低方差过滤
    X, _ = remove_low_variance_features(X)

    # 4. 共线性过滤
    X, kept_cols = remove_highly_correlated_features(X)

    # 5. 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 6. 标准化
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # 7. SMOTE
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"  [SMOTE] 平衡后训练样本: {len(X_train_scaled)} "
              f"(正例: {y_train.sum()}, 负例: {(1-y_train).sum()})")

    print(f"  训练集: {X_train_scaled.shape}, 测试集: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    # 测试预处理流水线
    from data_loader import load_all_datasets

    datasets = load_all_datasets()
    for name, df in datasets.items():
        print(f"\n{'='*50}")
        print(f"预处理: {name}")
        print(f"{'='*50}")
        X_tr, X_te, y_tr, y_te, scaler = prepare_data(df)
