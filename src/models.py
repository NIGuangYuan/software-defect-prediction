"""
模型训练模块

实现了四种分类模型的训练与超参数调优：

1. Logistic Regression (基线)
2. Support Vector Machine (SVM)
3. Random Forest (集成学习)
4. XGBoost (梯度提升)

参考：
- "Machine Learning Approaches in Software Fault Prediction: A Review" (2026)
- GitHub: TabNet-BugPredictor, software-defect-prediction 等项目
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from typing import Dict, Any, Optional


def create_model(model_type: str, random_state: int = 42) -> Any:
    """
    创建一个模型实例（使用默认超参数）

    Args:
        model_type: 'lr' | 'svm' | 'rf' | 'xgb'
        random_state: 随机种子

    Returns:
        scikit-learn 兼容的模型实例
    """
    if model_type == "lr":
        return LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
        )

    elif model_type == "svm":
        return SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state,
            class_weight="balanced",
        )

    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

    elif model_type == "xgb":
        return XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
        )

    else:
        raise ValueError(f"未知的模型类型: {model_type}")


def get_param_grid(model_type: str) -> Dict[str, list]:
    """
    获取对应模型的超参数搜索网格

    Args:
        model_type: 模型类型

    Returns:
        参数网格字典
    """
    grids = {
        "lr": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
        },
        "svm": {
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "kernel": ["rbf"],
        },
        "rf": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
        },
        "xgb": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.3],
        },
    }
    return grids.get(model_type, {})


def train_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune_hyperparams: bool = False,
    random_state: int = 42,
) -> Any:
    """
    训练单个模型

    Args:
        model_type: 模型类型
        X_train: 训练特征
        y_train: 训练标签
        tune_hyperparams: 是否进行超参数调优
        random_state: 随机种子

    Returns:
        训练好的模型
    """
    model = create_model(model_type, random_state)

    if tune_hyperparams:
        param_grid = get_param_grid(model_type)
        if param_grid:
            print(f"  [调参] 使用 GridSearchCV 搜索最佳参数...")
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(X_train, y_train)
            print(f"  [最佳参数] {grid_search.best_params_}")
            print(f"  [最佳CV-F1] {grid_search.best_score_:.4f}")
            return grid_search.best_estimator_

    model.fit(X_train, y_train)
    return model


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_types: list = None,
    tune_hyperparams: bool = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    训练所有指定类型的模型

    Args:
        X_train: 训练特征
        y_train: 训练标签
        model_types: 模型类型列表，默认全部
        tune_hyperparams: 是否调参
        random_state: 随机种子

    Returns:
        {模型名: 训练好的模型} 字典
    """
    if model_types is None:
        model_types = ["lr", "svm", "rf", "xgb"]

    models = {}
    model_names = {"lr": "Logistic Regression", "svm": "SVM", "rf": "Random Forest", "xgb": "XGBoost"}

    for mtype in model_types:
        name = model_names.get(mtype, mtype)
        print(f"\n--- 训练: {name} ---")
        models[name] = train_model(
            mtype, X_train, y_train,
            tune_hyperparams=tune_hyperparams,
            random_state=random_state,
        )

    return models


def get_feature_importance(
    model: Any, feature_names: list, model_name: str
) -> Optional[pd.DataFrame]:
    """
    提取模型的特征重要性

    Args:
        model: 训练好的模型
        feature_names: 特征名列表
        model_name: 模型名称

    Returns:
        特征重要性 DataFrame，若模型不支持则返回 None
    """
    importances = None

    if hasattr(model, "coef_"):
        # 线性模型
        coef = model.coef_
        if coef.shape[0] == 1:
            importances = np.abs(coef[0])
        else:
            importances = np.abs(coef).mean(axis=0)

    elif hasattr(model, "feature_importances_"):
        # 树模型
        importances = model.feature_importances_

    if importances is not None:
        df_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        return df_imp

    return None


if __name__ == "__main__":
    # 测试模型训练
    from data_loader import load_all_datasets
    from preprocess import prepare_data

    datasets = load_all_datasets()
    df = datasets["KC1"]
    X_tr, X_te, y_tr, y_te, scaler = prepare_data(df)

    models = train_all_models(X_tr, y_tr, tune_hyperparams=False)

    for name, model in models.items():
        imp = get_feature_importance(model, X_tr.columns.tolist(), name)
        if imp is not None:
            print(f"\n{name} Top-5 特征:")
            print(imp.head(5).to_string(index=False))
