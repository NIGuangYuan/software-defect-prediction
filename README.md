# 基于机器学习的软件缺陷预测

## 项目简介

本项目使用 NASA MDP (Metrics Data Program) 数据集，通过多种机器学习模型对软件模块是否存在缺陷进行预测。项目实现了从数据加载、预处理、模型训练到评估与可解释性分析的完整流水线。

## 数据集

使用 PROMISE 仓库中的 NASA MDP 数据集：
- **JM1**：实时预测地面系统（C语言，10,885条）
- **KC1**：地面数据存储管理系统（C++，2,109条）
- **PC1**：地球轨道卫星飞行软件（C语言，1,107条）

每条数据包含 21 个软件度量特征（McCabe圈复杂度、Halstead度量、代码行数等）和 1 个缺陷标签。

## 项目结构

```
software-defect-prediction/
├── data/raw/              # 原始数据集（ARFF格式）
├── src/
│   ├── data_loader.py     # 数据下载与加载
│   ├── preprocess.py      # 数据预处理
│   ├── models.py          # 模型定义与训练
│   ├── evaluate.py        # 评估与可视化
│   └── main.py            # 主流水线入口
├── results/figures/       # 实验结果图表
├── requirements.txt       # Python依赖
└── README.md              # 项目说明
```

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行完整流水线
```bash
cd src
python main.py
```

### 3. 使用自定义参数
```bash
python main.py --datasets JM1 KC1 PC1 --models lr rf svm xgb --smote
```

## 模型

| 模型 | 说明 |
|------|------|
| Logistic Regression | 线性基准模型 |
| Support Vector Machine | 经典非线性分类器 |
| Random Forest | 集成学习方法 |
| XGBoost | 梯度提升树模型 |

## 评估指标

- Accuracy、Precision、Recall、F1-Score
- ROC-AUC
- MCC (Matthews Correlation Coefficient)
- 混淆矩阵

## 主要特性

1. **多数据集对比**：在 JM1、KC1、PC1 三个数据集上进行评估
2. **类别不平衡处理**：使用 SMOTE 过采样
3. **可解释性分析**：SHAP 值解释模型预测
4. **特征重要性分析**：对比各软件度量指标的预测贡献
5. **综合评估**：多指标、多角度对比模型性能

## 实验结果摘要

### 各数据集最佳模型（按F1-Score）

| 数据集 | 最佳模型 | Accuracy | Precision | Recall | F1-Score | MCC | ROC-AUC |
|--------|----------|----------|-----------|--------|----------|-----|---------|
| JM1 | SVM | 0.6876 | 0.3685 | 0.5458 | 0.4400 | 0.2432 | 0.6952 |
| KC1 | Logistic Regression | 0.6593 | 0.3840 | 0.5053 | 0.4364 | 0.2026 | 0.6334 |
| PC1 | Random Forest | 0.8815 | 0.3243 | 0.5714 | 0.4138 | 0.3710 | 0.8476 |

### 模型平均排名（三数据集综合）

| 排名 | 模型 | Avg F1-Score | Avg MCC | Avg ROC-AUC | 适用场景 |
|------|------|-------------|---------|------------|----------|
| 🥇 | Random Forest | 0.4268 | 0.2896 | 0.7383 | 综合平衡 |
| 🥈 | Logistic Regression | 0.4139 | 0.2717 | 0.7383 | 高Recall需求 |
| 🥉 | XGBoost | 0.3829 | 0.2604 | 0.7255 | 高Precision需求 |
| 4 | SVM | 0.3842 | 0.2323 | 0.7172 | 非线性决策边界 |

### 关键发现

1. **代码规模度量（loc）和 Halstead volume（v）** 是最重要的缺陷预测因子
2. **SMOTE** 有效缓解了类别不平衡，但在极端不平衡场景下仍有局限
3. **模型选择需要场景权衡**：安全关键系统优先Recall，资源受限项目优先Precision
4. **数据集特性比模型选择对性能影响更大**

📄 **完整实验报告**：[报告.md](报告.md)（含39张可视化图表与详细分析）

## 参考来源

- 数据集：NASA MDP / PROMISE Repository (http://openscience.us/repo/)
- 参考文献："Machine Learning Approaches in Software Fault Prediction: A Review" (Cureus, 2026)
- 参考项目：GitHub 上基于 sklearn 的缺陷预测开源项目
