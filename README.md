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

## 参考来源

- 数据集：NASA MDP / PROMISE Repository (http://openscience.us/repo/)
- 参考文献："Machine Learning Approaches in Software Fault Prediction: A Review" (Cureus, 2026)
- 参考项目：GitHub 上基于 sklearn 的缺陷预测开源项目
