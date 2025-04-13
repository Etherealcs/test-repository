#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例Python文件，展示基本的机器学习流程
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 生成示例数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                          n_redundant=5, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建并训练模型
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 预测与评估
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"模型准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 可视化特征重要性
feature_importance = abs(model.coef_[0])
indices = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 6))
plt.title("特征重要性")
plt.bar(range(X.shape[1]), feature_importance[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.tight_layout()
plt.savefig("feature_importance.png")

if __name__ == "__main__":
    print("此脚本展示了基本的机器学习流程，包括数据准备、模型训练和评估。")
