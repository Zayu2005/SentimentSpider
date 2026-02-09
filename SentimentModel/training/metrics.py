# -*- coding: utf-8 -*-
"""
评估指标

计算分类任务的各种评估指标
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from ..config import LABEL_NAMES, LABEL_NAMES_CN


@dataclass
class MetricResult:
    """评估结果"""
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    precision_per_class: Dict[str, float]
    recall_per_class: Dict[str, float]
    f1_per_class: Dict[str, float]
    confusion_matrix: np.ndarray
    report: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
            "precision_per_class": self.precision_per_class,
            "recall_per_class": self.recall_per_class,
            "f1_per_class": self.f1_per_class
        }

    def summary(self) -> str:
        """生成摘要"""
        lines = [
            "=" * 50,
            "评估结果摘要",
            "=" * 50,
            f"准确率 (Accuracy):     {self.accuracy:.4f}",
            f"宏平均 F1 (Macro F1):  {self.f1_macro:.4f}",
            f"加权 F1 (Weighted F1): {self.f1_weighted:.4f}",
            "",
            "各类别 F1:",
        ]

        for label, f1 in self.f1_per_class.items():
            cn_name = LABEL_NAMES_CN[LABEL_NAMES.index(label)] if label in LABEL_NAMES else label
            lines.append(f"  {cn_name} ({label}): {f1:.4f}")

        lines.append("")
        lines.append("混淆矩阵:")
        lines.append(str(self.confusion_matrix))
        lines.append("=" * 50)

        return "\n".join(lines)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None
) -> MetricResult:
    """
    计算分类指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        label_names: 标签名称列表

    Returns:
        MetricResult 实例
    """
    if label_names is None:
        label_names = LABEL_NAMES

    # 转换为 numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算各项指标
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # 各类别指标
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # 处理类别数量可能不足的情况
    num_classes = len(label_names)
    if len(precision_per_class) < num_classes:
        # 补齐缺失的类别
        full_precision = np.zeros(num_classes)
        full_recall = np.zeros(num_classes)
        full_f1 = np.zeros(num_classes)

        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        for i, label in enumerate(unique_labels):
            if label < num_classes:
                full_precision[label] = precision_per_class[i] if i < len(precision_per_class) else 0
                full_recall[label] = recall_per_class[i] if i < len(recall_per_class) else 0
                full_f1[label] = f1_per_class[i] if i < len(f1_per_class) else 0

        precision_per_class = full_precision
        recall_per_class = full_recall
        f1_per_class = full_f1

    precision_dict = {label_names[i]: float(precision_per_class[i]) for i in range(len(label_names))}
    recall_dict = {label_names[i]: float(recall_per_class[i]) for i in range(len(label_names))}
    f1_dict = {label_names[i]: float(f1_per_class[i]) for i in range(len(label_names))}

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))

    # 分类报告
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0
    )

    return MetricResult(
        accuracy=float(accuracy),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        precision_weighted=float(precision_weighted),
        recall_weighted=float(recall_weighted),
        f1_weighted=float(f1_weighted),
        precision_per_class=precision_dict,
        recall_per_class=recall_dict,
        f1_per_class=f1_dict,
        confusion_matrix=cm,
        report=report
    )
