"""
Comprehensive multi-class evaluation metrics for root cause classification.
Calculates macro/micro precision, recall, F1 scores, per-class metrics,
confusion matrix visualization, and saves classification reports.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)


@dataclass
class MultiClassMetrics:
    """Container for comprehensive multi-class evaluation metrics."""

    # Basic metrics
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float

    # Per-class metrics
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    per_class_support: Dict[str, int]

    # Confusion matrix
    confusion_matrix: np.ndarray
    confusion_matrix_normalized: np.ndarray

    # Additional metrics
    roc_auc_ovo: Optional[float] = None
    roc_auc_ovr: Optional[float] = None
    average_precision: Optional[float] = None

    # Predictions
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    y_proba: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_precision": self.weighted_precision,
            "weighted_recall": self.weighted_recall,
            "weighted_f1": self.weighted_f1,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
            "per_class_support": self.per_class_support,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "confusion_matrix_normalized": self.confusion_matrix_normalized.tolist(),
            "roc_auc_ovo": self.roc_auc_ovo,
            "roc_auc_ovr": self.roc_auc_ovr,
            "average_precision": self.average_precision,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert per-class metrics to pandas DataFrame."""
        data = {
            "precision": self.per_class_precision,
            "recall": self.per_class_recall,
            "f1": self.per_class_f1,
            "support": self.per_class_support,
        }
        return pd.DataFrame(data).sort_index()

    def save(self, filepath: Union[str, Path]) -> None:
        """Save metrics to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class MultiClassEvaluator:
    """Comprehensive evaluator for multi-class classification tasks."""

    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize evaluator.

        Args:
            class_names: List of class names (e.g., ["RC-01", "RC-02", ...]).
                        If None, will use numeric indices.
        """
        self.class_names = class_names

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> MultiClassMetrics:
        """
        Compute comprehensive multi-class evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for ROC-AUC)

        Returns:
            MultiClassMetrics object with all computed metrics
        """
        # Get unique classes
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(unique_classes)

        # Generate class names if not provided
        if self.class_names is None:
            class_names = [f"RC-{i + 1:02d}" for i in range(n_classes)]
        else:
            class_names = self.class_names

        # Ensure we have names for all classes
        if len(class_names) < n_classes:
            class_names.extend(
                [f"Class-{i}" for i in range(len(class_names), n_classes)]
            )

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Macro and weighted averages
        macro_precision = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        weighted_precision = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        weighted_recall = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Per-class metrics
        per_class_precision = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Get support (count of true instances for each class)
        unique, counts = np.unique(y_true, return_counts=True)
        support_dict = {class_names[i]: counts[i] for i in range(len(unique))}

        # Convert per-class arrays to dictionaries
        per_class_precision_dict = {
            class_names[i]: float(per_class_precision[i])
            for i in range(len(per_class_precision))
        }
        per_class_recall_dict = {
            class_names[i]: float(per_class_recall[i])
            for i in range(len(per_class_recall))
        }
        per_class_f1_dict = {
            class_names[i]: float(per_class_f1[i]) for i in range(len(per_class_f1))
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)

        # Additional metrics (if probabilities available)
        roc_auc_ovo = None
        roc_auc_ovr = None
        average_precision = None

        if y_proba is not None and y_proba.shape[1] == n_classes:
            try:
                # One-vs-One ROC-AUC
                roc_auc_ovo = roc_auc_score(
                    y_true, y_proba, multi_class="ovo", average="macro"
                )
                # One-vs-Rest ROC-AUC
                roc_auc_ovr = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
                # Average precision
                average_precision = average_precision_score(
                    y_true, y_proba, average="macro"
                )
            except Exception:
                # Skip if calculation fails (e.g., only one class present)
                pass

        return MultiClassMetrics(
            accuracy=float(accuracy),
            macro_precision=float(macro_precision),
            macro_recall=float(macro_recall),
            macro_f1=float(macro_f1),
            weighted_precision=float(weighted_precision),
            weighted_recall=float(weighted_recall),
            weighted_f1=float(weighted_f1),
            per_class_precision=per_class_precision_dict,
            per_class_recall=per_class_recall_dict,
            per_class_f1=per_class_f1_dict,
            per_class_support=support_dict,
            confusion_matrix=cm,
            confusion_matrix_normalized=cm_normalized,
            roc_auc_ovo=roc_auc_ovo,
            roc_auc_ovr=roc_auc_ovr,
            average_precision=average_precision,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
        )

    def plot_confusion_matrix(
        self,
        metrics: MultiClassMetrics,
        title: str = "Confusion Matrix",
        normalize: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            metrics: MultiClassMetrics object
            title: Plot title
            normalize: Whether to plot normalized confusion matrix
            figsize: Figure size
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        if normalize:
            cm = metrics.confusion_matrix_normalized
            fmt = ".2f"
            label = "Normalized"
        else:
            cm = metrics.confusion_matrix
            fmt = "d"
            label = "Count"

        fig, ax = plt.subplots(figsize=figsize)

        # Use class names if available, otherwise use indices
        if self.class_names is not None and len(self.class_names) == cm.shape[0]:
            labels = self.class_names
        else:
            labels = [f"Class {i}" for i in range(cm.shape[0])]

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={"label": label},
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"{title} ({label})")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def generate_classification_report(
        self,
        metrics: MultiClassMetrics,
        save_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Generate detailed classification report.

        Args:
            metrics: MultiClassMetrics object
            save_path: Path to save report (optional)

        Returns:
            Formatted classification report string
        """
        if metrics.y_true is None or metrics.y_pred is None:
            raise ValueError("y_true and y_pred must be available in metrics")

        # Get class names
        if self.class_names is not None:
            target_names = self.class_names
        else:
            n_classes = len(np.unique(metrics.y_true))
            target_names = [f"RC-{i + 1:02d}" for i in range(n_classes)]

        # Generate sklearn classification report
        report = classification_report(
            metrics.y_true,
            metrics.y_pred,
            target_names=target_names,
            output_dict=False,
            digits=4,
        )

        # Add summary metrics
        summary = f"""
Classification Report
====================

{report}

Summary Metrics:
----------------
Accuracy: {metrics.accuracy:.4f}
Macro Precision: {metrics.macro_precision:.4f}
Macro Recall: {metrics.macro_recall:.4f}
Macro F1-Score: {metrics.macro_f1:.4f}
Weighted Precision: {metrics.weighted_precision:.4f}
Weighted Recall: {metrics.weighted_recall:.4f}
Weighted F1-Score: {metrics.weighted_f1:.4f}
"""

        if metrics.roc_auc_ovo is not None:
            summary += f"ROC-AUC (One-vs-One): {metrics.roc_auc_ovo:.4f}\n"
        if metrics.roc_auc_ovr is not None:
            summary += f"ROC-AUC (One-vs-Rest): {metrics.roc_auc_ovr:.4f}\n"
        if metrics.average_precision is not None:
            summary += f"Average Precision: {metrics.average_precision:.4f}\n"

        if save_path:
            with open(save_path, "w") as f:
                f.write(summary)

        return summary

    def save_all_metrics(
        self,
        metrics: MultiClassMetrics,
        output_dir: Union[str, Path],
        prefix: str = "evaluation",
    ) -> Dict[str, Path]:
        """
        Save all evaluation metrics to files.

        Args:
            metrics: MultiClassMetrics object
            output_dir: Directory to save files
            prefix: Prefix for filenames

        Returns:
            Dictionary mapping metric type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save metrics as JSON
        json_path = output_dir / f"{prefix}_metrics.json"
        metrics.save(json_path)
        saved_files["json_metrics"] = json_path

        # Save classification report
        report_path = output_dir / f"{prefix}_report.txt"
        self.generate_classification_report(metrics, save_path=report_path)
        saved_files["classification_report"] = report_path

        # Save per-class metrics as CSV
        csv_path = output_dir / f"{prefix}_per_class.csv"
        df = metrics.to_dataframe()
        df.to_csv(csv_path)
        saved_files["per_class_csv"] = csv_path

        # Plot and save confusion matrices
        cm_count_path = output_dir / f"{prefix}_confusion_matrix_count.png"
        self.plot_confusion_matrix(
            metrics,
            title="Confusion Matrix (Count)",
            normalize=False,
            save_path=cm_count_path,
        )
        saved_files["confusion_matrix_count"] = cm_count_path

        cm_norm_path = output_dir / f"{prefix}_confusion_matrix_normalized.png"
        self.plot_confusion_matrix(
            metrics,
            title="Confusion Matrix (Normalized)",
            normalize=True,
            save_path=cm_norm_path,
        )
        saved_files["confusion_matrix_normalized"] = cm_norm_path

        return saved_files
