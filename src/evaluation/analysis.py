"""
Root cause analysis and insights module.
Provides error analysis, misclassification patterns, feature importance,
and recommendations for improving specific RC predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance


@dataclass
class MisclassificationAnalysis:
    """Analysis of misclassified samples and patterns."""

    # Misclassification patterns
    confusion_matrix: np.ndarray
    class_names: List[str]

    # Most common misclassifications
    top_misclassifications: List[Dict[str, Any]]

    # Per-class error rates
    per_class_error_rate: Dict[str, float]

    # Challenging pairs (frequently confused classes)
    challenging_pairs: List[Dict[str, Any]]

    # Samples that were misclassified
    misclassified_samples: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary for serialization."""
        return {
            "confusion_matrix": self.confusion_matrix.tolist(),
            "class_names": self.class_names,
            "top_misclassifications": self.top_misclassifications,
            "per_class_error_rate": self.per_class_error_rate,
            "challenging_pairs": self.challenging_pairs,
            "misclassified_samples_count": len(self.misclassified_samples)
            if self.misclassified_samples is not None
            else 0,
        }


@dataclass
class FeatureImportanceAnalysis:
    """Analysis of feature importance for each root cause."""

    # Feature names
    feature_names: List[str]

    # Global feature importance
    global_importance: Dict[str, float]

    # Per-class feature importance
    per_class_importance: Dict[str, Dict[str, float]]

    # Top features per class
    top_features_per_class: Dict[str, List[str]]

    # Common important features across classes
    common_important_features: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary for serialization."""
        return {
            "feature_names": self.feature_names,
            "global_importance": self.global_importance,
            "per_class_importance": self.per_class_importance,
            "top_features_per_class": self.top_features_per_class,
            "common_important_features": self.common_important_features,
        }


@dataclass
class RootCauseInsights:
    """Comprehensive insights and recommendations for root cause analysis."""

    # Analysis components
    misclassification_analysis: MisclassificationAnalysis
    feature_importance_analysis: FeatureImportanceAnalysis

    # Summary statistics
    overall_accuracy: float
    most_challenging_class: str
    easiest_class: str

    # Recommendations
    recommendations: List[Dict[str, Any]]

    # Patterns identified
    identified_patterns: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert insights to dictionary for serialization."""
        return {
            "overall_accuracy": self.overall_accuracy,
            "most_challenging_class": self.most_challenging_class,
            "easiest_class": self.easiest_class,
            "recommendations": self.recommendations,
            "identified_patterns": self.identified_patterns,
            "misclassification_analysis": self.misclassification_analysis.to_dict(),
            "feature_importance_analysis": self.feature_importance_analysis.to_dict(),
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Save insights to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class RootCauseAnalyzer:
    """Analyzer for root cause classification errors and insights."""

    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize root cause analyzer.

        Args:
            class_names: List of class names (e.g., ["RC-01", "RC-02", ...]).
                        If None, will use numeric indices.
        """
        self.class_names = class_names

    def analyze_misclassifications(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: Optional[pd.DataFrame] = None,
        sample_data: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None,
    ) -> MisclassificationAnalysis:
        """
        Analyze misclassifications to identify patterns and challenging categories.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            X: Feature matrix (optional, for feature importance)
            sample_data: Original sample data with log messages (optional)
            feature_names: Names of features (optional)

        Returns:
            MisclassificationAnalysis object
        """
        # Get class names
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(unique_classes)

        if self.class_names is None:
            class_names = [f"RC-{i + 1:02d}" for i in range(n_classes)]
        else:
            class_names = self.class_names

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))

        # Calculate per-class error rate
        per_class_error_rate = {}
        for i in range(n_classes):
            total = cm[i, :].sum()
            correct = cm[i, i]
            error_rate = (total - correct) / total if total > 0 else 0.0
            per_class_error_rate[class_names[i]] = error_rate

        # Find top misclassifications (off-diagonal elements)
        top_misclassifications = []
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    top_misclassifications.append(
                        {
                            "true_class": class_names[i],
                            "predicted_class": class_names[j],
                            "count": int(cm[i, j]),
                            "percentage": cm[i, j] / cm[i, :].sum()
                            if cm[i, :].sum() > 0
                            else 0.0,
                        }
                    )

        # Sort by count descending
        top_misclassifications.sort(key=lambda x: x["count"], reverse=True)

        # Identify challenging pairs (frequently confused classes)
        challenging_pairs = []
        for misclass in top_misclassifications[:10]:  # Top 10
            # Ensure percentage is float for comparison
            percentage = float(misclass["percentage"])
            if percentage > 0.1:  # At least 10% error rate for this pair
                challenging_pairs.append(
                    {
                        "pair": f"{misclass['true_class']} → {misclass['predicted_class']}",
                        "count": misclass["count"],
                        "error_rate": percentage,
                        "description": f"{misclass['true_class']} frequently misclassified as {misclass['predicted_class']}",
                    }
                )

        # Extract misclassified samples if sample_data provided
        misclassified_samples = None
        if sample_data is not None and len(sample_data) == len(y_true):
            misclassified_mask = y_true != y_pred
            misclassified_samples = sample_data[misclassified_mask].copy()
            misclassified_samples["true_label"] = [
                class_names[i] for i in y_true[misclassified_mask]
            ]
            misclassified_samples["predicted_label"] = [
                class_names[i] for i in y_pred[misclassified_mask]
            ]

        return MisclassificationAnalysis(
            confusion_matrix=cm,
            class_names=class_names,
            top_misclassifications=top_misclassifications[:20],  # Top 20
            per_class_error_rate=per_class_error_rate,
            challenging_pairs=challenging_pairs,
            misclassified_samples=misclassified_samples,
        )

    def analyze_feature_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y_true: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> FeatureImportanceAnalysis:
        """
        Analyze feature importance for each root cause.

        Args:
            model: Trained model with predict_proba method
            X: Feature matrix
            y_true: True labels
            feature_names: Names of features
            n_repeats: Number of repeats for permutation importance
            random_state: Random seed

        Returns:
            FeatureImportanceAnalysis object
        """
        if feature_names is None:
            if hasattr(X, "columns"):
                feature_names = list(X.columns)
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Get class names
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)

        if self.class_names is None:
            class_names = [f"RC-{i + 1:02d}" for i in range(n_classes)]
        else:
            class_names = self.class_names

        # Global feature importance (if model has feature_importances_)
        global_importance = {}
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for name, importance in zip(feature_names, importances):
                global_importance[name] = float(importance)
        else:
            # Use permutation importance as fallback
            try:
                result = permutation_importance(
                    model,
                    X,
                    y_true,
                    n_repeats=n_repeats,
                    random_state=random_state,
                    n_jobs=-1,
                )
                for name, importance in zip(feature_names, result.importances_mean):
                    global_importance[name] = float(importance)
            except:
                # If permutation importance fails, use uniform importance
                for name in feature_names:
                    global_importance[name] = 1.0 / len(feature_names)

        # Per-class feature importance
        per_class_importance = {}
        top_features_per_class = {}

        for class_idx in range(n_classes):
            class_name = class_names[class_idx]

            # Create binary labels for this class
            y_binary = (y_true == class_idx).astype(int)

            # Compute permutation importance for this class
            try:
                result = permutation_importance(
                    model,
                    X,
                    y_binary,
                    n_repeats=n_repeats,
                    random_state=random_state,
                    n_jobs=-1,
                    scoring="roc_auc"
                    if hasattr(model, "predict_proba")
                    else "accuracy",
                )

                class_importance = {}
                for name, importance in zip(feature_names, result.importances_mean):
                    class_importance[name] = float(importance)

                per_class_importance[class_name] = class_importance

                # Get top 5 features for this class
                sorted_features = sorted(
                    class_importance.items(), key=lambda x: x[1], reverse=True
                )[:5]
                top_features_per_class[class_name] = [
                    feature for feature, _ in sorted_features
                ]

            except:
                # If per-class importance fails, use global importance
                per_class_importance[class_name] = global_importance.copy()
                sorted_features = sorted(
                    global_importance.items(), key=lambda x: x[1], reverse=True
                )[:5]
                top_features_per_class[class_name] = [
                    feature for feature, _ in sorted_features
                ]

        # Find common important features across classes
        common_features: Dict[str, int] = defaultdict(int)
        for class_name, top_features in top_features_per_class.items():
            for feature in top_features[:3]:  # Top 3 per class
                common_features[feature] += 1

        # Features that appear in at least 3 classes' top features
        common_important_features = [
            feature for feature, count in common_features.items() if count >= 3
        ]

        return FeatureImportanceAnalysis(
            feature_names=feature_names,
            global_importance=global_importance,
            per_class_importance=per_class_importance,
            top_features_per_class=top_features_per_class,
            common_important_features=common_important_features,
        )

    def generate_insights(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model: Any,
        X: pd.DataFrame,
        sample_data: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None,
    ) -> RootCauseInsights:
        """
        Generate comprehensive insights and recommendations.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model: Trained model
            X: Feature matrix
            sample_data: Original sample data with log messages
            feature_names: Names of features

        Returns:
            RootCauseInsights object with all analysis
        """
        # Perform misclassification analysis
        misclassification_analysis = self.analyze_misclassifications(
            y_true, y_pred, X, sample_data, feature_names
        )

        # Perform feature importance analysis
        feature_importance_analysis = self.analyze_feature_importance(
            model, X, y_true, feature_names
        )

        # Calculate overall accuracy
        overall_accuracy = np.mean(y_true == y_pred)

        # Identify most challenging and easiest classes
        error_rates = misclassification_analysis.per_class_error_rate
        most_challenging_class = max(error_rates.items(), key=lambda x: x[1])[0]
        easiest_class = min(error_rates.items(), key=lambda x: x[1])[0]

        # Generate recommendations
        recommendations = self._generate_recommendations(
            misclassification_analysis, feature_importance_analysis, overall_accuracy
        )

        # Identify patterns
        identified_patterns = self._identify_patterns(
            misclassification_analysis, feature_importance_analysis
        )

        return RootCauseInsights(
            misclassification_analysis=misclassification_analysis,
            feature_importance_analysis=feature_importance_analysis,
            overall_accuracy=overall_accuracy,
            most_challenging_class=most_challenging_class,
            easiest_class=easiest_class,
            recommendations=recommendations,
            identified_patterns=identified_patterns,
        )

    def _generate_recommendations(
        self,
        misclassification_analysis: MisclassificationAnalysis,
        feature_importance_analysis: FeatureImportanceAnalysis,
        overall_accuracy: float,
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Recommendation based on overall accuracy
        if overall_accuracy < 0.7:
            recommendations.append(
                {
                    "type": "model_improvement",
                    "priority": "high",
                    "description": f"Overall accuracy ({overall_accuracy:.1%}) is below 70%. Consider collecting more training data or trying more complex models.",
                    "action": "Increase training data or experiment with ensemble methods",
                }
            )
        elif overall_accuracy < 0.85:
            recommendations.append(
                {
                    "type": "model_tuning",
                    "priority": "medium",
                    "description": f"Overall accuracy ({overall_accuracy:.1%}) has room for improvement. Focus on hyperparameter tuning.",
                    "action": "Perform grid search for optimal hyperparameters",
                }
            )

        # Recommendations for challenging classes
        error_rates = misclassification_analysis.per_class_error_rate
        for class_name, error_rate in error_rates.items():
            if error_rate > 0.3:  # High error rate
                recommendations.append(
                    {
                        "type": "class_specific",
                        "priority": "high",
                        "description": f"{class_name} has high error rate ({error_rate:.1%}). This class may need more samples or specialized features.",
                        "action": f"Collect more samples for {class_name} or create class-specific features",
                    }
                )
            elif error_rate > 0.15:  # Medium error rate
                recommendations.append(
                    {
                        "type": "class_specific",
                        "priority": "medium",
                        "description": f"{class_name} has moderate error rate ({error_rate:.1%}). Review misclassifications for patterns.",
                        "action": f"Analyze misclassified samples for {class_name} to identify confusion patterns",
                    }
                )

        # Recommendations based on feature importance
        common_features = feature_importance_analysis.common_important_features
        if common_features:
            recommendations.append(
                {
                    "type": "feature_engineering",
                    "priority": "medium",
                    "description": f"Common important features across classes: {', '.join(common_features[:3])}. Focus feature engineering on these.",
                    "action": "Create additional features based on these important patterns",
                }
            )

        # Recommendations for challenging pairs
        for pair in misclassification_analysis.challenging_pairs[:3]:  # Top 3 pairs
            recommendations.append(
                {
                    "type": "confusion_resolution",
                    "priority": "medium",
                    "description": pair["description"],
                    "action": f"Create features to distinguish between {pair['pair'].split(' → ')[0]} and {pair['pair'].split(' → ')[1]}",
                }
            )

        return recommendations

    def _identify_patterns(
        self,
        misclassification_analysis: MisclassificationAnalysis,
        feature_importance_analysis: FeatureImportanceAnalysis,
    ) -> List[Dict[str, Any]]:
        """Identify patterns in the analysis results."""
        patterns = []

        # Pattern: Classes with similar error rates
        error_rates = misclassification_analysis.per_class_error_rate
        avg_error_rate = np.mean(list(error_rates.values()))

        high_error_classes = [
            cls for cls, rate in error_rates.items() if rate > avg_error_rate * 1.5
        ]
        if high_error_classes:
            patterns.append(
                {
                    "type": "error_clustering",
                    "description": f"Classes with disproportionately high error rates: {', '.join(high_error_classes)}",
                    "implication": "These classes may have insufficient training data or ambiguous patterns",
                }
            )

        # Pattern: Asymmetric confusion
        for misclass in misclassification_analysis.top_misclassifications[:5]:
            true_class = misclass["true_class"]
            pred_class = misclass["predicted_class"]
            count = misclass["count"]

            # Check if reverse confusion exists
            reverse_exists = any(
                m["true_class"] == pred_class and m["predicted_class"] == true_class
                for m in misclassification_analysis.top_misclassifications
            )

            if not reverse_exists and count > 2:
                patterns.append(
                    {
                        "type": "asymmetric_confusion",
                        "description": f"{true_class} is frequently confused as {pred_class} ({count} times), but not vice versa",
                        "implication": f"{pred_class} patterns may be subset of {true_class} patterns",
                    }
                )

        # Pattern: Feature dominance
        global_importance = feature_importance_analysis.global_importance
        if global_importance:
            # Convert values to float for comparison
            importance_items = [(k, float(v)) for k, v in global_importance.items()]
            top_feature = max(importance_items, key=lambda x: x[1])
            if top_feature[1] > 0.3:  # Dominant feature
                patterns.append(
                    {
                        "type": "feature_dominance",
                        "description": f"Feature '{top_feature[0]}' dominates importance ({top_feature[1]:.1%})",
                        "implication": "Model may be over-relying on single feature, consider regularization",
                    }
                )

        return patterns
