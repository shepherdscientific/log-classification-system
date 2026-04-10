"""
Multi-class classification model for root cause prediction.
Supports Logistic Regression, Random Forest, and XGBoost with class imbalance handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import joblib
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


@dataclass
class ModelConfig:
    """Configuration for classification model."""

    model_type: str = (
        "random_forest"  # "logistic_regression", "random_forest", "xgboost"
    )
    random_state: int = 42
    class_weight: Optional[str] = (
        "balanced"  # None, "balanced", or "balanced_subsample"
    )

    # Logistic Regression parameters
    logreg_c: float = 1.0
    logreg_max_iter: int = 1000
    logreg_solver: str = "lbfgs"

    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1

    # XGBoost parameters (if available)
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1

    # Cross-validation parameters
    cv_folds: int = 5
    cv_scoring: str = "f1_macro"

    # Hyperparameter tuning
    tune_hyperparameters: bool = True
    n_jobs: int = -1


class LogClassifier:
    """
    Multi-class classifier for root cause prediction.
    Handles 8-class classification with class imbalance.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize classifier with configuration.

        Args:
            config: Model configuration. If None, uses default config.
        """
        self.config = config or ModelConfig()
        self.model: Optional[Any] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.cv_results_: Optional[Dict[str, Any]] = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    def _create_model(self) -> Any:
        """Create model instance based on configuration."""
        if self.config.model_type == "logistic_regression":
            return LogisticRegression(
                C=self.config.logreg_c,
                max_iter=self.config.logreg_max_iter,
                solver=self.config.logreg_solver,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )
        elif self.config.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                min_samples_leaf=self.config.rf_min_samples_leaf,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )
        elif self.config.model_type == "xgboost":
            try:
                import xgboost as xgb

                return xgb.XGBClassifier(
                    n_estimators=self.config.xgb_n_estimators,
                    max_depth=self.config.xgb_max_depth,
                    learning_rate=self.config.xgb_learning_rate,
                    objective="multi:softprob",
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                )
            except ImportError:
                raise ImportError(
                    "XGBoost not installed. Install with: pip install xgboost"
                )
        else:
            raise ValueError(
                f"Unknown model type: {self.config.model_type}. "
                "Supported: 'logistic_regression', 'random_forest', 'xgboost'"
            )

    def _get_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        """Get hyperparameter grid for tuning based on model type."""
        if self.config.model_type == "logistic_regression":
            return {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"],
                "class_weight": [None, "balanced"],
            }
        elif self.config.model_type == "random_forest":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
                "class_weight": [None, "balanced", "balanced_subsample"],
            }
        elif self.config.model_type == "xgboost":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.8, 1.0],
            }
        else:
            return {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "LogClassifier":
        """
        Train classification model with optional hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Names of features (optional)

        Returns:
            self: Trained classifier
        """
        self.classes_ = np.unique(y_train)
        self.feature_names_ = feature_names

        # Create base model
        base_model = self._create_model()

        # Handle class imbalance - compute class weights if needed
        if self.config.class_weight == "balanced":
            # Compute class weights based on training data distribution
            from sklearn.utils.class_weight import compute_class_weight

            class_weights = compute_class_weight(
                class_weight="balanced", classes=np.unique(y_train), y=y_train
            )
            # Update model parameters with computed weights
            if hasattr(base_model, "set_params"):
                base_model.set_params(
                    class_weight={i: w for i, w in enumerate(class_weights)}
                )

        # For small datasets, adjust CV strategy
        n_classes = len(np.unique(y_train))
        n_samples = len(y_train)

        # Determine appropriate number of CV folds
        # Ensure at least 2 samples per class in each fold
        max_folds = min(self.config.cv_folds, n_samples // (2 * n_classes))
        if max_folds < 2:
            max_folds = 2

        # Hyperparameter tuning
        if self.config.tune_hyperparameters:
            param_grid = self._get_hyperparameter_grid()

            # Use appropriate CV for small dataset
            cv = StratifiedKFold(
                n_splits=max_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )

            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring=self.config.cv_scoring,
                n_jobs=self.config.n_jobs,
                verbose=1,
            )

            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            self.best_params_ = grid_search.best_params_
            self.cv_results_ = {
                "best_score": grid_search.best_score_,
                "best_params": grid_search.best_params_,
                "cv_results": grid_search.cv_results_,
            }
        else:
            # Train without tuning
            self.model = base_model
            if self.model is not None:
                self.model.fit(X_train, y_train)

        # Optional validation
        if X_val is not None and y_val is not None and self.model is not None:
            self.validation_score_ = self.model.score(X_val, y_val)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict root cause labels.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict root cause probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability matrix (n_samples × n_classes)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray, return_report: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test labels
            return_report: Whether to return classification report

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Calculate metrics
        accuracy = self.model.score(X_test, y_test)
        conf_matrix = confusion_matrix(y_test, y_pred)

        metrics = {
            "accuracy": float(accuracy),
            "confusion_matrix": conf_matrix.tolist(),
            "predictions": y_pred.tolist(),
            "probabilities": y_proba.tolist() if y_proba is not None else None,
        }

        if return_report:
            # Get unique classes in test set
            test_classes = np.unique(y_test)
            target_names = [f"RC-{i + 1:02d}" for i in test_classes]

            report = classification_report(
                y_test, y_pred, target_names=target_names, output_dict=True
            )
            metrics["classification_report"] = report

        return metrics

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if available.

        Returns:
            Dictionary of feature names to importance scores, or None if not available
        """
        if self.model is None:
            return None

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            # For logistic regression, use absolute coefficients averaged across classes
            if len(self.model.coef_.shape) > 1:
                importances = np.mean(np.abs(self.model.coef_), axis=0)
            else:
                importances = np.abs(self.model.coef_)
        else:
            return None

        if self.feature_names_ is not None and len(self.feature_names_) == len(
            importances
        ):
            return dict(zip(self.feature_names_, importances))
        else:
            return {"feature_importance": importances.tolist()}

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save trained model to file.

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained. Cannot save.")

        model_data = {
            "model": self.model,
            "config": self.config,
            "best_params": self.best_params_,
            "cv_results": self.cv_results_,
            "classes": self.classes_.tolist() if self.classes_ is not None else None,
            "feature_names": self.feature_names_,
        }

        joblib.dump(model_data, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LogClassifier":
        """
        Load trained model from file.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded classifier
        """
        model_data = joblib.load(filepath)

        classifier = cls(config=model_data["config"])
        classifier.model = model_data["model"]
        classifier.best_params_ = model_data["best_params"]
        classifier.cv_results_ = model_data["cv_results"]
        classifier.classes_ = (
            np.array(model_data["classes"])
            if model_data["classes"] is not None
            else None
        )
        classifier.feature_names_ = model_data["feature_names"]

        return classifier

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            "model_type": self.config.model_type,
            "random_state": self.config.random_state,
            "class_weight": self.config.class_weight,
            "cv_folds": self.config.cv_folds,
            "tune_hyperparameters": self.config.tune_hyperparameters,
            "best_params": self.best_params_,
            "classes": self.classes_.tolist() if self.classes_ is not None else None,
            "n_classes": len(self.classes_) if self.classes_ is not None else 0,
        }
