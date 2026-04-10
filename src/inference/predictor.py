"""
Inference pipeline for root cause prediction.
Loads trained model and preprocessing pipeline for predictions on new log entries.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import joblib
import json
from pathlib import Path
import logging

from sklearn.base import BaseEstimator

from src.inference.summary import SummaryGenerator, RootCauseSummary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Structured prediction result with confidence scores and summary."""

    root_cause: str
    confidence: float
    top_n_predictions: List[Tuple[str, float]]
    features_used: Dict[str, Any]
    summary: Optional[RootCauseSummary] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result_dict = {
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "top_n_predictions": [
                {"root_cause": rc, "confidence": conf}
                for rc, conf in self.top_n_predictions
            ],
            "features_used": self.features_used,
        }

        # Include summary if available
        if self.summary:
            result_dict["summary"] = self.summary.to_dict()

        return result_dict

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def generate_summary(
        self,
        log_message: str,
        service: str,
        severity: str,
        timestamp: Optional[str] = None,
        summary_generator: Optional[SummaryGenerator] = None,
    ) -> "PredictionResult":
        """
        Generate summary for this prediction result.

        Args:
            log_message: The log message text
            service: Service name
            severity: Severity level
            timestamp: Optional timestamp
            summary_generator: Optional SummaryGenerator instance

        Returns:
            Self with summary added (for method chaining)
        """
        if summary_generator is None:
            summary_generator = SummaryGenerator()

        self.summary = summary_generator.generate_summary(
            root_cause=self.root_cause,
            confidence=self.confidence,
            log_message=log_message,
            service=service,
            severity=severity,
            timestamp=timestamp,
        )
        return self


class LogPredictor:
    """
    Inference pipeline for predicting root causes from log entries.

    Loads trained model and preprocessing pipeline to make predictions
    on new log entries with confidence scores.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        feature_engineer_path: Optional[Union[str, Path]] = None,
        dataset_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize predictor with trained model and feature engineer.

        Args:
            model_path: Path to trained model (.joblib file)
            feature_engineer_path: Path to saved feature engineer (.joblib file)
                                   If None, will try to create from dataset
            dataset_path: Path to dataset CSV (required if feature_engineer_path is None)
        """
        self.model_path = Path(model_path)
        self.feature_engineer_path = (
            Path(feature_engineer_path) if feature_engineer_path else None
        )
        self.dataset_path = Path(dataset_path) if dataset_path else None

        # Load model and feature engineer
        self.model: BaseEstimator = None
        self.feature_engineer: Any = None
        self.root_cause_labels: List[str] = []
        self.summary_generator: SummaryGenerator = SummaryGenerator()

        self._load_model_and_features()

        logger.info(f"Loaded predictor from {model_path}")

    def _load_model_and_features(self) -> None:
        """Load trained model and feature engineering pipeline."""
        try:
            # Load model
            model_data = joblib.load(self.model_path)

            # Check if model data includes feature engineer
            if isinstance(model_data, dict) and "model" in model_data:
                self.model = model_data["model"]
                if "feature_engineer" in model_data:
                    self.feature_engineer = model_data["feature_engineer"]
                if "classes" in model_data and model_data["classes"] is not None:
                    self.root_cause_labels = model_data["classes"]
            else:
                # Assume it's just the model
                self.model = model_data

            # Load feature engineer components if available in model directory
            model_dir = self.model_path.parent
            if not self.feature_engineer:
                # Try to load individual components
                self.feature_engineer = self._load_feature_engineer_components(
                    model_dir
                )

            # Load feature engineer separately if provided
            if self.feature_engineer_path and not self.feature_engineer:
                self.feature_engineer = joblib.load(self.feature_engineer_path)

            # If feature engineer still not loaded and dataset path provided, create it
            if not self.feature_engineer and self.dataset_path:
                logger.info(
                    f"Creating feature engineer from dataset: {self.dataset_path}"
                )
                self._create_feature_engineer_from_dataset()

            # If root cause labels not loaded, use default RC-01 to RC-08
            if not self.root_cause_labels:
                self.root_cause_labels = [f"RC-{i:02d}" for i in range(1, 9)]
            elif isinstance(self.root_cause_labels[0], (int, np.integer)):
                # Convert numeric labels to RC format
                self.root_cause_labels = [
                    f"RC-{i + 1:02d}" for i in self.root_cause_labels
                ]

            logger.info(f"Model type: {type(self.model).__name__}")
            logger.info(f"Feature engineer loaded: {self.feature_engineer is not None}")
            logger.info(f"Root cause labels: {self.root_cause_labels}")

        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise

    def _load_feature_engineer_components(self, model_dir: Path) -> Any:
        """Load feature engineer components from individual files."""
        try:
            from src.data.features import LogFeatureEngineer
            import pandas as pd

            # Create a dummy feature engineer
            dummy_df = pd.DataFrame(
                {
                    "log_message": ["dummy"],
                    "service": ["dummy"],
                    "severity": ["INFO"],
                    "timestamp": ["2024-01-01 00:00:00"],
                    "root_cause_label": ["RC-01"],
                }
            )

            feature_engineer = LogFeatureEngineer(dummy_df)

            # Load TF-IDF vectorizer
            tfidf_path = model_dir / "tfidf_vectorizer.pkl"
            if tfidf_path.exists():
                feature_engineer.tfidf_vectorizer = joblib.load(tfidf_path)
                logger.info(f"Loaded TF-IDF vectorizer from {tfidf_path}")

            # Load label encoder
            label_encoder_path = model_dir / "label_encoder.pkl"
            if label_encoder_path.exists():
                feature_engineer.label_encoder = joblib.load(label_encoder_path)
                logger.info(f"Loaded label encoder from {label_encoder_path}")

            # Load service encoder
            service_encoder_path = model_dir / "service_encoder.pkl"
            if service_encoder_path.exists():
                feature_engineer.service_encoder = joblib.load(service_encoder_path)
                logger.info(f"Loaded service encoder from {service_encoder_path}")

            # Load severity encoder (if exists)
            severity_encoder_path = model_dir / "severity_encoder.pkl"
            if severity_encoder_path.exists():
                feature_engineer.severity_encoder = joblib.load(severity_encoder_path)
                logger.info(f"Loaded severity encoder from {severity_encoder_path}")

            return feature_engineer

        except Exception as e:
            logger.warning(f"Failed to load feature engineer components: {e}")
            return None

    def _create_feature_engineer_from_dataset(self) -> None:
        """Create feature engineer from dataset."""
        try:
            from src.data.loader import LogDataLoader
            from src.data.features import LogFeatureEngineer

            # Load dataset
            loader = LogDataLoader(str(self.dataset_path))
            df = loader.load_data()

            # Create feature engineer with same parameters as training
            self.feature_engineer = LogFeatureEngineer(df)

            # Create features to fit the vectorizers
            features, feature_names = self.feature_engineer.create_all_features(
                tfidf_max_features=100,
                tfidf_min_df=1,
                tfidf_max_df=1.0,
            )

            logger.info(f"Created feature engineer from {len(df)} samples")
            logger.info(f"Feature names: {len(feature_names)}")

        except Exception as e:
            logger.error(f"Failed to create feature engineer from dataset: {e}")
            raise

    def predict_single(
        self,
        log_message: str,
        service: str,
        severity: str,
        timestamp: Optional[str] = None,
        top_n: int = 3,
    ) -> PredictionResult:
        """
        Predict root cause for a single log entry.

        Args:
            log_message: The log message text
            service: Service name
            severity: Severity level
            timestamp: Optional timestamp string
            top_n: Number of top predictions to return

        Returns:
            PredictionResult with root cause, confidence, and top predictions
        """
        # Create single row DataFrame
        data = pd.DataFrame(
            [
                {
                    "log_message": log_message,
                    "service": service,
                    "severity": severity,
                    "timestamp": timestamp
                    if timestamp
                    else pd.Timestamp.now().isoformat(),
                }
            ]
        )

        # Get predictions
        predictions = self._predict_batch(data, top_n)

        # Return first (and only) prediction
        return predictions[0]

    def predict_batch(
        self,
        log_data: pd.DataFrame,
        top_n: int = 3,
    ) -> List[PredictionResult]:
        """
        Predict root causes for a batch of log entries.

        Args:
            log_data: DataFrame with columns: log_message, service, severity, timestamp
            top_n: Number of top predictions to return per entry

        Returns:
            List of PredictionResult objects
        """
        return self._predict_batch(log_data, top_n)

    def _predict_batch(
        self,
        log_data: pd.DataFrame,
        top_n: int,
    ) -> List[PredictionResult]:
        """
        Internal batch prediction method.

        Args:
            log_data: DataFrame with log entries
            top_n: Number of top predictions to return

        Returns:
            List of PredictionResult objects
        """
        try:
            # Validate input data
            self._validate_input_data(log_data)

            # Extract features using feature engineer
            if self.feature_engineer:
                # For now, create a simple transform method
                features = self._transform_features(log_data)
            else:
                # If no feature engineer, use raw data (model should handle it)
                features = log_data

            # Get predictions and probabilities
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features)
                predictions = self.model.predict(features)
            else:
                # For models without predict_proba, use decision function or just predictions
                predictions = self.model.predict(features)
                probabilities = np.zeros(
                    (len(predictions), len(self.root_cause_labels))
                )
                for i, pred in enumerate(predictions):
                    pred_idx = (
                        self.root_cause_labels.index(pred)
                        if pred in self.root_cause_labels
                        else 0
                    )
                    probabilities[i, pred_idx] = 1.0

            # Create prediction results
            results = []
            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                # Get top N predictions with confidence scores
                top_indices = np.argsort(probs)[-top_n:][::-1]
                top_predictions = []
                for idx in top_indices:
                    if 0 <= idx < len(self.root_cause_labels):
                        label = self.root_cause_labels[idx]
                    else:
                        label = str(idx)
                    top_predictions.append((label, float(probs[idx])))

                # Get prediction confidence
                pred_idx = (
                    self.root_cause_labels.index(pred)
                    if pred in self.root_cause_labels
                    else 0
                )
                confidence = float(probs[pred_idx])

                # Get features used for this prediction
                features_used = self._extract_features_used(log_data.iloc[i])

                # Create prediction result
                # Convert numeric prediction to RC label
                if isinstance(pred, (int, np.integer)) and 0 <= pred < len(
                    self.root_cause_labels
                ):
                    root_cause_label = self.root_cause_labels[pred]
                elif str(pred) in self.root_cause_labels:
                    root_cause_label = str(pred)
                else:
                    root_cause_label = str(pred)

                prediction_result = PredictionResult(
                    root_cause=root_cause_label,
                    confidence=confidence,
                    top_n_predictions=top_predictions,
                    features_used=features_used,
                )

                # Generate summary for the prediction
                log_entry = log_data.iloc[i]
                prediction_result.generate_summary(
                    log_message=log_entry.get("log_message", ""),
                    service=log_entry.get("service", ""),
                    severity=log_entry.get("severity", ""),
                    timestamp=log_entry.get("timestamp"),
                    summary_generator=self.summary_generator,
                )

                results.append(prediction_result)

            return results

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def _validate_input_data(self, log_data: pd.DataFrame) -> None:
        """
        Validate input data structure and content.

        Args:
            log_data: DataFrame to validate

        Raises:
            ValueError: If data is invalid
        """
        required_columns = {"log_message", "service", "severity"}

        # Check required columns
        missing_columns = required_columns - set(log_data.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Required columns: {required_columns}"
            )

        # Check for empty log messages
        empty_messages = log_data["log_message"].isna() | (
            log_data["log_message"] == ""
        )
        if empty_messages.any():
            raise ValueError(
                f"Found {empty_messages.sum()} empty log messages. "
                "Log messages cannot be empty."
            )

        # Check data types
        if not all(isinstance(msg, str) for msg in log_data["log_message"].dropna()):
            raise ValueError("log_message column must contain strings")

        logger.info(f"Validated {len(log_data)} log entries")

    def _extract_features_used(self, log_entry: pd.Series) -> Dict[str, Any]:
        """
        Extract relevant features from log entry for explanation.

        Args:
            log_entry: Single log entry as Series

        Returns:
            Dictionary of features used
        """
        features = {
            "log_message": log_entry.get("log_message", ""),
            "service": log_entry.get("service", ""),
            "severity": log_entry.get("severity", ""),
            "timestamp": log_entry.get("timestamp", ""),
        }

        # Clean up for JSON serialization
        for key, value in features.items():
            if pd.isna(value):
                features[key] = ""
            elif isinstance(value, (pd.Timestamp, np.datetime64)):
                features[key] = str(value)

        return features

    def _transform_features(self, log_data: pd.DataFrame) -> np.ndarray:
        """
        Transform new log data into features compatible with trained model.

        Uses the same feature engineering pipeline as training.

        Args:
            log_data: DataFrame with log entries

        Returns:
            Feature matrix
        """
        try:
            if not self.feature_engineer:
                logger.error("No feature engineer available for transformation")
                raise ValueError("Feature engineer not loaded")

            # Create a copy of the input data
            df = log_data.copy()

            # Apply the same preprocessing steps as during training
            # Preprocess log messages
            df["clean_message"] = df["log_message"].apply(
                self.feature_engineer.preprocess_text
            )

            # Extract error types
            df["error_type"] = df["log_message"].apply(
                self.feature_engineer.extract_error_type
            )

            # Extract service patterns
            df["service_patterns"] = df.apply(
                lambda row: self.feature_engineer.extract_service_patterns(
                    row["service"], row["log_message"]
                ),
                axis=1,
            )

            # Create combined text feature for TF-IDF
            df["combined_text"] = df.apply(
                lambda row: f"{row['clean_message']} {' '.join(row['service_patterns']) if isinstance(row['service_patterns'], list) else ''} {row['error_type']}",
                axis=1,
            )

            # Transform text using trained TF-IDF vectorizer
            if self.feature_engineer.tfidf_vectorizer:
                tfidf_features = self.feature_engineer.tfidf_vectorizer.transform(
                    df["combined_text"]
                )
            else:
                logger.warning("No TF-IDF vectorizer available, using empty features")
                tfidf_features = np.zeros((len(df), 0))

            # Transform categorical features
            categorical_features = []

            # Encode service
            if self.feature_engineer.service_encoder:
                service_encoded = self.feature_engineer.service_encoder.transform(
                    df["service"]
                )
            else:
                # Fallback: map to integers
                service_encoded = pd.factorize(df["service"])[0]

            categorical_features.append(service_encoded)

            # Encode severity
            severity_mapping = {"INFO": 0, "WARNING": 1, "ERROR": 2, "CRITICAL": 3}
            severity_encoded = (
                df["severity"].map(severity_mapping).fillna(0).astype(int)
            )
            categorical_features.append(severity_encoded)

            # Encode error type
            if hasattr(self.feature_engineer, "label_encoder"):
                error_type_encoded = pd.factorize(df["error_type"])[0]
            else:
                error_type_encoded = pd.factorize(df["error_type"])[0]
            categorical_features.append(error_type_encoded)

            # Extract timestamp features
            timestamp_features = []
            try:
                df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")

                # Hour of day
                hour_of_day = df["timestamp_dt"].dt.hour.fillna(0).astype(int)
                timestamp_features.append(hour_of_day)

                # Day of week
                day_of_week = df["timestamp_dt"].dt.dayofweek.fillna(0).astype(int)
                timestamp_features.append(day_of_week)

                # Is weekend
                is_weekend = (day_of_week >= 5).astype(int)
                timestamp_features.append(is_weekend)

                # Is business hours (9am-5pm)
                is_business_hours = ((hour_of_day >= 9) & (hour_of_day <= 17)).astype(
                    int
                )
                timestamp_features.append(is_business_hours)
            except Exception as e:
                logger.warning(f"Failed to extract timestamp features: {e}")
                # Add dummy timestamp features
                for _ in range(4):
                    timestamp_features.append(np.zeros(len(df)))

            # Combine all features
            from scipy.sparse import hstack

            # Convert categorical and timestamp features to arrays
            cat_array = (
                np.column_stack(categorical_features)
                if categorical_features
                else np.zeros((len(df), 0))
            )
            time_array = (
                np.column_stack(timestamp_features)
                if timestamp_features
                else np.zeros((len(df), 0))
            )

            # Combine sparse TF-IDF with dense features
            if tfidf_features.shape[1] > 0:
                all_features = hstack([tfidf_features, cat_array, time_array])
            else:
                all_features = np.hstack([cat_array, time_array])

            logger.info(
                f"Transformed {len(df)} samples into features with shape: {all_features.shape}"
            )
            return all_features

        except Exception as e:
            logger.error(f"Feature transformation failed: {e}")
            # Return dummy features as fallback with correct shape
            if hasattr(self.model, "feature_names_in_"):
                n_features = len(self.model.feature_names_in_)
            else:
                n_features = 107
            return np.zeros((len(log_data), n_features))

    def save(self, output_path: Union[str, Path]) -> None:
        """
        Save predictor state for later use.

        Args:
            output_path: Path to save predictor
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "model": self.model,
            "feature_engineer": self.feature_engineer,
            "root_cause_labels": self.root_cause_labels,
            "model_path": str(self.model_path),
            "summary_generator": self.summary_generator,
        }

        joblib.dump(save_data, output_path)
        logger.info(f"Saved predictor to {output_path}")

    @classmethod
    def load(cls, predictor_path: Union[str, Path]) -> "LogPredictor":
        """
        Load saved predictor.

        Args:
            predictor_path: Path to saved predictor

        Returns:
            Loaded LogPredictor instance
        """
        predictor_path = Path(predictor_path)

        # Load predictor data
        predictor_data = joblib.load(predictor_path)

        # Create new predictor instance
        predictor = cls.__new__(cls)

        # Set attributes from saved data
        predictor.model = predictor_data["model"]
        predictor.feature_engineer = predictor_data["feature_engineer"]
        predictor.root_cause_labels = predictor_data["root_cause_labels"]
        predictor.model_path = Path(predictor_data["model_path"])
        predictor.feature_engineer_path = None
        predictor.summary_generator = predictor_data.get(
            "summary_generator", SummaryGenerator()
        )

        logger.info(f"Loaded predictor from {predictor_path}")
        return predictor
