import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import Tuple, Dict, Any, List, Optional
import logging
from scipy.sparse import hstack, csr_matrix

logger = logging.getLogger(__name__)


class LogFeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.service_encoder: Optional[LabelEncoder] = None
        self.severity_encoder: Optional[LabelEncoder] = None
        self.feature_names: List[str] = []

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess log message text."""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def extract_error_type(self, text: str) -> str:
        """Extract error type from log message."""
        error_patterns = [
            (r"timeout", "timeout"),
            (r"connection.*failed", "connection_failed"),
            (r"authentication.*failed", "authentication_failed"),
            (r"permission.*denied", "permission_denied"),
            (r"resource.*not.*found", "resource_not_found"),
            (r"invalid.*parameter", "invalid_parameter"),
            (r"rate.*limit", "rate_limit"),
            (r"memory.*error", "memory_error"),
            (r"disk.*full", "disk_full"),
            (r"network.*error", "network_error"),
        ]

        for pattern, error_name in error_patterns:
            if re.search(pattern, text.lower()):
                return error_name

        return "other_error"

    def extract_service_patterns(self, service: str, text: str) -> List[str]:
        """Extract service-specific patterns from log messages."""
        patterns = []

        # Service-specific keywords
        service_keywords = {
            "kyc-service": ["identity", "verification", "document", "check"],
            "ml-inference": ["model", "prediction", "inference", "tensor"],
            "oauth-handler": ["token", "authorization", "scope", "client"],
            "data-export": ["export", "download", "csv", "json"],
            "ingestion-service": ["ingest", "stream", "batch", "queue"],
            "data-pipeline": ["pipeline", "etl", "transform", "load"],
            "reporting-api": ["report", "analytics", "dashboard", "metric"],
            "transaction-validator": ["transaction", "validate", "amount", "currency"],
            "db-pool": ["database", "connection", "query", "pool"],
            "payments-core": ["payment", "charge", "refund", "settlement"],
        }

        if service in service_keywords:
            for keyword in service_keywords[service]:
                if keyword in text.lower():
                    patterns.append(f"{service}_{keyword}")

        return patterns

    def create_text_features(self) -> pd.DataFrame:
        """Create text-based features from log messages."""
        logger.info("Creating text features from log messages")

        # Preprocess log messages
        self.df["clean_message"] = self.df["log_message"].apply(self.preprocess_text)

        # Extract error types
        self.df["error_type"] = self.df["log_message"].apply(self.extract_error_type)

        # Extract service patterns
        self.df["service_patterns"] = self.df.apply(
            lambda row: self.extract_service_patterns(
                row["service"], row["log_message"]
            ),
            axis=1,
        )

        # Create combined text feature for TF-IDF
        self.df["combined_text"] = self.df.apply(
            lambda row: f"{row['clean_message']} {' '.join(row['service_patterns'])} {row['error_type']}",
            axis=1,
        )

        return self.df

    def create_tfidf_features(
        self, max_features: int = 1000, min_df: int = 1, max_df: float = 1.0
    ) -> np.ndarray:
        """Create TF-IDF features from combined text."""
        logger.info(
            f"Creating TF-IDF features with max_features={max_features}, min_df={min_df}, max_df={max_df}"
        )

        if "combined_text" not in self.df.columns:
            self.create_text_features()

        # Initialize TF-IDF vectorizer
        # For small datasets, use max_df=1.0 to avoid pruning all terms
        # (default max_df=0.8 would remove terms appearing in >80% of documents)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=min_df,  # Ignore terms that appear in less than min_df documents
            max_df=max_df,  # For small datasets, use 1.0 to keep all terms
            stop_words="english",
        )

        # Fit and transform
        tfidf_features = self.tfidf_vectorizer.fit_transform(self.df["combined_text"])

        # Get feature names
        self.feature_names.extend(self.tfidf_vectorizer.get_feature_names_out())

        logger.info(f"Created TF-IDF matrix with shape: {tfidf_features.shape}")

        return tfidf_features

    def create_categorical_features(self) -> np.ndarray:
        """Create categorical features from service and severity."""
        logger.info("Creating categorical features")

        # Encode service
        self.service_encoder = LabelEncoder()
        service_encoded = self.service_encoder.fit_transform(self.df["service"])

        # Encode severity
        severity_mapping = {"INFO": 0, "WARNING": 1, "ERROR": 2, "CRITICAL": 3}
        severity_encoded = (
            self.df["severity"].map(severity_mapping).fillna(0).astype(int)
        )

        # Encode error type
        error_type_encoder = LabelEncoder()
        error_type_encoded = error_type_encoder.fit_transform(self.df["error_type"])

        # Stack categorical features
        categorical_features = np.column_stack(
            [service_encoded, severity_encoded, error_type_encoded]
        )

        # Add feature names
        self.feature_names.extend(
            ["service_encoded", "severity_encoded", "error_type_encoded"]
        )

        logger.info(
            f"Created categorical features with shape: {categorical_features.shape}"
        )

        return categorical_features

    def extract_timestamp_features(self) -> np.ndarray:
        """Extract features from timestamp."""
        logger.info("Extracting timestamp features")

        # Convert timestamp to datetime
        self.df["timestamp_dt"] = pd.to_datetime(self.df["timestamp"], errors="coerce")

        # Extract time-based features
        timestamp_features: np.ndarray

        if self.df["timestamp_dt"].notna().any():
            # Hour of day
            hour_of_day = self.df["timestamp_dt"].dt.hour.fillna(0).astype(int)

            # Day of week
            day_of_week = self.df["timestamp_dt"].dt.dayofweek.fillna(0).astype(int)

            # Is weekend
            is_weekend = (day_of_week >= 5).astype(int)

            # Is business hours (9am-5pm)
            is_business_hours = ((hour_of_day >= 9) & (hour_of_day <= 17)).astype(int)

            timestamp_features = np.column_stack(
                [hour_of_day, day_of_week, is_weekend, is_business_hours]
            )

            # Add feature names
            self.feature_names.extend(
                ["hour_of_day", "day_of_week", "is_weekend", "is_business_hours"]
            )
        else:
            # Create dummy features if timestamp parsing fails
            timestamp_features = np.zeros((len(self.df), 4))

        logger.info(
            f"Created timestamp features with shape: {timestamp_features.shape}"
        )

        return timestamp_features

    def create_all_features(
        self,
        tfidf_max_features: int = 1000,
        tfidf_min_df: int = 1,
        tfidf_max_df: float = 1.0,
    ) -> Tuple[np.ndarray, List[str]]:
        """Create all features and combine them."""
        logger.info("Creating all features")

        # Reset feature names
        self.feature_names = []

        # Create text features
        self.create_text_features()

        # Create TF-IDF features
        # For small datasets, use max_df=1.0 by default to avoid pruning all terms
        tfidf_features = self.create_tfidf_features(
            max_features=tfidf_max_features, min_df=tfidf_min_df, max_df=tfidf_max_df
        )

        # Create categorical features
        categorical_features = self.create_categorical_features()

        # Create timestamp features
        timestamp_features = self.extract_timestamp_features()

        # Combine all features
        if tfidf_features.shape[0] > 0 and tfidf_features.shape[1] > 0:
            # Combine sparse TF-IDF with dense features
            # Convert categorical and timestamp features to sparse matrices
            categorical_sparse = csr_matrix(categorical_features)
            timestamp_sparse = csr_matrix(timestamp_features)
            all_features = hstack([tfidf_features, categorical_sparse, timestamp_sparse])
        else:
            all_features = np.hstack([categorical_features, timestamp_features])

        logger.info(f"Created all features with shape: {all_features.shape}")
        logger.info(f"Total feature names: {len(self.feature_names)}")

        return all_features, self.feature_names

    def prepare_labels(self) -> np.ndarray:
        """Prepare labels for classification."""
        logger.info("Preparing labels for classification")

        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(self.df["root_cause_label"])

        logger.info(
            f"Encoded {len(self.label_encoder.classes_)} classes: {self.label_encoder.classes_}"
        )

        return labels

    def split_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets with stratification."""
        logger.info(
            f"Splitting data with test_size={test_size}, random_state={random_state}"
        )

        # Convert labels to array for stratification
        labels_array = np.array(labels)

        # Perform stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels_array,
            test_size=test_size,
            random_state=random_state,
            stratify=labels_array,
        )

        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test

    def get_feature_analysis(self) -> Dict[str, Any]:
        """Get analysis of created features."""
        analysis = {
            "text_features": {
                "unique_error_types": self.df["error_type"].nunique()
                if "error_type" in self.df.columns
                else 0,
                "avg_service_patterns": self.df["service_patterns"].apply(len).mean()
                if "service_patterns" in self.df.columns
                else 0,
            },
            "categorical_features": {
                "unique_services": self.df["service"].nunique(),
                "unique_severities": self.df["severity"].nunique(),
            },
            "timestamp_features": {
                "has_timestamp": "timestamp_dt" in self.df.columns
                and self.df["timestamp_dt"].notna().any(),
            },
            "feature_summary": {
                "total_samples": len(self.df),
                "total_features": len(self.feature_names)
                if hasattr(self, "feature_names")
                else 0,
            },
        }

        return analysis
