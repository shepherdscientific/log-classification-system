import pytest
import pandas as pd
import numpy as np
from src.data.features import LogFeatureEngineer
from unittest.mock import patch


@pytest.fixture
def sample_log_data():
    """Create sample log data for testing."""
    data = {
        "log_id": ["log_001", "log_002", "log_003", "log_004", "log_005"],
        "timestamp": [
            "2024-01-01 10:30:00",
            "2024-01-01 14:45:00",
            "2024-01-02 09:15:00",
            "2024-01-02 18:30:00",
            "2024-01-03 12:00:00",
        ],
        "service": [
            "kyc-service",
            "ml-inference",
            "oauth-handler",
            "data-export",
            "payments-core",
        ],
        "severity": ["ERROR", "WARNING", "ERROR", "INFO", "CRITICAL"],
        "log_message": [
            "Authentication failed for user admin: invalid credentials",
            "Model inference timeout after 30 seconds",
            "Token validation error: expired JWT token",
            "Data export completed successfully",
            "Payment gateway connection failed: upstream provider timeout",
        ],
        "root_cause_label": ["RC-01", "RC-02", "RC-03", "RC-04", "RC-05"],
    }
    return pd.DataFrame(data)


def test_preprocess_text():
    """Test text preprocessing function."""
    engineer = LogFeatureEngineer(pd.DataFrame())

    # Test basic preprocessing
    text = "ERROR: Connection timeout after 30s (retry #3)"
    result = engineer.preprocess_text(text)
    expected = "error connection timeout after s retry"
    assert result == expected

    # Test with numbers and special characters
    text = "API call failed with status 500: 'Internal Server Error'"
    result = engineer.preprocess_text(text)
    expected = "api call failed with status internal server error"
    assert result == expected

    # Test empty string
    result = engineer.preprocess_text("")
    assert result == ""

    # Test None input
    result = engineer.preprocess_text(None)
    assert result == ""


def test_extract_error_type():
    """Test error type extraction."""
    engineer = LogFeatureEngineer(pd.DataFrame())

    # Test timeout error
    text = "Request timeout after 60 seconds"
    result = engineer.extract_error_type(text)
    assert result == "timeout"

    # Test authentication error
    text = "Authentication failed: invalid password"
    result = engineer.extract_error_type(text)
    assert result == "authentication_failed"

    # Test other error
    text = "Some random error message"
    result = engineer.extract_error_type(text)
    assert result == "other_error"


def test_extract_service_patterns():
    """Test service pattern extraction."""
    engineer = LogFeatureEngineer(pd.DataFrame())

    # Test kyc-service patterns
    text = "Identity verification failed for document upload"
    result = engineer.extract_service_patterns("kyc-service", text)
    assert "kyc-service_identity" in result
    assert "kyc-service_verification" in result
    assert "kyc-service_document" in result

    # Test ml-inference patterns
    text = "Model prediction failed due to tensor shape mismatch"
    result = engineer.extract_service_patterns("ml-inference", text)
    assert "ml-inference_model" in result
    assert "ml-inference_prediction" in result
    # Note: 'inference' is not in the text, but 'tensor' is in the service keywords
    assert "ml-inference_tensor" in result

    # Test unknown service
    text = "Some error message"
    result = engineer.extract_service_patterns("unknown-service", text)
    assert result == []


def test_create_text_features(sample_log_data):
    """Test text feature creation."""
    engineer = LogFeatureEngineer(sample_log_data)
    df_with_features = engineer.create_text_features()

    # Check that new columns are created
    assert "clean_message" in df_with_features.columns
    assert "error_type" in df_with_features.columns
    assert "service_patterns" in df_with_features.columns
    assert "combined_text" in df_with_features.columns

    # Check clean_message preprocessing
    clean_messages = df_with_features["clean_message"].tolist()
    assert all(isinstance(msg, str) for msg in clean_messages)
    # Check that messages are cleaned (lowercase, no special chars)
    for msg in clean_messages:
        assert msg == msg.lower()
        # Should not contain common special characters (simplified check)
        assert not any(char in msg for char in ["(", ")", ":", "'", '"', "#"])

    # Check error type extraction
    error_types = df_with_features["error_type"].tolist()
    assert "authentication_failed" in error_types
    assert "timeout" in error_types
    assert "other_error" in error_types


def test_create_tfidf_features(sample_log_data):
    """Test TF-IDF feature creation."""
    engineer = LogFeatureEngineer(sample_log_data)

    # Create text features first
    engineer.create_text_features()

    # Create TF-IDF features
    tfidf_features = engineer.create_tfidf_features(max_features=50)

    # Check shape
    assert tfidf_features.shape[0] == len(sample_log_data)  # Same number of samples
    assert tfidf_features.shape[1] <= 50  # At most max_features

    # Check that vectorizer is created
    assert engineer.tfidf_vectorizer is not None
    assert hasattr(engineer.tfidf_vectorizer, "get_feature_names_out")

    # Check feature names
    feature_names = engineer.tfidf_vectorizer.get_feature_names_out()
    assert len(feature_names) == tfidf_features.shape[1]


def test_create_categorical_features(sample_log_data):
    """Test categorical feature creation."""
    engineer = LogFeatureEngineer(sample_log_data)

    # Create text features first (required for error_type column)
    engineer.create_text_features()

    # Create categorical features
    categorical_features = engineer.create_categorical_features()

    # Check shape
    assert categorical_features.shape[0] == len(sample_log_data)
    assert categorical_features.shape[1] == 3  # service, severity, error_type

    # Check that encoders are created
    assert engineer.service_encoder is not None
    assert hasattr(engineer.service_encoder, "classes_")

    # Check feature names
    assert len(engineer.feature_names) >= 3
    assert "service_encoded" in engineer.feature_names
    assert "severity_encoded" in engineer.feature_names
    assert "error_type_encoded" in engineer.feature_names


def test_extract_timestamp_features(sample_log_data):
    """Test timestamp feature extraction."""
    engineer = LogFeatureEngineer(sample_log_data)

    # Extract timestamp features
    timestamp_features = engineer.extract_timestamp_features()

    # Check shape
    assert timestamp_features.shape[0] == len(sample_log_data)
    assert (
        timestamp_features.shape[1] == 4
    )  # hour_of_day, day_of_week, is_weekend, is_business_hours

    # Check that timestamp column is created
    assert "timestamp_dt" in engineer.df.columns

    # Check feature names
    assert "hour_of_day" in engineer.feature_names
    assert "day_of_week" in engineer.feature_names
    assert "is_weekend" in engineer.feature_names
    assert "is_business_hours" in engineer.feature_names


def test_create_all_features(sample_log_data):
    """Test creation of all features combined."""
    engineer = LogFeatureEngineer(sample_log_data)

    # Create all features
    all_features, feature_names = engineer.create_all_features(tfidf_max_features=50)

    # Check shape
    assert all_features.shape[0] == len(sample_log_data)
    assert all_features.shape[1] <= 57  # 50 TF-IDF + 3 categorical + 4 timestamp

    # Check feature names
    assert len(feature_names) == all_features.shape[1]
    assert all(isinstance(name, str) for name in feature_names)


def test_prepare_labels(sample_log_data):
    """Test label preparation."""
    engineer = LogFeatureEngineer(sample_log_data)

    # Prepare labels
    labels = engineer.prepare_labels()

    # Check shape
    assert labels.shape[0] == len(sample_log_data)

    # Check that label encoder is created
    assert engineer.label_encoder is not None
    assert hasattr(engineer.label_encoder, "classes_")

    # Check that all classes are encoded
    assert len(engineer.label_encoder.classes_) == 5  # 5 unique labels in sample data


def test_split_data():
    """Test data splitting with stratification."""
    # Create a larger dataset for splitting test
    data = {
        "log_id": [f"log_{i:03d}" for i in range(40)],
        "timestamp": [f"2024-01-{(i % 30) + 1:02d} 10:30:00" for i in range(40)],
        "service": ["kyc-service", "ml-inference", "oauth-handler", "data-export"] * 10,
        "severity": ["ERROR", "WARNING", "INFO", "CRITICAL"] * 10,
        "log_message": [
            f"Error {i}: Something went wrong with service {i % 5}" for i in range(40)
        ],
        "root_cause_label": [f"RC-{(i % 8) + 1:02d}" for i in range(40)],
    }
    df = pd.DataFrame(data)

    engineer = LogFeatureEngineer(df)

    # Create features and labels with relaxed TF-IDF parameters for small dataset
    features, _ = engineer.create_all_features(tfidf_max_features=50, tfidf_max_df=1.0)
    labels = engineer.prepare_labels()

    # Split data with larger test size to ensure all classes have at least 2 samples
    X_train, X_test, y_train, y_test = engineer.split_data(
        features, labels, test_size=0.3, random_state=42
    )

    # Check shapes
    total_samples = len(df)
    train_samples = int(total_samples * 0.7)  # 1 - test_size(0.3)
    test_samples = total_samples - train_samples

    assert X_train.shape[0] == train_samples
    assert X_test.shape[0] == test_samples
    assert y_train.shape[0] == train_samples
    assert y_test.shape[0] == test_samples

    # Check that features have same number of columns
    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[1] == features.shape[1]

    # Check that all classes are represented in both splits
    train_classes = np.unique(y_train)
    test_classes = np.unique(y_test)
    all_classes = np.unique(labels)

    assert len(train_classes) == len(all_classes)
    assert len(test_classes) == len(all_classes)


def test_get_feature_analysis(sample_log_data):
    """Test feature analysis generation."""
    engineer = LogFeatureEngineer(sample_log_data)

    # Create some features first
    engineer.create_text_features()

    # Get analysis
    analysis = engineer.get_feature_analysis()

    # Check analysis structure
    assert "text_features" in analysis
    assert "categorical_features" in analysis
    assert "timestamp_features" in analysis
    assert "feature_summary" in analysis

    # Check specific values
    assert analysis["text_features"]["unique_error_types"] > 0
    assert analysis["categorical_features"]["unique_services"] == 5
    assert analysis["categorical_features"]["unique_severities"] == 4
    assert analysis["feature_summary"]["total_samples"] == 5


def test_feature_engineer_integration():
    """Integration test for the complete feature engineering pipeline."""
    # Create a more realistic dataset
    data = {
        "log_id": [f"log_{i:03d}" for i in range(20)],
        "timestamp": [f"2024-01-{i + 1:02d} 10:30:00" for i in range(20)],
        "service": ["kyc-service", "ml-inference"] * 10,
        "severity": ["ERROR", "WARNING", "INFO", "CRITICAL"] * 5,
        "log_message": [
            f"Error {i}: Something went wrong with service {i % 5}" for i in range(20)
        ],
        "root_cause_label": [f"RC-{(i % 8) + 1:02d}" for i in range(20)],
    }
    df = pd.DataFrame(data)

    # Create feature engineer
    engineer = LogFeatureEngineer(df)

    # Test complete pipeline with relaxed TF-IDF parameters for small dataset
    features, feature_names = engineer.create_all_features(
        tfidf_max_features=50, tfidf_max_df=1.0
    )
    labels = engineer.prepare_labels()

    # Verify outputs
    assert features.shape[0] == 20
    assert features.shape[1] <= 57  # 50 TF-IDF + 3 categorical + 4 timestamp
    assert len(feature_names) == features.shape[1]
    assert labels.shape[0] == 20

    # Test data splitting with larger test size to ensure at least 2 samples per class
    X_train, X_test, y_train, y_test = engineer.split_data(
        features, labels, test_size=0.4, random_state=42
    )

    assert X_train.shape[0] == 12  # 60% of 20
    assert X_test.shape[0] == 8  # 40% of 20
    assert y_train.shape[0] == 12
    assert y_test.shape[0] == 8

    # Verify class distribution in splits
    train_classes = np.unique(y_train)
    test_classes = np.unique(y_test)

    # Should have multiple classes in both splits due to stratification
    assert len(train_classes) > 1
    assert len(test_classes) > 1
