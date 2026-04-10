#!/usr/bin/env python3
"""
Example inference script for root cause prediction.
Demonstrates single and batch prediction with confidence scores.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predictor import LogPredictor


def run_single_prediction(predictor: LogPredictor) -> None:
    """Demonstrate single log entry prediction."""
    print("\n" + "=" * 60)
    print("Single Log Entry Prediction")
    print("=" * 60)

    # Example log entries for different root causes
    examples = [
        {
            "log_message": "Database connection timeout after 30 seconds. Retrying...",
            "service": "db-pool",
            "severity": "ERROR",
            "description": "Database connectivity issue (likely RC-01)",
        },
        {
            "log_message": "API rate limit exceeded for user 12345. Please wait 60 seconds.",
            "service": "api-gateway",
            "severity": "WARNING",
            "description": "Rate limiting issue (likely RC-02)",
        },
        {
            "log_message": "Memory usage at 95%. Initiating garbage collection.",
            "service": "auth-service",
            "severity": "CRITICAL",
            "description": "Resource exhaustion (likely RC-03)",
        },
        {
            "log_message": "Invalid JWT token format in authorization header",
            "service": "oauth-handler",
            "severity": "ERROR",
            "description": "Authentication/authorization issue (likely RC-04)",
        },
        {
            "log_message": "Payment transaction failed: insufficient funds",
            "service": "payments-core",
            "severity": "ERROR",
            "description": "Business logic error (likely RC-05)",
        },
        {
            "log_message": "Network latency detected: 500ms average response time",
            "service": "service-mesh",
            "severity": "WARNING",
            "description": "Network/performance issue (likely RC-06)",
        },
        {
            "log_message": "Configuration file not found: /etc/app/config.yaml",
            "service": "iam-middleware",
            "severity": "ERROR",
            "description": "Configuration issue (likely RC-07)",
        },
        {
            "log_message": "Third-party API response timeout after 10 seconds",
            "service": "http-client",
            "severity": "ERROR",
            "description": "External dependency failure (likely RC-08)",
        },
    ]

    for example in examples:
        print(f"\nExample: {example['description']}")
        print(f"Log: {example['log_message']}")
        print(f"Service: {example['service']}, Severity: {example['severity']}")

        try:
            result = predictor.predict_single(
                log_message=example["log_message"],
                service=example["service"],
                severity=example["severity"],
            )

            print(f"Predicted Root Cause: {result.root_cause}")
            print(f"Confidence: {result.confidence:.3f}")
            print("Top 3 Predictions:")
            for rc, conf in result.top_n_predictions:
                print(f"  - {rc}: {conf:.3f}")

        except Exception as e:
            print(f"Prediction failed: {e}")


def run_batch_prediction(predictor: LogPredictor) -> None:
    """Demonstrate batch prediction with multiple log entries."""
    print("\n" + "=" * 60)
    print("Batch Log Entry Prediction")
    print("=" * 60)

    # Create batch of log entries
    batch_data = pd.DataFrame(
        [
            {
                "log_message": "Database connection pool exhausted. Max connections: 100",
                "service": "db-pool",
                "severity": "ERROR",
                "timestamp": "2026-04-09T10:15:30",
            },
            {
                "log_message": "Invalid request payload: missing required field 'amount'",
                "service": "api-gateway",
                "severity": "WARNING",
                "timestamp": "2026-04-09T10:16:45",
            },
            {
                "log_message": "CPU usage at 98% for 5 minutes",
                "service": "log-aggregator",
                "severity": "CRITICAL",
                "timestamp": "2026-04-09T10:17:20",
            },
            {
                "log_message": "SSL certificate expired for domain api.example.com",
                "service": "audit-service",
                "severity": "CRITICAL",
                "timestamp": "2026-04-09T10:18:10",
            },
        ]
    )

    print(f"Processing batch of {len(batch_data)} log entries...")

    try:
        results = predictor.predict_batch(batch_data, top_n=2)

        for i, result in enumerate(results):
            print(f"\nEntry {i + 1}:")
            print(f"  Log: {batch_data.iloc[i]['log_message'][:50]}...")
            print(
                f"  Predicted: {result.root_cause} (confidence: {result.confidence:.3f})"
            )
            print(
                f"  Top predictions: {', '.join([f'{rc}:{conf:.3f}' for rc, conf in result.top_n_predictions])}"
            )

    except Exception as e:
        print(f"Batch prediction failed: {e}")


def test_error_handling(predictor: LogPredictor) -> None:
    """Test error handling for malformed inputs."""
    print("\n" + "=" * 60)
    print("Error Handling Tests")
    print("=" * 60)

    # Test 1: Empty log message
    print("\nTest 1: Empty log message")
    try:
        result = predictor.predict_single(
            log_message="", service="test-service", severity="ERROR"
        )
        print(f"Unexpected success: {result}")
    except ValueError as e:
        print(f"✓ Correctly rejected: {e}")

    # Test 2: Missing required column in batch
    print("\nTest 2: Missing required column in batch")
    try:
        invalid_batch = pd.DataFrame(
            [
                {"service": "test", "severity": "ERROR"}  # Missing log_message
            ]
        )
        results = predictor.predict_batch(invalid_batch)
        print(f"Unexpected success: {results}")
    except ValueError as e:
        print(f"✓ Correctly rejected: {e}")

    # Test 3: Non-string log message
    print("\nTest 3: Non-string log message")
    try:
        invalid_batch = pd.DataFrame(
            [{"log_message": 123, "service": "test", "severity": "ERROR"}]
        )
        results = predictor.predict_batch(invalid_batch)
        print(f"Unexpected success: {results}")
    except ValueError as e:
        print(f"✓ Correctly rejected: {e}")


def main():
    """Main function to run inference examples."""
    # Path to inference pipeline
    inference_pipeline_path = "models/inference_pipeline.joblib"

    if not Path(inference_pipeline_path).exists():
        print(f"Error: Inference pipeline not found at {inference_pipeline_path}")
        print(
            "Please create inference pipeline first using scripts/create_inference_pipeline.py"
        )
        return

    print("Loading inference pipeline...")
    try:
        # Load the complete inference pipeline
        predictor = LogPredictor.load(inference_pipeline_path)

        # Run examples
        run_single_prediction(predictor)
        run_batch_prediction(predictor)
        test_error_handling(predictor)

        print(
            f"\n✓ Inference pipeline loaded successfully from {inference_pipeline_path}"
        )

    except Exception as e:
        print(f"Failed to load inference pipeline: {e}")
        return


if __name__ == "__main__":
    main()
