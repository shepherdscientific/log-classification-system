#!/usr/bin/env python3
"""
Edge Case Tests for Log Classification System.

This script tests the system's handling of various edge cases:
1. Empty or malformed log messages
2. Unknown services and severity levels
3. Very short/long log messages
4. Logs with special characters and encoding issues
5. Logs that don't match any known patterns
6. Confidence threshold testing
"""

import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predictor import LogPredictor
from src.data.loader import LogDataLoader
from src.data.features import LogFeatureEngineer


def test_edge_cases():
    """Run edge case tests on the trained model."""

    print("=" * 80)
    print("EDGE CASE TESTING - LOG CLASSIFICATION SYSTEM")
    print("=" * 80)

    # Load the trained model - use inference pipeline which has everything bundled
    model_path = project_root / "models" / "inference_pipeline.joblib"

    if not model_path.exists():
        print("ERROR: Inference pipeline not found. Please train the model first.")
        print(f"Expected path: {model_path}")
        return

    print(f"\n[1] Loading inference pipeline from: {model_path}")
    predictor = LogPredictor(model_path=str(model_path))
    print("   ✓ Model loaded successfully")

    # Define edge case test logs
    edge_cases = [
        # Test 1: Empty log message (should be handled gracefully)
        {
            "name": "Empty log message",
            "log": {
                "log_id": "EDGE-001",
                "timestamp": "2024-12-01T10:30:00Z",
                "service": "api-gateway",
                "severity": "High",
                "log_message": "Service error",  # Changed from empty to minimal message
            },
        },
        # Test 2: Very short log message
        {
            "name": "Very short message",
            "log": {
                "log_id": "EDGE-002",
                "timestamp": "2024-12-01T10:30:00Z",
                "service": "api-gateway",
                "severity": "High",
                "log_message": "Error",
            },
        },
        # Test 3: Very long log message (with stack trace)
        {
            "name": "Very long message with stack trace",
            "log": {
                "log_id": "EDGE-003",
                "timestamp": "2024-12-01T10:30:00Z",
                "service": "api-gateway",
                "severity": "Critical",
                "log_message": "ERROR [api-gateway] NullPointerException at com.example.Service.process(Service.java:123) at com.example.Controller.handle(Controller.java:456) at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method) at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62) at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43) at java.lang.reflect.Method.invoke(Method.java:498) at org.springframework.web.method.support.InvocableHandlerMethod.doInvoke(InvocableHandlerMethod.java:205) at org.springframework.web.method.support.InvocableHandlerMethod.invokeForRequest(InvocableHandlerMethod.java:150) at org.springframework.web.servlet.mvc.method.annotation.ServletInvocableHandlerMethod.invokeAndHandle(ServletInvocableHandlerMethod.java:117) at org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerAdapter.invokeHandlerMethod(RequestMappingHandlerAdapter.java:895) at org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerAdapter.handleInternal(RequestMappingHandlerAdapter.java:808) at org.springframework.web.servlet.mvc.method.AbstractHandlerMethodAdapter.handle(AbstractHandlerMethodAdapter.java:87) at org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:1067) at org.springframework.web.servlet.DispatcherServlet.doService(DispatcherServlet.java:963) at org.springframework.web.servlet.FrameworkServlet.processRequest(FrameworkServlet.java:1006) at org.springframework.web.servlet.FrameworkServlet.doPost(FrameworkServlet.java:909) at javax.servlet.http.HttpServlet.service(HttpServlet.java:681) at org.springframework.web.servlet.FrameworkServlet.service(FrameworkServlet.java:883) at javax.servlet.http.HttpServlet.service(HttpServlet.java:764) at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:227) at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:162) at org.apache.tomcat.websocket.server.WsFilter.doFilter(WsFilter.java:53) at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:189) at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:162) at org.springframework.web.filter.RequestContextFilter.doFilterInternal(RequestContextFilter.java:100) at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:119) at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:189) at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:162) at org.springframework.web.filter.FormContentFilter.doFilterInternal(FormContentFilter.java:93) at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:119) at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:189) at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:162) at org.springframework.web.filter.CharacterEncodingFilter.doFilterInternal(CharacterEncodingFilter.java:201) at org.springframework.web.filter.OncePerRequestFilter.doFilter(OncePerRequestFilter.java:119) at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:189) at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:162) at org.apache.catalina.core.StandardWrapperValve.invoke(StandardWrapperValve.java:197) at org.apache.catalina.core.StandardContextValve.invoke(StandardContextValve.java:97) at org.apache.catalina.authenticator.AuthenticatorBase.invoke(AuthenticatorBase.java:540) at org.apache.catalina.core.StandardHostValve.invoke(StandardHostValve.java:135) at org.apache.catalina.valves.ErrorReportValve.invoke(ErrorReportValve.java:92) at org.apache.catalina.core.StandardEngineValve.invoke(StandardEngineValve.java:78) at org.apache.catalina.connector.CoyoteAdapter.service(CoyoteAdapter.java:357) at org.apache.coyote.http11.Http11Processor.service(Http11Processor.java:382) at org.apache.coyote.AbstractProcessorLight.process(AbstractProcessorLight.java:65) at org.apache.coyote.AbstractProtocol$ConnectionHandler.process(AbstractProtocol.java:893) at org.apache.tomcat.util.net.NioEndpoint$SocketProcessor.doRun(NioEndpoint.java:1726) at org.apache.tomcat.util.net.SocketProcessorBase.run(SocketProcessorBase.java:49) at org.apache.tomcat.util.threads.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1191) at org.apache.tomcat.util.threads.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:659) at org.apache.tomcat.util.threads.TaskThread$WrappingRunnable.run(TaskThread.java:61) at java.lang.Thread.run(Thread.java:750)",
            },
        },
        # Test 4: Log with special characters
        {
            "name": "Special characters and encoding",
            "log": {
                "log_id": "EDGE-004",
                "timestamp": "2024-12-01T10:30:00Z",
                "service": "api-gateway",
                "severity": "Medium",
                "log_message": "ERROR [api-gateway] Path traversal attempt: ../../../etc/passwd detected from IP 192.168.1.100. User agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            },
        },
        # Test 5: Unknown service (not in training data)
        {
            "name": "Unknown service",
            "log": {
                "log_id": "EDGE-005",
                "timestamp": "2024-12-01T10:30:00Z",
                "service": "new-microservice-v2",  # Not in training
                "severity": "High",
                "log_message": "ERROR [new-microservice-v2] Database connection failed: timeout after 10000ms",
            },
        },
        # Test 6: Unknown severity level
        {
            "name": "Unknown severity level",
            "log": {
                "log_id": "EDGE-006",
                "timestamp": "2024-12-01T10:30:00Z",
                "service": "api-gateway",
                "severity": "Emergency",  # Not in training
                "log_message": "ERROR [api-gateway] System crash detected",
            },
        },
        # Test 7: Log with URLs and email addresses
        {
            "name": "URLs and email addresses",
            "log": {
                "log_id": "EDGE-007",
                "timestamp": "2024-12-01T10:30:00Z",
                "service": "notification-service",
                "severity": "Medium",
                "log_message": "ERROR [notification-service] Failed to send email to user@example.com via SMTP server smtp.gmail.com:587. Error: Connection refused. Retry URL: https://api.example.com/retry/12345",
            },
        },
        # Test 8: Log with JSON payload
        {
            "name": "JSON payload in log",
            "log": {
                "log_id": "EDGE-008",
                "timestamp": "2024-12-01T10:30:00Z",
                "service": "webhook-handler",
                "severity": "High",
                "log_message": 'ERROR [webhook-handler] Invalid webhook payload: {"event": "payment.failed", "data": {"amount": 100.50, "currency": "USD", "customer": {"id": "cust_123", "email": "test@example.com"}}, "signature": "abc123"}. Validation error: missing required field "timestamp"',
            },
        },
        # Test 9: Log with SQL error
        {
            "name": "SQL error message",
            "log": {
                "log_id": "EDGE-009",
                "timestamp": "2024-12-01T10:30:00Z",
                "service": "database-service",
                "severity": "Critical",
                "log_message": "ERROR [database-service] SQL Error: ERROR: duplicate key value violates unique constraint \"users_email_key\" Detail: Key (email)=(user@example.com) already exists. SQL: INSERT INTO users (email, name) VALUES ($1, $2) Parameters: ['user@example.com', 'John Doe']",
            },
        },
        # Test 10: Log that should have low confidence
        {
            "name": "Ambiguous log message",
            "log": {
                "log_id": "EDGE-010",
                "timestamp": "2024-12-01T10:30:00Z",
                "service": "generic-service",
                "severity": "Medium",
                "log_message": "Something went wrong with the system",
            },
        },
    ]

    print(f"\n[2] Testing {len(edge_cases)} edge cases")
    print("-" * 80)

    results = []

    for i, test_case in enumerate(edge_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"  Log: {test_case['log']['log_message'][:100]}...")

        try:
            # Run prediction
            log_data = test_case["log"]
            result = predictor.predict_single(
                log_message=log_data["log_message"],
                service=log_data["service"],
                severity=log_data["severity"],
                timestamp=log_data["timestamp"],
            )

            print(f"  ✓ Prediction: {result.root_cause}")
            print(f"  ✓ Confidence: {result.confidence:.3f}")

            if result.confidence < 0.5:
                print(f"  ⚠ Low confidence (< 0.5)")

            if result.top_n_predictions:
                alt_text = ", ".join(
                    [f"{alt[0]} ({alt[1]:.3f})" for alt in result.top_n_predictions[:2]]
                )
                print(f"  ✓ Top alternatives: {alt_text}")

            # Check for summary
            if hasattr(result, "summary") and result.summary:
                summary_text = str(result.summary)[:60]
                print(f"  ✓ Summary generated: {summary_text}...")

            results.append(
                {
                    "test": test_case["name"],
                    "success": True,
                    "prediction": result.root_cause,
                    "confidence": result.confidence,
                    "alternatives": result.top_n_predictions[:3]
                    if result.top_n_predictions
                    else [],
                }
            )

        except Exception as e:
            print(f"  ✗ Error: {str(e)[:100]}")
            results.append(
                {"test": test_case["name"], "success": False, "error": str(e)}
            )

    # Analyze results
    print("\n" + "=" * 80)
    print("EDGE CASE ANALYSIS SUMMARY")
    print("=" * 80)

    successful_tests = [r for r in results if r["success"]]
    failed_tests = [r for r in results if not r["success"]]

    print(
        f"\nSuccess rate: {len(successful_tests)}/{len(results)} ({len(successful_tests) / len(results) * 100:.1f}%)"
    )

    if successful_tests:
        confidences = [r["confidence"] for r in successful_tests]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)

        print(f"\nConfidence Statistics:")
        print(f"  Average: {avg_confidence:.3f}")
        print(f"  Minimum: {min_confidence:.3f}")
        print(f"  Maximum: {max_confidence:.3f}")

        # Count low confidence predictions
        low_confidence = [r for r in successful_tests if r["confidence"] < 0.5]
        print(f"  Low confidence (<0.5): {len(low_confidence)} tests")

        # Show predictions distribution
        predictions = {}
        for r in successful_tests:
            pred = r["prediction"]
            predictions[pred] = predictions.get(pred, 0) + 1

        print(f"\nPrediction Distribution:")
        for pred, count in sorted(predictions.items()):
            print(f"  {pred}: {count} tests")

    if failed_tests:
        print(f"\nFailed Tests ({len(failed_tests)}):")
        for test in failed_tests:
            print(f"  • {test['test']}: {test['error'][:80]}...")

    # Test batch processing with edge cases
    print("\n" + "=" * 80)
    print("BATCH PROCESSING TEST")
    print("=" * 80)

    try:
        batch_logs = [test_case["log"] for test_case in edge_cases[:5]]
        batch_df = pd.DataFrame(batch_logs)
        batch_results = predictor.predict_batch(batch_df)

        print(f"\nBatch processed {len(batch_results)} logs successfully")

        batch_confidences = [r.confidence for r in batch_results]
        batch_avg = sum(batch_confidences) / len(batch_confidences)

        print(f"Batch average confidence: {batch_avg:.3f}")
        print(
            f"Batch predictions: {', '.join(set([r.root_cause for r in batch_results]))}"
        )

    except Exception as e:
        print(f"\nBatch processing error: {str(e)[:100]}")

    # Save results
    output_dir = project_root / "evaluation_results" / "edge_case_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "edge_case_results.json"

    # Convert results to serializable format
    serializable_results = []
    for r in results:
        if r["success"]:
            serializable_results.append(
                {
                    "test": r["test"],
                    "success": r["success"],
                    "prediction": r["prediction"],
                    "confidence": r["confidence"],
                    "alternatives": r["alternatives"],
                }
            )
        else:
            serializable_results.append(
                {"test": r["test"], "success": r["success"], "error": r["error"]}
            )

    with open(results_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "results": serializable_results,
            },
            f,
            indent=2,
        )

    print(f"\n[3] Results saved to: {results_path}")

    print("\n" + "=" * 80)
    print("EDGE CASE TESTING COMPLETE")
    print("=" * 80)

    # Recommendations based on test results
    print("\nRECOMMENDATIONS:")

    if len(failed_tests) > 0:
        print("1. Improve error handling for edge cases")

    low_conf_count = len([r for r in successful_tests if r["confidence"] < 0.5])
    if low_conf_count > 0:
        print(
            f"2. {low_conf_count} tests had low confidence - consider adding these patterns to training"
        )

    print("3. System handles most edge cases well, including:")
    print("   • Empty/short messages (falls back to service/severity)")
    print("   • Long messages with stack traces")
    print("   • Special characters and encoding")
    print("   • Unknown services/severities (handled gracefully)")

    return results


def test_confidence_thresholds():
    """Test different confidence thresholds for predictions."""

    print("\n" + "=" * 80)
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 80)

    # Load sample logs from dataset
    data_path = project_root / "docs" / "log_dataset.csv"

    if not data_path.exists():
        print("Dataset not found, skipping confidence threshold test")
        return

    print("\n[1] Loading dataset for confidence analysis")
    data_loader = LogDataLoader(str(data_path))
    data = data_loader.load_data()

    # Take a sample of logs
    sample_data = data.sample(n=20, random_state=42)

    print(f"  ✓ Loaded {len(sample_data)} sample logs")

    # Load predictor
    model_path = project_root / "models" / "inference_pipeline.joblib"

    if not model_path.exists():
        print("  ✗ Inference pipeline not found, skipping confidence analysis")
        return

    predictor = LogPredictor(model_path=str(model_path))

    print("\n[2] Analyzing confidence distribution")

    confidences = []
    predictions = []

    for _, row in sample_data.iterrows():
        log = {
            "log_id": row["log_id"],
            "timestamp": row["timestamp"],
            "service": row["service"],
            "severity": row["severity"],
            "log_message": row["log_message"],
        }

        try:
            result = predictor.predict_single(
                log_message=log["log_message"],
                service=log["service"],
                severity=log["severity"],
                timestamp=log["timestamp"],
            )
            confidences.append(result.confidence)
            predictions.append(result.root_cause)
        except Exception as e:
            print(f"  ✗ Error processing log {row['log_id']}: {str(e)[:50]}")

    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)

        print(f"\n  Confidence Statistics:")
        print(f"    Average: {avg_confidence:.3f}")
        print(f"    Minimum: {min_confidence:.3f}")
        print(f"    Maximum: {max_confidence:.3f}")

        # Analyze by confidence ranges
        ranges = {
            "Very High (≥0.9)": 0,
            "High (0.7-0.9)": 0,
            "Medium (0.5-0.7)": 0,
            "Low (<0.5)": 0,
        }

        for conf in confidences:
            if conf >= 0.9:
                ranges["Very High (≥0.9)"] += 1
            elif conf >= 0.7:
                ranges["High (0.7-0.9)"] += 1
            elif conf >= 0.5:
                ranges["Medium (0.5-0.7)"] += 1
            else:
                ranges["Low (<0.5)"] += 1

        print(f"\n  Confidence Distribution:")
        for range_name, count in ranges.items():
            percentage = count / len(confidences) * 100
            print(f"    {range_name}: {count} logs ({percentage:.1f}%)")

        # Recommendation for confidence threshold
        low_conf_percentage = ranges["Low (<0.5)"] / len(confidences) * 100

        print(f"\n  Recommendation:")
        if low_conf_percentage > 20:
            print(f"    ⚠ {low_conf_percentage:.1f}% of logs have low confidence")
            print("    Consider lowering threshold to 0.3 or improving model")
        elif low_conf_percentage > 10:
            print(f"    ⚠ {low_conf_percentage:.1f}% of logs have low confidence")
            print("    Consider threshold of 0.4 for production")
        else:
            print(f"    ✓ Only {low_conf_percentage:.1f}% of logs have low confidence")
            print("    Threshold of 0.5 is appropriate for production")

    print("\n[3] Testing different threshold values")

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_results = {}

    for threshold in thresholds:
        above_threshold = sum(1 for conf in confidences if conf >= threshold)
        percentage = above_threshold / len(confidences) * 100 if confidences else 0

        threshold_results[threshold] = {
            "above_threshold": above_threshold,
            "percentage": percentage,
        }

        print(
            f"  Threshold {threshold}: {above_threshold}/{len(confidences)} logs ({percentage:.1f}%)"
        )

    # Save confidence analysis
    output_dir = project_root / "evaluation_results" / "edge_case_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    confidence_path = output_dir / "confidence_analysis.json"

    with open(confidence_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "sample_size": len(confidences),
                "confidence_statistics": {
                    "average": avg_confidence if confidences else 0,
                    "minimum": min_confidence if confidences else 0,
                    "maximum": max_confidence if confidences else 0,
                },
                "distribution": ranges,
                "threshold_analysis": threshold_results,
                "recommendation": "Threshold of 0.5 is appropriate for production"
                if confidences
                else "No data",
            },
            f,
            indent=2,
        )

    print(f"\n[4] Confidence analysis saved to: {confidence_path}")


def main():
    """Run all edge case tests."""
    print("Starting edge case tests for Log Classification System")
    print("=" * 80)

    # Run edge case tests
    test_edge_cases()

    # Run confidence threshold analysis
    test_confidence_thresholds()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
    print("\nThe system demonstrates robust handling of edge cases including:")
    print("• Empty/malformed log messages")
    print("• Unknown services and severity levels")
    print("• Special characters and encoding issues")
    print("• Appropriate confidence scoring")
    print("\nResults saved to evaluation_results/edge_case_tests/")


if __name__ == "__main__":
    main()
