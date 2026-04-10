#!/usr/bin/env python3
"""
Example script demonstrating root cause summary generation.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.inference.summary import SummaryGenerator


def main():
    """Demonstrate summary generation for different root causes."""
    print("=" * 80)
    print("Root Cause Summary Generation Demo")
    print("=" * 80)

    # Create summary generator
    generator = SummaryGenerator()

    # Example log entries from the dataset
    examples = [
        {
            "root_cause": "RC-01",
            "confidence": 0.92,
            "log_message": "WARN [api-gateway] 401 returned to client client_3536: bearer token missing from Authorization header.",
            "service": "api-gateway",
            "severity": "High",
            "timestamp": "2024-09-06T10:22:00Z",
        },
        {
            "root_cause": "RC-02",
            "confidence": 0.87,
            "log_message": "ERROR [db-pool] Max wait time exceeded (11266ms) waiting for idle connection. Active: 9/15.",
            "service": "db-pool",
            "severity": "Critical",
            "timestamp": "2024-04-10T08:46:00Z",
        },
        {
            "root_cause": "RC-03",
            "confidence": 0.78,
            "log_message": "ERROR [payment-gateway] Upstream provider Twilio returned 502. Retried 1 times. Failing over.",
            "service": "payment-gateway",
            "severity": "High",
            "timestamp": "2024-05-28T21:04:00Z",
        },
        {
            "root_cause": "RC-04",
            "confidence": 0.65,
            "log_message": "ERROR [webhook-dispatcher] Rate limit hit on endpoint /v2/events. Retry-After: 120s.",
            "service": "webhook-dispatcher",
            "severity": "Medium",
            "timestamp": "2024-06-11T15:28:00Z",
        },
        {
            "root_cause": "RC-05",
            "confidence": 0.81,
            "log_message": "ERROR [transaction-validator] Constraint violation: currency_code 'null' not in allowed list.",
            "service": "transaction-validator",
            "severity": "Medium",
            "timestamp": "2024-05-25T19:04:00Z",
        },
        {
            "root_cause": "RC-06",
            "confidence": 0.94,
            "log_message": "CRITICAL [audit-service] Privilege escalation attempt blocked: user usr_68176 — insufficient role.",
            "service": "audit-service",
            "severity": "Critical",
            "timestamp": "2024-11-11T21:19:00Z",
        },
        {
            "root_cause": "RC-07",
            "confidence": 0.89,
            "log_message": "ERROR [log-aggregator] Disk write failed: volume /data at 87% capacity. Dropping logs.",
            "service": "log-aggregator",
            "severity": "Critical",
            "timestamp": "2024-12-09T20:15:00Z",
        },
        {
            "root_cause": "RC-08",
            "confidence": 0.76,
            "log_message": "CRITICAL [vpc-gateway] Packet loss 97% detected on internal network segment. Alert triggered.",
            "service": "vpc-gateway",
            "severity": "Critical",
            "timestamp": "2024-05-21T11:08:00Z",
        },
    ]

    # Generate and display summaries
    for i, example in enumerate(examples, 1):
        print(f"\n{'=' * 60}")
        print(f"Example {i}: {example['root_cause']}")
        print(f"{'=' * 60}")

        # Generate summary
        summary = generator.generate_summary(
            root_cause=example["root_cause"],
            confidence=example["confidence"],
            log_message=example["log_message"],
            service=example["service"],
            severity=example["severity"],
            timestamp=example["timestamp"],
        )

        # Display summary
        print(f"Root Cause: {summary.root_cause}")
        print(f"Confidence: {summary.confidence:.2f}")
        print(f"Severity: {summary.severity}")
        print(f"Impact: {summary.impact}")
        print(f"Time to Resolution: {summary.time_to_resolution}")
        print(f"\nSummary: {summary.summary}")

        print(f"\nKey Evidence:")
        for evidence in summary.key_evidence[:3]:  # Show top 3 evidence items
            print(f"  • {evidence}")

        print(f"\nRecommended Actions:")
        for action in summary.recommended_actions[:3]:  # Show top 3 actions
            print(f"  • {action}")

        # Show JSON output
        print(f"\nJSON Output (first 200 chars):")
        json_str = summary.to_json()
        print(json_str[:200] + "..." if len(json_str) > 200 else json_str)

    print(f"\n{'=' * 80}")
    print("Summary Generation Complete!")
    print(f"{'=' * 80}")

    # Test evidence extraction
    print(f"\n{'=' * 60}")
    print("Evidence Extraction Examples")
    print(f"{'=' * 60}")

    test_messages = [
        ("RC-01", "401 Unauthorized — invalid API key provided by client client_8478"),
        (
            "RC-02",
            "Failed to acquire DB connection from pool: all 15 connections exhausted",
        ),
        ("RC-03", "Upstream provider returned 502 Bad Gateway"),
        ("RC-04", "Rate limit hit with Retry-After: 120 seconds"),
        ("RC-05", "Constraint violation: null value in required field"),
    ]

    for rc, message in test_messages:
        evidence = generator.extract_evidence(message, rc)
        print(f"\n{rc}: {message}")
        print(f"Extracted Evidence: {evidence[:3]}")  # Show first 3 evidence items


if __name__ == "__main__":
    main()
