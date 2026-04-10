"""
Root cause summary generation module.
Generates structured, human-readable summaries for root cause predictions.
"""

import json
from typing import Dict, List, Any, Optional, Union, Sequence
from dataclasses import dataclass, asdict
import re
from datetime import datetime


@dataclass
class RootCauseSummary:
    """Structured summary for a root cause prediction."""

    root_cause: str
    confidence: float
    summary: str
    key_evidence: List[str]
    recommended_actions: List[str]
    severity: str
    impact: str
    time_to_resolution: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class SummaryGenerator:
    """
    Generates structured summaries for root cause predictions.

    Uses template-based summaries for each RC category with
    evidence extraction from log messages.
    """

    def __init__(self):
        """Initialize summary generator with templates."""
        self.templates = self._load_templates()
        self.patterns = self._load_patterns()

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load summary templates for each root cause category."""
        return {
            "RC-01": {
                "title": "Authentication/Authorization Failure",
                "summary_template": "Authentication or authorization failure detected. {evidence}",
                "key_evidence_patterns": [
                    r"401.*Unauthorized",
                    r"bearer token missing",
                    r"invalid API key",
                    r"authentication.*failed",
                    r"HMAC.*signature.*mismatch",
                ],
                "recommended_actions": [
                    "Check API key/token validity and expiration",
                    "Verify authentication headers are properly formatted",
                    "Review IP allowlist/blocklist configurations",
                    "Check identity provider service health",
                ],
                "severity": "High",
                "impact": "Service access blocked for affected clients",
                "time_to_resolution": "15-30 minutes",
            },
            "RC-02": {
                "title": "Database Connection Pool Exhaustion",
                "summary_template": "Database connection pool exhausted or connection timeout. {evidence}",
                "key_evidence_patterns": [
                    r"connection.*exhausted",
                    r"wait.*time.*exceeded",
                    r"failed.*acquire.*connection",
                    r"database.*unreachable",
                    r"slow query.*detected",
                ],
                "recommended_actions": [
                    "Increase database connection pool size",
                    "Optimize slow-running queries",
                    "Check database server resource utilization",
                    "Implement query timeout limits",
                    "Add database connection retry logic",
                ],
                "severity": "Critical",
                "impact": "Service degradation or complete outage",
                "time_to_resolution": "30-60 minutes",
            },
            "RC-03": {
                "title": "Upstream Service Failure",
                "summary_template": "Upstream service dependency failure. {evidence}",
                "key_evidence_patterns": [
                    r"upstream.*provider.*returned",
                    r"502.*Bad Gateway",
                    r"503.*Service Unavailable",
                    r"failing over",
                    r"dependency.*failure",
                ],
                "recommended_actions": [
                    "Check upstream service health and status",
                    "Implement circuit breaker pattern",
                    "Add fallback mechanisms",
                    "Monitor dependency service metrics",
                    "Contact upstream service provider",
                ],
                "severity": "High",
                "impact": "Partial service functionality loss",
                "time_to_resolution": "30-60 minutes",
            },
            "RC-04": {
                "title": "Rate Limiting/Throttling",
                "summary_template": "Rate limit exceeded or throttling applied. {evidence}",
                "key_evidence_patterns": [
                    r"rate limit.*hit",
                    r"Retry-After",
                    r"throttling.*applied",
                    r"429.*Too Many Requests",
                    r"quota.*exceeded",
                ],
                "recommended_actions": [
                    "Review rate limit configuration",
                    "Implement exponential backoff for retries",
                    "Monitor API usage patterns",
                    "Consider increasing rate limits if appropriate",
                    "Add client-specific rate limiting",
                ],
                "severity": "Medium",
                "impact": "Temporary service degradation for specific clients",
                "time_to_resolution": "15-30 minutes",
            },
            "RC-05": {
                "title": "Data Validation/Constraint Violation",
                "summary_template": "Data validation error or constraint violation. {evidence}",
                "key_evidence_patterns": [
                    r"constraint.*violation",
                    r"validation.*error",
                    r"null.*value.*required",
                    r"format.*error",
                    r"enum.*mismatch",
                ],
                "recommended_actions": [
                    "Review input data validation rules",
                    "Check database constraints and schemas",
                    "Implement data sanitization at entry points",
                    "Add comprehensive error logging for validation failures",
                    "Update API documentation for required fields",
                ],
                "severity": "Medium",
                "impact": "Data processing failures for specific records",
                "time_to_resolution": "30-60 minutes",
            },
            "RC-06": {
                "title": "Security/Authorization Policy Violation",
                "summary_template": "Security policy violation or unauthorized access attempt. {evidence}",
                "key_evidence_patterns": [
                    r"privilege.*escalation",
                    r"insufficient.*role",
                    r"scope.*mismatch",
                    r"unauthorized.*access",
                    r"security.*violation",
                ],
                "recommended_actions": [
                    "Review user roles and permissions",
                    "Check access control policies",
                    "Audit user activity logs",
                    "Implement principle of least privilege",
                    "Review security monitoring alerts",
                ],
                "severity": "Critical",
                "impact": "Potential security breach or data exposure",
                "time_to_resolution": "Immediate investigation required",
            },
            "RC-07": {
                "title": "Resource Exhaustion (Memory/Disk)",
                "summary_template": "System resource exhaustion detected. {evidence}",
                "key_evidence_patterns": [
                    r"memory.*limit.*reached",
                    r"disk.*capacity",
                    r"file descriptor.*limit",
                    r"evicting.*keys",
                    r"dropping.*logs",
                ],
                "recommended_actions": [
                    "Monitor system resource utilization",
                    "Implement automatic scaling",
                    "Add resource cleanup procedures",
                    "Review memory leak detection",
                    "Increase resource limits if appropriate",
                ],
                "severity": "Critical",
                "impact": "Service degradation or failure",
                "time_to_resolution": "30-60 minutes",
            },
            "RC-08": {
                "title": "Network/Infrastructure Issue",
                "summary_template": "Network or infrastructure problem detected. {evidence}",
                "key_evidence_patterns": [
                    r"packet.*loss",
                    r"network.*segment",
                    r"connectivity.*issue",
                    r"latency.*spike",
                    r"timeout.*network",
                ],
                "recommended_actions": [
                    "Check network connectivity and routing",
                    "Monitor network performance metrics",
                    "Review infrastructure health checks",
                    "Contact network operations team",
                    "Implement network redundancy",
                ],
                "severity": "Critical",
                "impact": "Widespread service disruption",
                "time_to_resolution": "60+ minutes",
            },
        }

    def _load_patterns(self) -> Dict[str, List[str]]:
        """Load regex patterns for evidence extraction."""
        return {
            "error_codes": [
                r"\b\d{3}\b",  # HTTP status codes
                r"ERROR\s+\d+",  # Custom error codes
            ],
            "services": [
                r"\[([^\]]+)\]",  # Service names in brackets
            ],
            "metrics": [
                r"\b\d+(?:\.\d+)?%",  # Percentages
                r"\b\d+(?:\.\d+)?(?:ms|s|m|h)\b",  # Time durations
                r"\b\d+(?:\.\d+)?(?:MB|GB|TB)\b",  # Memory sizes
            ],
            "entities": [
                r"client_\w+",  # Client IDs
                r"user_\w+",  # User IDs
                r"usr_\w+",  # Alternative user IDs
                r"\bIP:\s*\d+\.\d+\.\d+\.\d+\b",  # IP addresses
            ],
        }

    def extract_evidence(self, log_message: str, root_cause: str) -> List[str]:
        """
        Extract key evidence from log message for summary generation.

        Args:
            log_message: The log message text
            root_cause: Root cause category (RC-01 to RC-08)

        Returns:
            List of extracted evidence strings
        """
        evidence = []

        # Extract patterns specific to this root cause
        if root_cause in self.templates:
            template = self.templates[root_cause]
            for pattern in template["key_evidence_patterns"]:
                matches = re.findall(pattern, log_message, re.IGNORECASE)
                if matches:
                    evidence.extend(matches)

        # Extract general patterns
        for pattern_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, log_message)
                if matches:
                    for match in matches:
                        evidence.append(f"{pattern_type}: {match}")

        # If no specific evidence found, use key phrases from log
        if not evidence:
            # Take first 50 characters as evidence
            evidence.append(f"Log excerpt: {log_message[:50]}...")

        # Deduplicate and limit
        evidence = list(set(evidence))[:5]

        return evidence

    def generate_summary(
        self,
        root_cause: str,
        confidence: float,
        log_message: str,
        service: str,
        severity: str,
        timestamp: Optional[str] = None,
    ) -> RootCauseSummary:
        """
        Generate structured summary for a root cause prediction.

        Args:
            root_cause: Predicted root cause (RC-01 to RC-08)
            confidence: Prediction confidence score (0.0 to 1.0)
            log_message: Original log message
            service: Service name
            severity: Log severity level
            timestamp: Optional timestamp

        Returns:
            RootCauseSummary object with structured information
        """
        # Get template for this root cause
        if root_cause not in self.templates:
            # Default template for unknown root causes
            template: Dict[str, Any] = {
                "title": f"Root Cause {root_cause}",
                "summary_template": "Root cause identified: {evidence}",
                "key_evidence_patterns": [],
                "recommended_actions": [
                    "Investigate the specific error pattern",
                    "Check service logs for related errors",
                    "Review system metrics around the time of error",
                ],
                "severity": "Medium",
                "impact": "Service impact varies based on error type",
                "time_to_resolution": "Investigation required",
            }
        else:
            template = self.templates[root_cause]

        # Extract evidence from log message
        evidence = self.extract_evidence(log_message, root_cause)

        # Format evidence for summary
        if evidence:
            evidence_text = "Evidence includes: " + "; ".join(evidence[:3])
        else:
            evidence_text = "No specific evidence patterns detected."

        # Generate summary text
        summary_template: str = template["summary_template"]
        summary_text = summary_template.format(evidence=evidence_text)

        # Add context if available
        context_parts = []
        if service:
            context_parts.append(f"Service: {service}")
        if severity:
            context_parts.append(f"Severity: {severity}")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                context_parts.append(f"Time: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            except:
                context_parts.append(f"Timestamp: {timestamp}")

        if context_parts:
            summary_text += f" Context: {', '.join(context_parts)}."

        # Adjust severity based on confidence if low
        final_severity: str = template["severity"]
        if confidence < 0.5:
            final_severity = f"{final_severity} (Low confidence prediction)"

        # Create summary object
        return RootCauseSummary(
            root_cause=root_cause,
            confidence=confidence,
            summary=summary_text,
            key_evidence=evidence,
            recommended_actions=list(template["recommended_actions"]),
            severity=final_severity,
            impact=str(template["impact"]),
            time_to_resolution=str(template["time_to_resolution"]),
        )

    def generate_summary_from_prediction(
        self, prediction_result: Dict[str, Any], log_data: Dict[str, str]
    ) -> RootCauseSummary:
        """
        Generate summary from a prediction result dictionary.

        Args:
            prediction_result: Dictionary with prediction results
            log_data: Dictionary with log entry data

        Returns:
            RootCauseSummary object
        """
        return self.generate_summary(
            root_cause=prediction_result.get("root_cause", "Unknown"),
            confidence=prediction_result.get("confidence", 0.0),
            log_message=log_data.get("log_message", ""),
            service=log_data.get("service", ""),
            severity=log_data.get("severity", ""),
            timestamp=log_data.get("timestamp"),
        )


def create_summary_generator() -> SummaryGenerator:
    """Factory function to create a summary generator."""
    return SummaryGenerator()
