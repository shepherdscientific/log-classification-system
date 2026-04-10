"""
Root cause category analysis module.
Analyzes patterns in log messages for each RC category to understand distinguishing features.
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from pathlib import Path


@dataclass
class CategoryPattern:
    """Pattern analysis for a specific root cause category."""

    root_cause: str
    common_patterns: List[str]
    key_phrases: List[str]
    error_types: List[str]
    typical_system_issues: List[str]
    category_description: str
    example_logs: List[str]
    service_distribution: Dict[str, int]
    severity_distribution: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)

        # Convert numpy types to Python native types for JSON serialization
        if self.service_distribution:
            result["service_distribution"] = {
                str(k): int(v) for k, v in self.service_distribution.items()
            }

        if self.severity_distribution:
            result["severity_distribution"] = {
                str(k): int(v) for k, v in self.severity_distribution.items()
            }

        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class RootCauseCategoryAnalyzer:
    """
    Analyzes patterns in log messages for each RC category.

    Identifies common patterns, key phrases, error types, and maps
    root causes to typical system issues for each category.
    """

    def __init__(self):
        """Initialize the category analyzer with predefined patterns."""
        self.root_causes = [f"RC-{i:02d}" for i in range(1, 9)]
        self.pattern_definitions = self._load_pattern_definitions()
        self.error_type_patterns = self._load_error_type_patterns()

    def _load_pattern_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load pattern definitions for each root cause category."""
        return {
            "RC-01": {
                "name": "Authentication/Authorization Failure",
                "description": "Issues related to user authentication, authorization tokens, API keys, or access control.",
                "key_phrases": [
                    "401",
                    "Unauthorized",
                    "bearer token",
                    "API key",
                    "authentication failed",
                    "HMAC signature",
                    "scope mismatch",
                    "insufficient permissions",
                ],
                "error_types": ["AuthError", "PermissionError", "TokenError"],
                "typical_issues": [
                    "Expired or invalid authentication tokens",
                    "Missing or malformed authorization headers",
                    "Insufficient user permissions for requested operation",
                    "API key validation failures",
                    "IP address not in allowlist",
                ],
            },
            "RC-02": {
                "name": "Database Connection Pool Exhaustion",
                "description": "Database connectivity issues including connection pool exhaustion, timeouts, and slow queries.",
                "key_phrases": [
                    "connection exhausted",
                    "wait time exceeded",
                    "failed to acquire connection",
                    "database unreachable",
                    "slow query",
                    "DB connection",
                    "connection pool",
                ],
                "error_types": [
                    "ConnectionError",
                    "TimeoutError",
                    "PoolExhaustionError",
                ],
                "typical_issues": [
                    "Database connection pool size insufficient for load",
                    "Long-running queries blocking connections",
                    "Database server resource constraints",
                    "Network connectivity issues to database",
                    "Connection leak in application code",
                ],
            },
            "RC-03": {
                "name": "Upstream Service Failure",
                "description": "Failures in external dependencies or upstream services that the system relies on.",
                "key_phrases": [
                    "upstream provider",
                    "502 Bad Gateway",
                    "503 Service Unavailable",
                    "failing over",
                    "dependency failure",
                    "partner API",
                    "external service",
                ],
                "error_types": [
                    "DependencyError",
                    "ServiceUnavailableError",
                    "GatewayError",
                ],
                "typical_issues": [
                    "Third-party API outages or degraded performance",
                    "Network issues between services",
                    "Upstream service rate limiting",
                    "Incompatible API version changes",
                    "External service authentication failures",
                ],
            },
            "RC-04": {
                "name": "Rate Limiting or Throttling",
                "description": "Requests exceeding rate limits, burst limits, or throttling thresholds.",
                "key_phrases": [
                    "rate limit",
                    "burst limit",
                    "429 Too Many Requests",
                    "Retry-After",
                    "throttling",
                    "RPS exceeded",
                    "quota exceeded",
                ],
                "error_types": ["RateLimitError", "ThrottlingError", "QuotaError"],
                "typical_issues": [
                    "Client making excessive requests beyond allowed limits",
                    "Misconfigured rate limiting policies",
                    "Distributed denial of service (DDoS) attempts",
                    "Legitimate traffic spikes exceeding capacity",
                    "Cascading failures due to retry storms",
                ],
            },
            "RC-05": {
                "name": "Data Validation or Schema Mismatch",
                "description": "Issues with data validation, schema compliance, or format errors in input data.",
                "key_phrases": [
                    "validation error",
                    "schema validation",
                    "constraint violation",
                    "null value",
                    "format error",
                    "type mismatch",
                    "required field",
                ],
                "error_types": ["ValidationError", "SchemaError", "ConstraintError"],
                "typical_issues": [
                    "Malformed or incomplete data from clients",
                    "Schema drift between services",
                    "Missing required fields in API requests",
                    "Data type mismatches (string vs number, etc.)",
                    "Business rule violations in transaction data",
                ],
            },
            "RC-06": {
                "name": "Security Policy Violation",
                "description": "Security-related issues including privilege escalation attempts, access violations, or policy breaches.",
                "key_phrases": [
                    "privilege escalation",
                    "security policy",
                    "access violation",
                    "insufficient role",
                    "blocked attempt",
                    "unauthorized access",
                ],
                "error_types": ["SecurityError", "PolicyViolationError", "AccessError"],
                "typical_issues": [
                    "Attempted access beyond user's permission level",
                    "Security policy configuration errors",
                    "Role-based access control (RBAC) misconfigurations",
                    "Malicious activity detection",
                    "Session hijacking or token theft attempts",
                ],
            },
            "RC-07": {
                "name": "Resource Exhaustion",
                "description": "System resource limitations including memory, disk space, file descriptors, or CPU constraints.",
                "key_phrases": [
                    "memory limit",
                    "disk full",
                    "file descriptor",
                    "OOM error",
                    "resource exhausted",
                    "capacity reached",
                    "heap exhausted",
                ],
                "error_types": ["ResourceError", "OOMError", "CapacityError"],
                "typical_issues": [
                    "Memory leaks in application code",
                    "Insufficient disk space for logs or data",
                    "File descriptor limits reached",
                    "Cache eviction due to memory pressure",
                    "Inadequate resource provisioning for workload",
                ],
            },
            "RC-08": {
                "name": "Network Connectivity Issues",
                "description": "Network-related problems including packet loss, connection timeouts, DNS failures, or TLS handshake issues.",
                "key_phrases": [
                    "packet loss",
                    "connection timeout",
                    "TLS handshake",
                    "network segment",
                    "DNS resolution",
                    "unreachable",
                    "timeout after",
                ],
                "error_types": ["NetworkError", "TimeoutError", "ConnectivityError"],
                "typical_issues": [
                    "Network infrastructure failures (routers, switches)",
                    "DNS server outages or misconfigurations",
                    "TLS certificate expiration or validation failures",
                    "Network congestion or bandwidth limitations",
                    "Firewall rules blocking legitimate traffic",
                ],
            },
        }

    def _load_error_type_patterns(self) -> Dict[str, List[str]]:
        """Load regex patterns for identifying error types in log messages."""
        return {
            "AuthError": [
                r"401.*Unauthorized",
                r"bearer token.*missing",
                r"invalid.*API.*key",
                r"authentication.*failed",
                r"HMAC.*signature.*mismatch",
            ],
            "PermissionError": [
                r"insufficient.*permissions?",
                r"access.*denied",
                r"scope.*mismatch",
                r"privilege.*escalation",
            ],
            "ConnectionError": [
                r"connection.*exhausted",
                r"failed.*acquire.*connection",
                r"database.*unreachable",
                r"connection.*timeout",
            ],
            "TimeoutError": [
                r"wait.*time.*exceeded",
                r"timeout.*after",
                r"slow.*query",
                r"handshake.*timeout",
            ],
            "DependencyError": [
                r"upstream.*provider",
                r"dependency.*failure",
                r"external.*service",
                r"partner.*API",
            ],
            "RateLimitError": [
                r"rate.*limit",
                r"burst.*limit",
                r"429.*Too Many Requests",
                r"RPS.*exceeded",
            ],
            "ValidationError": [
                r"validation.*error",
                r"schema.*validation",
                r"constraint.*violation",
                r"format.*error",
            ],
            "ResourceError": [
                r"memory.*limit",
                r"disk.*full",
                r"file.*descriptor",
                r"resource.*exhausted",
            ],
            "NetworkError": [
                r"packet.*loss",
                r"network.*segment",
                r"DNS.*resolution",
                r"TLS.*handshake",
            ],
        }

    def analyze_dataset(self, data: pd.DataFrame) -> Dict[str, CategoryPattern]:
        """
        Analyze the dataset to extract patterns for each root cause category.

        Args:
            data: DataFrame containing log data with columns:
                  'log_message', 'service', 'severity', 'root_cause_label'

        Returns:
            Dictionary mapping root cause labels to CategoryPattern objects
        """
        results = {}

        for rc in self.root_causes:
            # Filter data for this root cause
            rc_data = data[data["root_cause_label"] == rc]

            if len(rc_data) == 0:
                continue

            # Extract common patterns from log messages
            common_patterns = self._extract_common_patterns(
                rc_data["log_message"].tolist()
            )

            # Extract key phrases
            key_phrases = self._extract_key_phrases(rc_data["log_message"].tolist())

            # Identify error types
            error_types = self._identify_error_types(rc_data["log_message"].tolist())

            # Get typical system issues from pattern definitions
            typical_issues = self.pattern_definitions.get(rc, {}).get(
                "typical_issues", []
            )

            # Create category description
            category_description = self._create_category_description(rc, rc_data)

            # Get example logs
            example_logs = rc_data["log_message"].head(3).tolist()

            # Calculate service distribution
            service_distribution = dict(rc_data["service"].value_counts())

            # Calculate severity distribution
            severity_distribution = dict(rc_data["severity"].value_counts())

            # Create CategoryPattern object
            pattern = CategoryPattern(
                root_cause=rc,
                common_patterns=common_patterns,
                key_phrases=key_phrases,
                error_types=error_types,
                typical_system_issues=typical_issues,
                category_description=category_description,
                example_logs=example_logs,
                service_distribution=service_distribution,
                severity_distribution=severity_distribution,
            )

            results[rc] = pattern

        return results

    def _extract_common_patterns(self, log_messages: List[str]) -> List[str]:
        """Extract common patterns from log messages."""
        patterns = []

        # Common log patterns to look for
        common_patterns = [
            (r"ERROR.*\[.*\].*", "Error with service context"),
            (r"WARN.*\[.*\].*", "Warning with service context"),
            (r"CRITICAL.*\[.*\].*", "Critical error with service context"),
            (r".*returned.*\d{3}.*", "HTTP status code returned"),
            (r".*timeout.*\d+.*ms.*", "Timeout with duration"),
            (r".*limit.*reached.*", "Limit reached pattern"),
            (r".*failed.*", "Failure pattern"),
            (r".*error.*", "Error pattern"),
            (r".*violation.*", "Violation pattern"),
            (r".*validation.*", "Validation pattern"),
        ]

        for pattern, description in common_patterns:
            matches = [
                msg for msg in log_messages if re.search(pattern, msg, re.IGNORECASE)
            ]
            if len(matches) > len(log_messages) * 0.3:  # At least 30% of logs match
                patterns.append(description)

        return patterns

    def _extract_key_phrases(self, log_messages: List[str]) -> List[str]:
        """Extract key phrases from log messages."""
        # Common technical terms to look for
        technical_terms = [
            "timeout",
            "limit",
            "failed",
            "error",
            "violation",
            "validation",
            "connection",
            "memory",
            "disk",
            "authentication",
            "authorization",
            "rate",
            "burst",
            "quota",
            "schema",
            "constraint",
            "privilege",
            "escalation",
            "network",
            "packet",
            "DNS",
            "TLS",
            "handshake",
            "upstream",
            "dependency",
            "provider",
            "service",
            "API",
            "database",
            "pool",
            "exhausted",
            "resource",
            "capacity",
            "OOM",
            "heap",
            "401",
            "unauthorized",
            "bearer token",
            "API key",
            "HMAC",
            "signature",
            "scope",
            "permissions",
            "502",
            "503",
            "bad gateway",
            "service unavailable",
            "429",
            "too many requests",
            "retry-after",
            "RPS",
            "throttling",
            "null",
            "format",
            "type mismatch",
            "required field",
            "security",
            "policy",
            "access violation",
            "blocked",
            "file descriptor",
            "full",
            "packet loss",
            "timeout after",
            "TLS handshake",
            "network segment",
        ]

        phrase_counter: Counter[str] = Counter()
        for msg in log_messages:
            msg_lower = msg.lower()
            for term in technical_terms:
                if term in msg_lower:
                    phrase_counter[term] += 1

        # Return top phrases that appear in at least 20% of logs
        threshold = max(
            1, len(log_messages) * 0.2
        )  # At least 1 match for small datasets
        return [
            phrase
            for phrase, count in phrase_counter.most_common()
            if count >= threshold
        ]

    def _identify_error_types(self, log_messages: List[str]) -> List[str]:
        """Identify error types in log messages using regex patterns."""
        error_type_counts: Counter[str] = Counter()

        for error_type, patterns in self.error_type_patterns.items():
            for pattern in patterns:
                matches = [
                    msg
                    for msg in log_messages
                    if re.search(pattern, msg, re.IGNORECASE)
                ]
                if matches:
                    error_type_counts[error_type] += len(matches)

        # Return error types that appear in at least 30% of logs
        threshold = len(log_messages) * 0.3
        return [
            error_type
            for error_type, count in error_type_counts.most_common()
            if count >= threshold
        ]

    def _create_category_description(
        self, root_cause: str, rc_data: pd.DataFrame
    ) -> str:
        """Create a comprehensive description for a root cause category."""
        pattern_def = self.pattern_definitions.get(root_cause, {})
        name = pattern_def.get("name", root_cause)
        base_description = pattern_def.get("description", "")

        # Add statistics
        total_logs = len(rc_data)
        services = rc_data["service"].nunique()
        severities = rc_data["severity"].unique().tolist()

        # Get most common service
        most_common_service = rc_data["service"].mode()
        most_common_service_str = (
            most_common_service[0] if len(most_common_service) > 0 else "various"
        )

        # Get most common severity
        most_common_severity = rc_data["severity"].mode()
        most_common_severity_str = (
            most_common_severity[0] if len(most_common_severity) > 0 else "mixed"
        )

        description = f"{name}. {base_description} "
        description += f"This category contains {total_logs} log entries across {services} different services. "
        description += f"Most commonly affects the {most_common_service_str} service with {most_common_severity_str} severity levels."

        return description

    def generate_category_report(
        self, data: pd.DataFrame, output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report of category analysis.

        Args:
            data: DataFrame containing log data
            output_path: Optional path to save JSON report

        Returns:
            Dictionary containing the full analysis report
        """
        # Analyze all categories
        category_patterns = self.analyze_dataset(data)

        # Convert to serializable format
        report = {
            "summary": {
                "total_categories_analyzed": len(category_patterns),
                "categories": list(category_patterns.keys()),
            },
            "category_details": {
                rc: pattern.to_dict() for rc, pattern in category_patterns.items()
            },
            "cross_category_analysis": self._perform_cross_category_analysis(
                category_patterns
            ),
        }

        # Save to file if output path provided
        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

        return report

    def _perform_cross_category_analysis(
        self, category_patterns: Dict[str, CategoryPattern]
    ) -> Dict[str, Any]:
        """Perform analysis across categories to identify similarities and differences."""
        # Collect all key phrases across categories
        all_key_phrases = set()
        for pattern in category_patterns.values():
            all_key_phrases.update(pattern.key_phrases)

        # Find overlapping phrases between categories
        overlapping_phrases: Dict[str, List[str]] = defaultdict(list)
        for rc1, pattern1 in category_patterns.items():
            for rc2, pattern2 in category_patterns.items():
                if rc1 >= rc2:
                    continue
                overlap = set(pattern1.key_phrases) & set(pattern2.key_phrases)
                if overlap:
                    overlapping_phrases[f"{rc1}-{rc2}"] = list(overlap)

        # Find distinctive phrases for each category
        distinctive_phrases = {}
        for rc, pattern in category_patterns.items():
            other_phrases = set()
            for other_rc, other_pattern in category_patterns.items():
                if rc != other_rc:
                    other_phrases.update(other_pattern.key_phrases)

            distinctive = set(pattern.key_phrases) - other_phrases
            if distinctive:
                distinctive_phrases[rc] = list(distinctive)

        return {
            "total_unique_key_phrases": len(all_key_phrases),
            "overlapping_phrases_between_categories": dict(overlapping_phrases),
            "distinctive_phrases_per_category": distinctive_phrases,
            "category_similarity_matrix": self._calculate_category_similarity(
                category_patterns
            ),
        }

    def _calculate_category_similarity(
        self, category_patterns: Dict[str, CategoryPattern]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate similarity matrix between categories based on key phrases."""
        similarity_matrix: Dict[str, Dict[str, float]] = {}

        for rc1, pattern1 in category_patterns.items():
            similarity_matrix[rc1] = {}
            phrases1 = set(pattern1.key_phrases)

            for rc2, pattern2 in category_patterns.items():
                phrases2 = set(pattern2.key_phrases)

                if not phrases1 and not phrases2:
                    similarity = 0.0
                else:
                    # Jaccard similarity
                    intersection = len(phrases1 & phrases2)
                    union = len(phrases1 | phrases2)
                    similarity = intersection / union if union > 0 else 0.0

                similarity_matrix[rc1][rc2] = round(similarity, 3)

        return similarity_matrix

    def get_category_descriptions(self) -> Dict[str, str]:
        """Get concise descriptions for each root cause category."""
        descriptions = {}
        for rc in self.root_causes:
            pattern_def = self.pattern_definitions.get(rc, {})
            name = pattern_def.get("name", rc)
            desc = pattern_def.get("description", "")
            descriptions[rc] = f"{name}: {desc}"

        return descriptions

    def map_log_to_category_patterns(self, log_message: str) -> Dict[str, float]:
        """
        Map a log message to root cause categories based on pattern matching.

        Args:
            log_message: The log message to analyze

        Returns:
            Dictionary mapping root cause labels to match scores (0-1)
        """
        scores = {}
        log_lower = log_message.lower()

        for rc in self.root_causes:
            pattern_def = self.pattern_definitions.get(rc, {})
            key_phrases = pattern_def.get("key_phrases", [])

            # Calculate match score based on key phrases
            matches = 0
            for phrase in key_phrases:
                if phrase.lower() in log_lower:
                    matches += 1

            # Normalize score
            score = matches / len(key_phrases) if key_phrases else 0
            scores[rc] = round(score, 3)

        return scores
