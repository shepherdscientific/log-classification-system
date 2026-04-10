"""
Unit tests for root cause category analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile

from src.evaluation.category_analysis import CategoryPattern, RootCauseCategoryAnalyzer


class TestCategoryPattern:
    """Tests for CategoryPattern dataclass."""

    def test_category_pattern_creation(self):
        """Test creating a CategoryPattern object."""
        pattern = CategoryPattern(
            root_cause="RC-01",
            common_patterns=["Error with service context", "HTTP status code returned"],
            key_phrases=["401", "Unauthorized", "bearer token"],
            error_types=["AuthError", "PermissionError"],
            typical_system_issues=[
                "Expired or invalid authentication tokens",
                "Missing or malformed authorization headers",
            ],
            category_description="Authentication/Authorization Failure category",
            example_logs=[
                "ERROR [api-gateway] 401 returned to client",
                "WARN [api-gateway] bearer token missing",
            ],
            service_distribution={"api-gateway": 5, "iam-middleware": 3},
            severity_distribution={"High": 6, "Medium": 2},
        )

        assert pattern.root_cause == "RC-01"
        assert len(pattern.common_patterns) == 2
        assert len(pattern.key_phrases) == 3
        assert len(pattern.error_types) == 2
        assert len(pattern.typical_system_issues) == 2
        assert "Authentication/Authorization Failure" in pattern.category_description
        assert len(pattern.example_logs) == 2
        assert pattern.service_distribution["api-gateway"] == 5
        assert pattern.severity_distribution["High"] == 6

    def test_to_dict(self):
        """Test converting CategoryPattern to dictionary."""
        pattern = CategoryPattern(
            root_cause="RC-01",
            common_patterns=["pattern1"],
            key_phrases=["phrase1"],
            error_types=["AuthError"],
            typical_system_issues=["issue1"],
            category_description="Test category",
            example_logs=["log1"],
            service_distribution={"service1": 1},
            severity_distribution={"High": 1},
        )

        result = pattern.to_dict()

        assert isinstance(result, dict)
        assert result["root_cause"] == "RC-01"
        assert result["common_patterns"] == ["pattern1"]
        assert result["key_phrases"] == ["phrase1"]
        assert result["error_types"] == ["AuthError"]
        assert result["typical_system_issues"] == ["issue1"]
        assert result["category_description"] == "Test category"
        assert result["example_logs"] == ["log1"]
        assert result["service_distribution"] == {"service1": 1}
        assert result["severity_distribution"] == {"High": 1}

    def test_to_json(self):
        """Test converting CategoryPattern to JSON."""
        pattern = CategoryPattern(
            root_cause="RC-01",
            common_patterns=["pattern1"],
            key_phrases=["phrase1"],
            error_types=["AuthError"],
            typical_system_issues=["issue1"],
            category_description="Test category",
            example_logs=["log1"],
            service_distribution={"service1": 1},
            severity_distribution={"High": 1},
        )

        json_str = pattern.to_json()

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["root_cause"] == "RC-01"


class TestRootCauseCategoryAnalyzer:
    """Tests for RootCauseCategoryAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a RootCauseCategoryAnalyzer instance."""
        return RootCauseCategoryAnalyzer()

    @pytest.fixture
    def sample_data(self):
        """Create sample log data for testing."""
        data = {
            "log_message": [
                "ERROR [api-gateway] 401 returned to client: bearer token missing",
                "ERROR [db-pool] Max wait time exceeded waiting for idle connection",
                "ERROR [fx-rate-fetcher] Upstream provider returned 502",
                "WARN [api-gateway] Rate limit hit on endpoint",
                "ERROR [transaction-validator] Constraint violation",
                "CRITICAL [audit-service] Privilege escalation attempt blocked",
                "ERROR [log-aggregator] Disk write failed: volume at 87% capacity",
                "CRITICAL [vpc-gateway] Packet loss 97% detected",
            ],
            "service": [
                "api-gateway",
                "db-pool",
                "fx-rate-fetcher",
                "api-gateway",
                "transaction-validator",
                "audit-service",
                "log-aggregator",
                "vpc-gateway",
            ],
            "severity": [
                "High",
                "Critical",
                "High",
                "Medium",
                "Medium",
                "High",
                "Critical",
                "High",
            ],
            "root_cause_label": [
                "RC-01",
                "RC-02",
                "RC-03",
                "RC-04",
                "RC-05",
                "RC-06",
                "RC-07",
                "RC-08",
            ],
        }
        return pd.DataFrame(data)

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert len(analyzer.root_causes) == 8
        assert "RC-01" in analyzer.root_causes
        assert "RC-08" in analyzer.root_causes
        assert "RC-01" in analyzer.pattern_definitions
        assert "RC-08" in analyzer.pattern_definitions
        assert "AuthError" in analyzer.error_type_patterns
        assert "NetworkError" in analyzer.error_type_patterns

    def test_analyze_dataset(self, analyzer, sample_data):
        """Test analyzing a dataset."""
        results = analyzer.analyze_dataset(sample_data)

        assert isinstance(results, dict)
        assert len(results) == 8  # All 8 root causes should be present

        # Check each root cause has a CategoryPattern
        for rc in analyzer.root_causes:
            assert rc in results
            pattern = results[rc]
            assert isinstance(pattern, CategoryPattern)
            assert pattern.root_cause == rc

        # Check specific patterns
        rc01_pattern = results["RC-01"]
        assert (
            "Authentication/Authorization Failure" in rc01_pattern.category_description
        )
        # With small test dataset, key phrases might be limited
        assert len(rc01_pattern.key_phrases) > 0
        assert (
            "401" in rc01_pattern.key_phrases
            or "bearer token" in rc01_pattern.key_phrases
        )

        rc02_pattern = results["RC-02"]
        assert (
            "Database Connection Pool Exhaustion" in rc02_pattern.category_description
        )
        assert (
            "connection" in rc02_pattern.key_phrases
            or "wait time" in rc01_pattern.key_phrases
        )

    def test_extract_common_patterns(self, analyzer):
        """Test extracting common patterns from log messages."""
        log_messages = [
            "ERROR [service1] Something failed with code 500",
            "WARN [service2] Rate limit exceeded for client",
            "ERROR [service3] Database connection timeout",
            "CRITICAL [service4] Memory limit reached",
        ]

        patterns = analyzer._extract_common_patterns(log_messages)

        assert isinstance(patterns, list)
        # Should find at least "Error with service context" pattern
        assert any("service context" in p for p in patterns)

    def test_extract_key_phrases(self, analyzer):
        """Test extracting key phrases from log messages."""
        log_messages = [
            "ERROR: Authentication failed for user",
            "WARN: Database connection timeout after 5000ms",
            "CRITICAL: Memory limit exceeded, OOM error",
        ]

        phrases = analyzer._extract_key_phrases(log_messages)

        assert isinstance(phrases, list)
        # Should find common technical terms
        assert any(
            term in phrases for term in ["authentication", "timeout", "memory", "error"]
        )

    def test_identify_error_types(self, analyzer):
        """Test identifying error types in log messages."""
        log_messages = [
            "401 Unauthorized: bearer token missing",
            "Failed to acquire DB connection from pool",
            "Rate limit hit on endpoint",
        ]

        error_types = analyzer._identify_error_types(log_messages)

        assert isinstance(error_types, list)
        # Should identify AuthError from first message
        assert "AuthError" in error_types or "ConnectionError" in error_types

    def test_create_category_description(self, analyzer, sample_data):
        """Test creating category descriptions."""
        rc01_data = sample_data[sample_data["root_cause_label"] == "RC-01"]
        description = analyzer._create_category_description("RC-01", rc01_data)

        assert isinstance(description, str)
        assert "Authentication/Authorization Failure" in description
        assert "log entries" in description
        assert "services" in description

    def test_generate_category_report(self, analyzer, sample_data):
        """Test generating a category report."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            report = analyzer.generate_category_report(sample_data, temp_path)

            assert isinstance(report, dict)
            assert "summary" in report
            assert "category_details" in report
            assert "cross_category_analysis" in report

            # Check summary
            summary = report["summary"]
            assert summary["total_categories_analyzed"] == 8
            assert len(summary["categories"]) == 8

            # Check category details
            category_details = report["category_details"]
            assert len(category_details) == 8
            assert "RC-01" in category_details
            assert "RC-08" in category_details

            # Check cross category analysis
            cross_analysis = report["cross_category_analysis"]
            assert "total_unique_key_phrases" in cross_analysis
            assert "overlapping_phrases_between_categories" in cross_analysis
            assert "distinctive_phrases_per_category" in cross_analysis
            assert "category_similarity_matrix" in cross_analysis

            # Verify file was created
            assert Path(temp_path).exists()
            with open(temp_path, "r") as f:
                saved_report = json.load(f)
            assert saved_report["summary"]["total_categories_analyzed"] == 8

        finally:
            Path(temp_path).unlink()

    def test_get_category_descriptions(self, analyzer):
        """Test getting category descriptions."""
        descriptions = analyzer.get_category_descriptions()

        assert isinstance(descriptions, dict)
        assert len(descriptions) == 8

        for rc in analyzer.root_causes:
            assert rc in descriptions
            desc = descriptions[rc]
            assert isinstance(desc, str)
            assert ":" in desc  # Should have format "Name: Description"

    def test_map_log_to_category_patterns(self, analyzer):
        """Test mapping a log message to category patterns."""
        # Test with authentication log
        auth_log = "ERROR [api-gateway] 401 Unauthorized: bearer token missing from Authorization header"
        scores = analyzer.map_log_to_category_patterns(auth_log)

        assert isinstance(scores, dict)
        assert len(scores) == 8

        # RC-01 (Authentication) should have highest score
        assert scores["RC-01"] > 0
        # Other categories should have lower scores
        assert scores["RC-01"] >= max(scores.values())

        # Test with database connection log
        db_log = "ERROR [db-pool] Failed to acquire DB connection from pool: all connections exhausted"
        db_scores = analyzer.map_log_to_category_patterns(db_log)

        # RC-02 (Database Connection) should have highest score
        assert db_scores["RC-02"] > 0
        assert db_scores["RC-02"] >= max(db_scores.values())

    def test_perform_cross_category_analysis(self, analyzer, sample_data):
        """Test performing cross-category analysis."""
        category_patterns = analyzer.analyze_dataset(sample_data)
        cross_analysis = analyzer._perform_cross_category_analysis(category_patterns)

        assert isinstance(cross_analysis, dict)
        assert "total_unique_key_phrases" in cross_analysis
        assert "overlapping_phrases_between_categories" in cross_analysis
        assert "distinctive_phrases_per_category" in cross_analysis
        assert "category_similarity_matrix" in cross_analysis

        # Check similarity matrix structure
        similarity_matrix = cross_analysis["category_similarity_matrix"]
        for rc1 in analyzer.root_causes:
            assert rc1 in similarity_matrix
            for rc2 in analyzer.root_causes:
                assert rc2 in similarity_matrix[rc1]
                similarity = similarity_matrix[rc1][rc2]
                assert 0 <= similarity <= 1
                # Self-similarity should be 1
                if rc1 == rc2:
                    assert similarity == 1.0

    def test_calculate_category_similarity(self, analyzer, sample_data):
        """Test calculating category similarity."""
        category_patterns = analyzer.analyze_dataset(sample_data)
        similarity_matrix = analyzer._calculate_category_similarity(category_patterns)

        assert isinstance(similarity_matrix, dict)
        for rc1 in similarity_matrix:
            assert isinstance(similarity_matrix[rc1], dict)
            for rc2 in similarity_matrix[rc1]:
                similarity = similarity_matrix[rc1][rc2]
                assert isinstance(similarity, float)
                assert 0 <= similarity <= 1

    def test_empty_dataset(self, analyzer):
        """Test analyzing an empty dataset."""
        empty_data = pd.DataFrame(
            columns=["log_message", "service", "severity", "root_cause_label"]
        )
        results = analyzer.analyze_dataset(empty_data)

        assert isinstance(results, dict)
        assert len(results) == 0

    def test_partial_dataset(self, analyzer):
        """Test analyzing a dataset with only some root causes."""
        partial_data = pd.DataFrame(
            {
                "log_message": ["ERROR: Auth failed", "ERROR: DB connection failed"],
                "service": ["api-gateway", "db-pool"],
                "severity": ["High", "Critical"],
                "root_cause_label": ["RC-01", "RC-02"],
            }
        )

        results = analyzer.analyze_dataset(partial_data)

        assert isinstance(results, dict)
        assert len(results) == 2
        assert "RC-01" in results
        assert "RC-02" in results
        assert "RC-03" not in results

    def test_pattern_definitions_completeness(self, analyzer):
        """Test that pattern definitions cover all root causes."""
        for rc in analyzer.root_causes:
            assert rc in analyzer.pattern_definitions
            pattern_def = analyzer.pattern_definitions[rc]

            assert "name" in pattern_def
            assert "description" in pattern_def
            assert "key_phrases" in pattern_def
            assert "error_types" in pattern_def
            assert "typical_issues" in pattern_def

            assert isinstance(pattern_def["name"], str)
            assert isinstance(pattern_def["description"], str)
            assert isinstance(pattern_def["key_phrases"], list)
            assert isinstance(pattern_def["error_types"], list)
            assert isinstance(pattern_def["typical_issues"], list)

            assert len(pattern_def["key_phrases"]) > 0
            assert len(pattern_def["typical_issues"]) > 0
