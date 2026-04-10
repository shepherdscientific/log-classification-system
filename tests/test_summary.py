"""
Unit tests for summary generation module.
"""

import pytest
import json
import re

from src.inference.summary import SummaryGenerator, RootCauseSummary


class TestRootCauseSummary:
    """Test RootCauseSummary dataclass."""

    def test_root_cause_summary_creation(self):
        """Test creating a RootCauseSummary."""
        summary = RootCauseSummary(
            root_cause="RC-01",
            confidence=0.85,
            summary="Authentication failure detected",
            key_evidence=["401 Unauthorized", "bearer token missing"],
            recommended_actions=[
                "Check API key validity",
                "Verify authentication headers",
            ],
            severity="High",
            impact="Service access blocked",
            time_to_resolution="15-30 minutes",
        )

        assert summary.root_cause == "RC-01"
        assert summary.confidence == 0.85
        assert summary.summary == "Authentication failure detected"
        assert len(summary.key_evidence) == 2
        assert len(summary.recommended_actions) == 2
        assert summary.severity == "High"
        assert summary.impact == "Service access blocked"
        assert summary.time_to_resolution == "15-30 minutes"

    def test_to_dict(self):
        """Test converting RootCauseSummary to dictionary."""
        summary = RootCauseSummary(
            root_cause="RC-01",
            confidence=0.85,
            summary="Test summary",
            key_evidence=["evidence1"],
            recommended_actions=["action1"],
            severity="High",
            impact="Test impact",
            time_to_resolution="30 min",
        )

        result_dict = summary.to_dict()

        assert result_dict["root_cause"] == "RC-01"
        assert result_dict["confidence"] == 0.85
        assert result_dict["summary"] == "Test summary"
        assert result_dict["key_evidence"] == ["evidence1"]
        assert result_dict["recommended_actions"] == ["action1"]
        assert result_dict["severity"] == "High"
        assert result_dict["impact"] == "Test impact"
        assert result_dict["time_to_resolution"] == "30 min"

    def test_to_json(self):
        """Test converting RootCauseSummary to JSON."""
        summary = RootCauseSummary(
            root_cause="RC-01",
            confidence=0.85,
            summary="Test summary",
            key_evidence=["evidence1"],
            recommended_actions=["action1"],
            severity="High",
            impact="Test impact",
            time_to_resolution="30 min",
        )

        json_str = summary.to_json()
        json_data = json.loads(json_str)

        assert json_data["root_cause"] == "RC-01"
        assert json_data["confidence"] == 0.85
        assert "Test summary" in json_str


class TestSummaryGenerator:
    """Test SummaryGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a SummaryGenerator instance."""
        return SummaryGenerator()

    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator is not None
        assert hasattr(generator, "templates")
        assert hasattr(generator, "patterns")

        # Check that templates are loaded for all RC categories
        for rc in [f"RC-{i:02d}" for i in range(1, 9)]:
            assert rc in generator.templates

        # Check template structure
        rc01_template = generator.templates["RC-01"]
        assert "title" in rc01_template
        assert "summary_template" in rc01_template
        assert "key_evidence_patterns" in rc01_template
        assert "recommended_actions" in rc01_template
        assert "severity" in rc01_template
        assert "impact" in rc01_template
        assert "time_to_resolution" in rc01_template

    def test_extract_evidence_rc01(self, generator):
        """Test evidence extraction for RC-01."""
        log_message = "ERROR [api-gateway] 401 Unauthorized — invalid API key provided by client client_8478"

        evidence = generator.extract_evidence(log_message, "RC-01")

        # Should extract evidence patterns
        assert len(evidence) > 0

        # Check for specific evidence patterns
        evidence_text = " ".join(evidence).lower()
        assert any(
            pattern in evidence_text for pattern in ["401", "unauthorized", "api key"]
        )

    def test_extract_evidence_rc02(self, generator):
        """Test evidence extraction for RC-02."""
        log_message = "ERROR [db-pool] Max wait time exceeded (11266ms) waiting for idle connection. Active: 9/15."

        evidence = generator.extract_evidence(log_message, "RC-02")

        assert len(evidence) > 0
        evidence_text = " ".join(evidence).lower()
        assert any(
            pattern in evidence_text
            for pattern in ["wait", "time", "exceeded", "connection"]
        )

    def test_extract_evidence_general_patterns(self, generator):
        """Test extraction of general patterns."""
        log_message = "ERROR [payment-service] Client client_12345 failed with error 500. Memory usage: 85%"

        evidence = generator.extract_evidence(log_message, "RC-01")

        # Should extract general patterns like error codes and percentages
        evidence_text = " ".join(evidence)
        assert any("error_codes" in ev or "metrics" in ev for ev in evidence)

    def test_generate_summary_rc01(self, generator):
        """Test summary generation for RC-01."""
        summary = generator.generate_summary(
            root_cause="RC-01",
            confidence=0.92,
            log_message="WARN [api-gateway] 401 returned to client client_3536: bearer token missing from Authorization header.",
            service="api-gateway",
            severity="High",
            timestamp="2024-09-06T10:22:00Z",
        )

        assert isinstance(summary, RootCauseSummary)
        assert summary.root_cause == "RC-01"
        assert summary.confidence == 0.92
        assert summary.severity == "High"
        assert len(summary.key_evidence) > 0
        assert len(summary.recommended_actions) > 0
        assert (
            "Authentication" in summary.summary
            or "authorization" in summary.summary.lower()
        )

        # Check that evidence is included in summary
        summary_text = summary.summary.lower()
        assert any(ev.lower() in summary_text for ev in summary.key_evidence[:2])

    def test_generate_summary_rc02(self, generator):
        """Test summary generation for RC-02."""
        summary = generator.generate_summary(
            root_cause="RC-02",
            confidence=0.87,
            log_message="ERROR [db-pool] Max wait time exceeded (11266ms) waiting for idle connection. Active: 9/15.",
            service="db-pool",
            severity="Critical",
            timestamp="2024-04-10T08:46:00Z",
        )

        assert summary.root_cause == "RC-02"
        assert summary.severity == "Critical"
        assert "Database" in summary.summary or "connection" in summary.summary.lower()
        assert "db-pool" in summary.summary or "Service: db-pool" in summary.summary

    def test_generate_summary_low_confidence(self, generator):
        """Test summary generation with low confidence."""
        summary = generator.generate_summary(
            root_cause="RC-03",
            confidence=0.35,  # Low confidence
            log_message="ERROR [payment-gateway] Upstream provider Twilio returned 502. Retried 1 times. Failing over.",
            service="payment-gateway",
            severity="High",
            timestamp="2024-05-28T21:04:00Z",
        )

        assert summary.root_cause == "RC-03"
        # Severity should indicate low confidence
        assert "(Low confidence prediction)" in summary.severity

    def test_generate_summary_unknown_root_cause(self, generator):
        """Test summary generation for unknown root cause."""
        summary = generator.generate_summary(
            root_cause="RC-99",  # Unknown root cause
            confidence=0.75,
            log_message="Some unknown error occurred",
            service="test-service",
            severity="Medium",
            timestamp="2024-01-01T00:00:00Z",
        )

        assert summary.root_cause == "RC-99"
        assert (
            "Root cause identified" in summary.summary
            or "Root Cause RC-99" in summary.summary
        )
        assert len(summary.recommended_actions) > 0

    def test_generate_summary_from_prediction(self, generator):
        """Test generating summary from prediction result."""
        prediction_result = {"root_cause": "RC-01", "confidence": 0.85}

        log_data = {
            "log_message": "401 Unauthorized — invalid API key",
            "service": "api-gateway",
            "severity": "High",
            "timestamp": "2024-09-06T10:22:00Z",
        }

        summary = generator.generate_summary_from_prediction(
            prediction_result, log_data
        )

        assert summary.root_cause == "RC-01"
        assert summary.confidence == 0.85
        assert (
            "api-gateway" in summary.summary
            or "Service: api-gateway" in summary.summary
        )

    def test_pattern_matching(self, generator):
        """Test regex pattern matching for evidence extraction."""
        test_cases = [
            ("RC-01", "401 Unauthorized error", ["401.*Unauthorized"]),
            ("RC-02", "Connection pool exhausted", ["connection.*exhausted"]),
            (
                "RC-03",
                "Upstream provider Twilio returned 502",
                ["upstream.*provider.*returned"],
            ),
            (
                "RC-04",
                "Rate limit hit with Retry-After: 120s",
                ["rate limit.*hit", "Retry-After"],
            ),
            ("RC-05", "Constraint violation in database", ["constraint.*violation"]),
            (
                "RC-06",
                "Privilege escalation attempt blocked",
                ["privilege.*escalation"],
            ),
            ("RC-07", "Memory limit reached 2457MB", ["memory.*limit.*reached"]),
            ("RC-08", "Packet loss 97% detected", ["packet.*loss"]),
        ]

        for rc, message, expected_patterns in test_cases:
            evidence = generator.extract_evidence(message, rc)
            evidence_text = " ".join(evidence).lower()

            # Check that at least one expected pattern is found
            pattern_found = False
            for pattern in expected_patterns:
                if re.search(pattern, evidence_text, re.IGNORECASE):
                    pattern_found = True
                    break

            assert pattern_found, (
                f"No expected patterns found for {rc}: {evidence_text}"
            )

    def test_summary_includes_context(self, generator):
        """Test that summary includes service and severity context."""
        summary = generator.generate_summary(
            root_cause="RC-01",
            confidence=0.9,
            log_message="401 Unauthorized",
            service="auth-service",
            severity="High",
            timestamp="2024-01-01T12:00:00Z",
        )

        summary_text = summary.summary
        assert "auth-service" in summary_text or "Service: auth-service" in summary_text
        assert "High" in summary_text or "Severity: High" in summary_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
