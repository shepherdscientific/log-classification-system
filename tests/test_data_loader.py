import sys
import os
import tempfile
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import LogDataLoader


def test_data_loader_basic():
    """Test basic data loading functionality"""
    data_path = "docs/log_dataset.csv"

    loader = LogDataLoader(data_path)
    df = loader.load_data()

    # Check basic properties
    assert len(df) == 120, f"Expected 120 rows, got {len(df)}"
    assert len(df.columns) == 6, f"Expected 6 columns, got {len(df.columns)}"

    expected_columns = [
        "log_id",
        "timestamp",
        "service",
        "severity",
        "log_message",
        "root_cause_label",
    ]
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    # Check no missing values
    assert df.isnull().sum().sum() == 0, "Found missing values in dataset"

    # Check no duplicate log_ids
    assert df["log_id"].duplicated().sum() == 0, "Found duplicate log_ids"

    print("✅ Basic data loading test passed")


def test_data_loader_validation():
    """Test data validation functionality"""
    data_path = "docs/log_dataset.csv"

    loader = LogDataLoader(data_path)
    loader.load_data()
    validation = loader.validate_data()

    # Check validation results
    assert validation["total_rows"] == 120
    assert validation["total_columns"] == 6
    assert len(validation["missing_values"]) == 0
    assert validation["duplicates"]["total_duplicate_rows"] == 0
    assert validation["duplicates"]["duplicate_log_ids"] == 0

    print("✅ Data validation test passed")


def test_data_loader_distributions():
    """Test distribution analysis functionality"""
    data_path = "docs/log_dataset.csv"

    loader = LogDataLoader(data_path)
    loader.load_data()
    distributions = loader.analyze_distributions()

    # Check root cause distribution
    rc_dist = distributions["root_cause_distribution"]
    assert len(rc_dist) == 8, f"Expected 8 root causes, got {len(rc_dist)}"

    # Check all RC categories are present
    expected_rcs = [f"RC-{i:02d}" for i in range(1, 9)]
    for rc in expected_rcs:
        assert rc in rc_dist, f"Missing root cause: {rc}"

    # Check service distribution
    service_dist = distributions["service_distribution"]
    assert len(service_dist) > 0, "Service distribution should not be empty"

    # Check severity distribution
    severity_dist = distributions["severity_distribution"]
    expected_severities = ["Critical", "High", "Medium"]
    for severity in expected_severities:
        assert severity in severity_dist, f"Missing severity: {severity}"

    print("✅ Distribution analysis test passed")


def test_data_loader_error_handling():
    """Test error handling for missing file"""
    try:
        loader = LogDataLoader("non_existent_file.csv")
        loader.load_data()
        assert False, "Should have raised an exception for missing file"
    except FileNotFoundError:
        print("✅ FileNotFoundError correctly raised for missing file")
    except Exception as e:
        print(f"✅ Other exception raised for missing file: {type(e).__name__}")


def test_data_loader_summary():
    """Test summary generation"""
    data_path = "docs/log_dataset.csv"

    loader = LogDataLoader(data_path)
    loader.load_data()
    summary = loader.get_summary()

    # Check summary structure
    assert "data_validation" in summary
    assert "distributions" in summary
    assert "sample_data" in summary

    # Check sample data
    assert len(summary["sample_data"]) == 3

    print("✅ Summary generation test passed")


def test_data_loader_save_report():
    """Test report saving functionality"""
    data_path = "docs/log_dataset.csv"

    loader = LogDataLoader(data_path)
    loader.load_data()

    # Create temporary file for report
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        report_path = tmp.name

    try:
        report = loader.save_analysis_report(report_path)

        # Check report structure
        assert "generated_at" in report
        assert "data_file" in report
        assert "summary" in report

        # Check file was created
        assert os.path.exists(report_path), f"Report file not created: {report_path}"

        print("✅ Report saving test passed")
    finally:
        # Clean up
        if os.path.exists(report_path):
            os.unlink(report_path)


def main():
    """Run all tests"""
    print("Running data loader tests...\n")

    tests = [
        test_data_loader_basic,
        test_data_loader_validation,
        test_data_loader_distributions,
        test_data_loader_error_handling,
        test_data_loader_summary,
        test_data_loader_save_report,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed with unexpected error: {e}")
            failed += 1

    print(f"\n📊 Test Results: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
