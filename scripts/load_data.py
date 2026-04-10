#!/usr/bin/env python3
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import LogDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    # Path to dataset
    data_path = "docs/log_dataset.csv"

    # Create loader instance
    loader = LogDataLoader(data_path)

    try:
        # Load data
        df = loader.load_data()
        print(f"\n✅ Successfully loaded {len(df)} rows")
        print(f"📊 Columns: {list(df.columns)}")

        # Validate data
        print("\n🔍 Data Validation Results:")
        validation = loader.validate_data()
        print(f"   Total rows: {validation['total_rows']}")
        print(f"   Total columns: {validation['total_columns']}")

        if validation.get("missing_values"):
            print(f"   ⚠️ Missing values found:")
            for col, count in validation["missing_values"].items():
                print(f"     - {col}: {count}")
        else:
            print("   ✅ No missing values")

        if validation["duplicates"]["total_duplicate_rows"] > 0:
            print(
                f"   ⚠️ Duplicate rows: {validation['duplicates']['total_duplicate_rows']}"
            )
        else:
            print("   ✅ No duplicate rows")

        if validation["duplicates"]["duplicate_log_ids"] > 0:
            print(
                f"   ⚠️ Duplicate log IDs: {validation['duplicates']['duplicate_log_ids']}"
            )
        else:
            print("   ✅ No duplicate log IDs")

        # Analyze distributions
        print("\n📈 Distribution Analysis:")
        distributions = loader.analyze_distributions()

        print(f"\n   Root Cause Distribution (RC-01 to RC-08):")
        for rc, count in sorted(distributions["root_cause_distribution"].items()):
            print(f"     - {rc}: {count} samples")

        print(f"\n   Service Distribution:")
        for service, count in distributions["service_distribution"].items():
            print(f"     - {service}: {count} samples")

        print(f"\n   Severity Distribution:")
        for severity, count in distributions["severity_distribution"].items():
            print(f"     - {severity}: {count} samples")

        # Class balance analysis
        balance = distributions["class_balance"]
        print(f"\n   Class Balance Analysis:")
        print(f"     Total samples: {balance['total_samples']}")
        print(f"     Unique classes: {balance['unique_classes']}")
        print(f"     Min samples per class: {balance['min_samples']}")
        print(f"     Max samples per class: {balance['max_samples']}")

        print(f"\n   Class Percentages:")
        for rc, pct in balance["class_percentages"].items():
            print(f"     - {rc}: {pct}%")

        # Save analysis report
        print("\n💾 Saving analysis report...")
        report = loader.save_analysis_report("reports/data_analysis.json")
        print(f"   Report saved to reports/data_analysis.json")

        # Show sample data
        print("\n📋 Sample Data (first 3 rows):")
        sample = loader.df.head(3)
        for idx, row in sample.iterrows():
            print(f"\n   Row {idx + 1}:")
            print(f"     log_id: {row['log_id']}")
            print(f"     service: {row['service']}")
            print(f"     severity: {row['severity']}")
            print(f"     root_cause: {row['root_cause_label']}")
            print(f"     log_message: {row['log_message'][:80]}...")

        return True

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
