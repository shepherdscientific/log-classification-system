#!/usr/bin/env python3
"""
Root Cause Category Analysis Demo Script.

Demonstrates the RootCauseCategoryAnalyzer functionality for analyzing
patterns in log messages for each RC category.

Usage:
    python scripts/run_category_analysis.py
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import json
from src.data.loader import LogDataLoader
from src.evaluation.category_analysis import RootCauseCategoryAnalyzer


def main():
    """Run the category analysis demonstration."""
    print("=" * 80)
    print("Root Cause Category Analysis Demo")
    print("=" * 80)

    # Step 1: Load the dataset
    print("\n1. Loading dataset...")
    try:
        dataset_path = (
            project_root
            / "docs"
            / "Flutterwave AI Engineer Assessment Dataset.xlsx - log_dataset.csv"
        )
        data_loader = LogDataLoader(str(dataset_path))
        data = data_loader.load_data()
        print(f"   ✓ Loaded {len(data)} log entries")
        print(f"   ✓ Dataset columns: {list(data.columns)}")
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        return

    # Step 2: Initialize the category analyzer
    print("\n2. Initializing RootCauseCategoryAnalyzer...")
    analyzer = RootCauseCategoryAnalyzer()
    print(
        f"   ✓ Analyzer initialized with {len(analyzer.root_causes)} root cause categories"
    )

    # Step 3: Get category descriptions
    print("\n3. Category Descriptions:")
    descriptions = analyzer.get_category_descriptions()
    for rc, desc in descriptions.items():
        print(f"   {rc}: {desc}")

    # Step 4: Analyze the dataset
    print("\n4. Analyzing dataset patterns...")
    category_patterns = analyzer.analyze_dataset(data)
    print(f"   ✓ Analyzed patterns for {len(category_patterns)} categories")

    # Step 5: Show sample analysis for each category
    print("\n5. Sample Analysis for Each Category:")
    for rc, pattern in category_patterns.items():
        print(f"\n   {rc}:")
        print(
            f"     • Key Phrases: {', '.join(pattern.key_phrases[:5])}{'...' if len(pattern.key_phrases) > 5 else ''}"
        )
        print(f"     • Error Types: {', '.join(pattern.error_types)}")
        print(f"     • Services: {len(pattern.service_distribution)} services")
        print(
            f"     • Most Common Service: {max(pattern.service_distribution.items(), key=lambda x: x[1])[0] if pattern.service_distribution else 'N/A'}"
        )

    # Step 6: Generate comprehensive report
    print("\n6. Generating comprehensive category analysis report...")
    output_dir = project_root / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "category_analysis_report.json"

    report = analyzer.generate_category_report(data, str(report_path))
    print(f"   ✓ Report saved to: {report_path}")

    # Step 7: Show cross-category analysis
    print("\n7. Cross-Category Analysis:")
    cross_analysis = report["cross_category_analysis"]
    print(
        f"   • Total Unique Key Phrases: {cross_analysis['total_unique_key_phrases']}"
    )

    # Show distinctive phrases for each category
    print("   • Distinctive Phrases per Category:")
    distinctive = cross_analysis["distinctive_phrases_per_category"]
    for rc, phrases in distinctive.items():
        if phrases:
            print(
                f"     {rc}: {', '.join(phrases[:3])}{'...' if len(phrases) > 3 else ''}"
            )

    # Show similarity matrix (simplified)
    print("   • Category Similarity (simplified):")
    similarity_matrix = cross_analysis["category_similarity_matrix"]
    for rc1 in ["RC-01", "RC-02", "RC-03"]:  # Show first 3 for brevity
        similarities = []
        for rc2 in ["RC-01", "RC-02", "RC-03", "RC-04", "RC-05"]:  # Compare to first 5
            if rc1 != rc2:
                sim = similarity_matrix[rc1][rc2]
                if sim > 0.3:  # Only show significant similarities
                    similarities.append(f"{rc2}:{sim:.2f}")
        if similarities:
            print(f"     {rc1} similar to: {', '.join(similarities)}")

    # Step 8: Demonstrate log message mapping
    print("\n8. Log Message to Category Mapping Examples:")
    test_logs = [
        "ERROR [api-gateway] 401 returned to client: bearer token missing",
        "ERROR [db-pool] Max wait time exceeded waiting for idle connection",
        "WARN [api-gateway] Rate limit hit on endpoint /v2/events",
        "ERROR [transaction-validator] Constraint violation: currency_code 'null' not in allowed list",
        "CRITICAL [audit-service] Privilege escalation attempt blocked",
    ]

    for i, log_msg in enumerate(test_logs[:3], 1):  # Show first 3 for brevity
        print(f"\n   Example {i}: {log_msg[:60]}...")
        scores = analyzer.map_log_to_category_patterns(log_msg)
        top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for rc, score in top_3:
            if score > 0:
                print(f"     • {rc}: {score:.3f}")

    # Step 9: Show typical system issues for each category
    print("\n9. Typical System Issues by Category:")
    for rc in ["RC-01", "RC-02", "RC-07"]:  # Show 3 categories for brevity
        pattern_def = analyzer.pattern_definitions.get(rc, {})
        if pattern_def:
            print(f"\n   {rc} - {pattern_def['name']}:")
            for i, issue in enumerate(
                pattern_def["typical_issues"][:2], 1
            ):  # Show first 2
                print(f"     {i}. {issue}")

    print("\n" + "=" * 80)
    print("Category Analysis Complete!")
    print("=" * 80)
    print(f"\nSummary:")
    print(
        f"• Analyzed {len(data)} log entries across {len(category_patterns)} root cause categories"
    )
    print(f"• Generated comprehensive report at: {report_path}")
    print(
        f"• Identified {cross_analysis['total_unique_key_phrases']} unique key phrases"
    )
    print(f"• Created detailed category descriptions and pattern analysis")

    # Save a summary file
    summary_path = output_dir / "category_analysis_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Root Cause Category Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total log entries analyzed: {len(data)}\n")
        f.write(f"Root cause categories: {len(category_patterns)}\n")
        f.write(
            f"Unique key phrases identified: {cross_analysis['total_unique_key_phrases']}\n\n"
        )

        f.write("Category Overview:\n")
        for rc, pattern in category_patterns.items():
            pattern_def = analyzer.pattern_definitions.get(rc, {})
            f.write(f"\n{rc} - {pattern_def.get('name', rc)}:\n")
            f.write(f"  • Key phrases: {', '.join(pattern.key_phrases[:5])}\n")
            f.write(f"  • Services affected: {len(pattern.service_distribution)}\n")
            f.write(f"  • Log count: {sum(pattern.service_distribution.values())}\n")

    print(f"• Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
