import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LogDataLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}

    def load_data(self) -> pd.DataFrame:
        try:
            logger.info(f"Loading data from {self.data_path}")

            # Use pandas with proper quoting for CSV with quoted fields containing commas
            self.df = pd.read_csv(
                self.data_path,
                quotechar='"',
                escapechar="\\",
                doublequote=True,
                skipinitialspace=True,
            )

            if self.df is not None:
                logger.info(f"Successfully loaded {len(self.df)} rows")
                logger.info(f"Columns: {list(self.df.columns)}")
            else:
                logger.error("Failed to load data - DataFrame is None")
                raise ValueError("Failed to load data - DataFrame is None")

            return self.df

        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def validate_data(self) -> Dict[str, Any]:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        validation_results: Dict[str, Any] = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "missing_values": {},
            "duplicates": {},
            "data_types": {},
        }

        # Check for missing values
        missing_counts = self.df.isnull().sum()
        validation_results["missing_values"] = missing_counts[
            missing_counts > 0
        ].to_dict()

        # Check for duplicates
        duplicate_rows = self.df.duplicated().sum()
        validation_results["duplicates"]["total_duplicate_rows"] = int(duplicate_rows)

        # Check for duplicate log_ids
        duplicate_ids = self.df["log_id"].duplicated().sum()
        validation_results["duplicates"]["duplicate_log_ids"] = int(duplicate_ids)

        # Get data types
        validation_results["data_types"] = self.df.dtypes.astype(str).to_dict()

        # Validate expected columns
        expected_columns = [
            "log_id",
            "timestamp",
            "service",
            "severity",
            "log_message",
            "root_cause_label",
        ]
        missing_columns = [
            col for col in expected_columns if col not in self.df.columns
        ]

        if missing_columns:
            validation_results["missing_columns"] = missing_columns
            logger.warning(f"Missing expected columns: {missing_columns}")

        return validation_results

    def analyze_distributions(self) -> Dict[str, Any]:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        analysis: Dict[str, Any] = {
            "root_cause_distribution": {},
            "service_distribution": {},
            "severity_distribution": {},
            "class_balance": {},
        }

        # Root cause distribution
        rc_counts = self.df["root_cause_label"].value_counts()
        analysis["root_cause_distribution"] = rc_counts.to_dict()

        # Service distribution
        service_counts = self.df["service"].value_counts()
        analysis["service_distribution"] = service_counts.to_dict()

        # Severity distribution
        severity_counts = self.df["severity"].value_counts()
        analysis["severity_distribution"] = severity_counts.to_dict()

        # Class balance analysis
        total_samples = len(self.df)
        rc_percentages = (rc_counts / total_samples * 100).round(2)
        analysis["class_balance"] = {
            "total_samples": total_samples,
            "unique_classes": len(rc_counts),
            "min_samples": int(rc_counts.min()),
            "max_samples": int(rc_counts.max()),
            "class_percentages": rc_percentages.to_dict(),
        }

        return analysis

    def get_summary(self) -> Dict[str, Any]:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        validation = self.validate_data()
        distributions = self.analyze_distributions()

        summary = {
            "data_validation": validation,
            "distributions": distributions,
            "sample_data": self.df.head(3).to_dict(orient="records"),
        }

        return summary

    def save_analysis_report(self, output_path: str):
        import json
        from datetime import datetime

        summary = self.get_summary()

        report = {
            "generated_at": datetime.now().isoformat(),
            "data_file": str(self.data_path),
            "summary": summary,
        }

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Analysis report saved to {output_path_obj}")

        return report
