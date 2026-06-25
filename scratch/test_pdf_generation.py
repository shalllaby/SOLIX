import sys
import os
import io

# Setup paths
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)

from backend.utils.cleaning_studio_pdf_generator import CleaningStudioPDFReportGenerator

# Mock task data
mock_task_data = {
    "status": "completed",
    "progress": 100,
    "goal": "Clean the missing values, format email addresses, and detect salary outliers.",
    "warnings": ["Kaggle execution fell back to local engine due to token validation warnings."],
    "result": {
        "dataset_id": "test-dataset-id",
        "cleaned_dataset_id": "test-cleaned-id",
        "strategy_used": "gamma",
        "audit_id": "test-audit-id",
        "filename": "customer_data_2026.csv",
        "stats": {
            "rows_before": 15230,
            "rows_after": 14980,
            "missing_before": 2340,
            "missing_after": 12,
            "cells_fixed": 2328
        },
        "report": {
            "truth_confidence_score": 98.4,
            "blocked_actions": ["Column 'SSN': modification blocked due to privacy policy"],
            "warnings": ["Detected high null value percentage in column 'Age'"],
            "actions": [
                "Imputed missing numerical values in column 'Age' using median",
                "Converted column 'Email' values to standardized lowercase",
                "Removed 250 rows containing duplicate primary keys"
            ]
        },
        "audit_log": {
            "truth_confidence_score": 98.4,
            "actions_log": [
                {"column": "Age", "issue": "Missing values", "resolution": "AI Imputation (Median)"},
                {"column": "Email", "issue": "Non-standard uppercase letters", "resolution": "Standardized to lowercase"},
                {"column": "Salary", "issue": "Outlier values detected (Z-Score > 3)", "resolution": "Capped outlier values to 99th percentile"},
                {"column": "DateOfBirth", "issue": "Inconsistent date formats", "resolution": "Parsed and formatted to ISO-8601 YYYY-MM-DD"}
            ]
        }
    }
}

def test_pdf_generation():
    print("Testing English PDF report generation...")
    try:
        pdf_en = CleaningStudioPDFReportGenerator.generate_report("test-task-123", mock_task_data, is_arabic=False)
        pdf_bytes_en = pdf_en.getvalue()
        print(f"Success! English PDF generated size: {len(pdf_bytes_en)} bytes")
        
        # Save locally to inspect
        with open("scratch/test_report_en.pdf", "wb") as f:
            f.write(pdf_bytes_en)
        print("Saved to scratch/test_report_en.pdf")
    except Exception as e:
        print(f"Error generating English PDF: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting Arabic PDF report generation...")
    try:
        pdf_ar = CleaningStudioPDFReportGenerator.generate_report("test-task-123", mock_task_data, is_arabic=True)
        pdf_bytes_ar = pdf_ar.getvalue()
        print(f"Success! Arabic PDF generated size: {len(pdf_bytes_ar)} bytes")
        
        # Save locally to inspect
        with open("scratch/test_report_ar.pdf", "wb") as f:
            f.write(pdf_bytes_ar)
        print("Saved to scratch/test_report_ar.pdf")
    except Exception as e:
        print(f"Error generating Arabic PDF: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pdf_generation()
