import pandas as pd
from datetime import datetime, timedelta
import pytz
from logic.utils import calculate_cm  # Ensure calculate_cm is imported


def run_logic(df: pd.DataFrame, cm_filters: dict) -> pd.DataFrame:
    """
    Processes queries related to Contribution Margin (CM) analysis for customers.
    Applies date and CM filters, then calculates and returns CM details.
    """
    # Ensure 'Month' column is datetime
    # Removed infer_datetime_format as it's deprecated and a strict version is now default
    df["Month"] = pd.to_datetime(df["Month"], errors='coerce', dayfirst=True)
    df.dropna(subset=['Month'], inplace=True)  # Drop rows where Month could not be parsed

    # Apply date filter if present
    if cm_filters.get("date_filter") and cm_filters.get("start_date") and cm_filters.get("end_date"):
        start_date = cm_filters["start_date"]
        end_date = cm_filters["end_date"]
        df = df[
            (df["Month"].dt.date >= start_date) &
            (df["Month"].dt.date <= end_date)
            ].copy()  # Use .copy() to avoid SettingWithCopyWarning

        if df.empty:
            return pd.DataFrame({"Message": ["No data found for the specified date range."]})

    # Calculate CM and apply CM filters
    result_df = calculate_cm(df, cm_filters)

    return result_df

