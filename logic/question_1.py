import pandas as pd
from logic.utils import calculate_cm


def run_logic(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Logic to retrieve customers with CM% based on specified thresholds and date filters.
    This function prepares the data for display and visualization in the main app.
    """
    filtered_df_by_date = df.copy()

    # Apply date filtering if specified
    if filters.get("date_filter") and filters.get("start_date") and filters.get("end_date"):
        # Ensure 'Month' column is datetime
        # This conversion should ideally happen once during data loading in app.py,
        # but kept here for robustness in case df is passed without it.
        filtered_df_by_date["Month"] = pd.to_datetime(filtered_df_by_date["Month"], errors='coerce', dayfirst=True,
                                                      infer_datetime_format=True)

        filtered_df_by_date = filtered_df_by_date.dropna(subset=['Month'])  # Drop rows where month parsing failed

        filtered_df_by_date = filtered_df_by_date[
            (filtered_df_by_date["Month"].dt.date >= filters["start_date"]) &
            (filtered_df_by_date["Month"].dt.date <= filters["end_date"])
            ]
        if filtered_df_by_date.empty:
            # Return a DataFrame with a message if no data for the date range
            return pd.DataFrame({"Message": [
                f"No data available for the date range: {filters['start_date'].strftime('%Y-%m-%d')} to {filters['end_date'].strftime('%Y-%m-%d')}"]})

    # Calculate CM and apply CM filters using the updated calculate_cm from utils
    result_df = calculate_cm(filtered_df_by_date, filters)

    if result_df.empty:
        # Return a DataFrame with a message if no customers match CM criteria
        return pd.DataFrame({"Message": ["No customers found matching the specified CM criteria."]})

    return result_df
