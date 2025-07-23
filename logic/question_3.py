import pandas as pd
from datetime import datetime, timedelta

# Define the relevant Group Descriptions for C&B costs globally for clarity
CB_GROUPS = ["C&B Cost Offshore", "C&B Cost Onsite"]


def run_logic(df: pd.DataFrame, parsed_filters: dict) -> pd.DataFrame:
    """
    Calculates the variation in C&B cost between two dynamically specified periods (months or quarters).

    Args:
        df (pd.DataFrame): The main DataFrame containing financial data.
        parsed_filters (dict): Dictionary containing parsed period details,
                                including comparison_type, period1_name, period1_start, period1_end,
                                period2_name, period2_start, period2_end, and comparison_valid.

    Returns:
        pd.DataFrame: A DataFrame summarizing the C&B costs for each period and their variation.
                      Returns a DataFrame with a "Message" column if parsing failed or no data.
    """
    comparison_type = parsed_filters.get("comparison_type")
    p1_name = parsed_filters.get("period1_name")
    p1_start = parsed_filters.get("period1_start")
    p1_end = parsed_filters.get("period1_end")

    p2_name = parsed_filters.get("period2_name")
    p2_start = parsed_filters.get("period2_start")
    p2_end = parsed_filters.get("period2_end")

    if not parsed_filters.get("comparison_valid"):
        return pd.DataFrame({"Message": [
            "Could not identify valid periods for comparison. Please specify two periods (e.g., 'FY26 Q1' and 'FY25 Q4', or 'April 2025' and 'May 2025')."]})

    # Filter the DataFrame for C&B related rows
    cb_df = df[df["Group Description"].isin(CB_GROUPS)].copy()

    if cb_df.empty:
        return pd.DataFrame({"Message": ["No C&B cost data found in the dataset for the relevant groups."]})

    # Ensure 'Month' column is in datetime format
    cb_df["Month"] = pd.to_datetime(cb_df["Month"], errors='coerce', dayfirst=True)
    cb_df.dropna(subset=['Month'], inplace=True)

    if cb_df.empty:
        return pd.DataFrame({"Message": ["No valid date data for C&B costs after parsing."]})

    # Filter data for Period 1
    p1_df = cb_df[
        (cb_df["Month"].dt.date >= p1_start) &
        (cb_df["Month"].dt.date <= p1_end)
        ]

    # Filter data for Period 2
    p2_df = cb_df[
        (cb_df["Month"].dt.date >= p2_start) &
        (cb_df["Month"].dt.date <= p2_end)
        ]

    # Calculate total C&B cost for each period
    cb_p1_value = p1_df["Amount in USD"].sum()  # Store numeric value
    cb_p2_value = p2_df["Amount in USD"].sum()  # Store numeric value

    # Determine if data exists for each period
    p1_data_exists = not p1_df.empty
    p2_data_exists = not p2_df.empty

    # Calculate the variation (Period 2 - Period 1)
    variation = None
    percentage_variation = None
    variation_display = ""

    if p1_data_exists and p2_data_exists:
        variation = cb_p2_value - cb_p1_value
        variation_display = f"${variation:,.2f}"
        if cb_p1_value != 0:
            percentage_variation = (variation / cb_p1_value) * 100
    elif not p1_data_exists and not p2_data_exists:
        variation_display = "N/A (No data for either period)"
    elif not p1_data_exists:
        variation_display = f"N/A (No data for {p1_name})"
    else:  # not p2_data_exists
        variation_display = f"N/A (No data for {p2_name})"

    # Determine period label for display
    period_label = "Quarter" if comparison_type == "quarter" else "Month"

    # Generate formatted period names for the table
    formatted_p1_name = f"{p1_name} ({p1_start.strftime('%b %Y')}-{p1_end.strftime('%b %Y')})" if comparison_type == "quarter" else f"{p1_name} ({p1_start.strftime('%b %Y')})"
    formatted_p2_name = f"{p2_name} ({p2_start.strftime('%b %Y')}-{p2_end.strftime('%b %Y')})" if comparison_type == "quarter" else f"{p2_name} ({p2_start.strftime('%b %Y')})"

    # Prepare results for display
    results = {
        period_label: [formatted_p1_name,
                       formatted_p2_name,
                       f"Variation ({p2_name} vs {p1_name})"],
        "C&B Cost (USD)": [cb_p1_value, cb_p2_value, variation_display],
        # Add numeric values and percentage for app.py to use in visuals/metrics
        "Period1_Cost_Numeric": cb_p1_value,
        "Period2_Cost_Numeric": cb_p2_value,
        "Percentage_Variation": percentage_variation,
        "Comparison_Type": comparison_type  # Pass this back for app.py display logic
    }
    result_df = pd.DataFrame(results)

    # Format currency for display for the period costs
    result_df.loc[0, "C&B Cost (USD)"] = f"${cb_p1_value:,.2f}" if p1_data_exists else f"N/A (No data for {p1_name})"
    result_df.loc[1, "C&B Cost (USD)"] = f"${cb_p2_value:,.2f}" if p2_data_exists else f"N/A (No data for {p2_name})"

    return result_df

