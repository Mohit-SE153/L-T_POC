# In logic/question_2.py

import pandas as pd
from datetime import datetime, timedelta
import re  # Import regex module


def run_logic(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Analyzes cost changes month-over-month for a specific segment.
    Identifies 'Group Description' where cost increased from previous month to month of interest.
    """
    segment = filters.get("segment")

    month_of_interest_name = filters.get("month_of_interest_name")
    month_of_interest_start = filters.get("month_of_interest_start")
    month_of_interest_end = filters.get("month_of_interest_end")

    compare_to_month_name = filters.get("compare_to_month_name")
    compare_to_month_start = filters.get("compare_to_month_start")
    compare_to_month_end = filters.get("compare_to_month_end")

    # Check for validity of parsed date periods
    if not (month_of_interest_start and month_of_interest_end and
            compare_to_month_start and compare_to_month_end):
        return pd.DataFrame({"Message": [
            "Could not identify valid months for comparison. Please specify a month or a valid range (e.g., 'last month', 'March 2025 to April 2025')."]})

    # --- Debugging print statement ---
    print(f"DEBUG: question_2 run_logic received segment: '{segment}'")
    print(f"DEBUG: Month of Interest: {month_of_interest_name} ({month_of_interest_start} to {month_of_interest_end})")
    print(f"DEBUG: Compare To Month: {compare_to_month_name} ({compare_to_month_start} to {compare_to_month_end})")
    # --- End Debugging print statement ---

    # (Keep the rest of your existing segment filtering logic)
    # --- Start Dynamic Segment Filtering Enhancement ---
    if segment:
        cleaned_query_segment = segment.lower()
        cleaned_query_segment = re.sub(r'\s+', ' ', cleaned_query_segment).strip()
        cleaned_query_segment = cleaned_query_segment.replace(' and ', ' & ')
        cleaned_query_segment = cleaned_query_segment.replace('.', '')
        cleaned_query_segment = cleaned_query_segment.replace('enggr', 'engineering')

        df['Cleaned_Segment'] = df["Segment"].astype(str).str.lower()
        df['Cleaned_Segment'] = df['Cleaned_Segment'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        df['Cleaned_Segment'] = df['Cleaned_Segment'].str.replace(' and ', ' & ')
        df['Cleaned_Segment'] = df['Cleaned_Segment'].str.replace('.', '')
        df['Cleaned_Segment'] = df['Cleaned_Segment'].str.replace('enggr', 'engineering')

        df_filtered_segment = df[df['Cleaned_Segment'].str.contains(cleaned_query_segment, case=False, na=False)]

        if df_filtered_segment.empty:
            return pd.DataFrame({"Message": [
                f"No data found for segment: '{segment}'. Please check the segment name in your data or try a broader term."]})
    else:
        df_filtered_segment = df
    # --- End Dynamic Segment Filtering Enhancement ---

    # Filter by Type = "Cost"
    df_costs = df_filtered_segment[df_filtered_segment["Type"] == "Cost"]
    if df_costs.empty:
        return pd.DataFrame(
            {"Message": [f"No cost data found for segment '{segment}'." if segment else "No cost data found."]})

    # Ensure 'Month' column is datetime objects for filtering
    df_costs["Month"] = pd.to_datetime(df_costs["Month"], errors='coerce', dayfirst=True)
    df_costs = df_costs.dropna(subset=['Month'])

    # Filter data for the month of interest
    df_current_month = df_costs[
        (df_costs["Month"].dt.date >= month_of_interest_start) &
        (df_costs["Month"].dt.date <= month_of_interest_end)
        ]

    # Filter data for the previous (comparison) month
    df_previous_month = df_costs[
        (df_costs["Month"].dt.date >= compare_to_month_start) &
        (df_costs["Month"].dt.date <= compare_to_month_end)
        ]

    if df_current_month.empty and df_previous_month.empty:
        return pd.DataFrame({"Message": [
            f"No cost data available for {month_of_interest_name} or {compare_to_month_name} for segment '{segment}'."]})

    # Group by "Group Description" and sum "Amount in USD" for each month
    current_month_costs = df_current_month.groupby("Group Description")["Amount in USD"].sum().reset_index()
    # Dynamic column name
    current_month_costs.rename(columns={"Amount in USD": f"Cost in {month_of_interest_name}"}, inplace=True)

    previous_month_costs = df_previous_month.groupby("Group Description")["Amount in USD"].sum().reset_index()
    # Dynamic column name
    previous_month_costs.rename(columns={"Amount in USD": f"Cost in {compare_to_month_name}"}, inplace=True)

    # Merge the two dataframes to compare costs
    merged_costs = pd.merge(
        current_month_costs,
        previous_month_costs,
        on="Group Description",
        how="outer"
    ).fillna(0)

    # Calculate the change
    merged_costs["Cost Change"] = merged_costs[f"Cost in {month_of_interest_name}"] - merged_costs[
        f"Cost in {compare_to_month_name}"]

    # Filter for costs that increased (triggered margin drop)
    triggered_costs = merged_costs[merged_costs["Cost Change"] > 0].sort_values(by="Cost Change", ascending=False)

    if triggered_costs.empty:
        return pd.DataFrame({"Message": [
            f"No cost increases found in '{segment}' for {month_of_interest_name} compared to {compare_to_month_name}."]})

    # Format for display
    triggered_costs[f"Cost in {month_of_interest_name}"] = triggered_costs[f"Cost in {month_of_interest_name}"].apply(
        lambda x: f"${x:,.2f}")
    triggered_costs[f"Cost in {compare_to_month_name}"] = triggered_costs[f"Cost in {compare_to_month_name}"].apply(
        lambda x: f"${x:,.2f}")

    # Dynamic column name for Cost Change
    cost_change_col_name = f"Cost Change ({month_of_interest_name} vs {compare_to_month_name})"
    triggered_costs[cost_change_col_name] = triggered_costs["Cost Change"].apply(lambda x: f"${x:,.2f}")

    # Drop the original 'Cost Change' column before final selection
    triggered_costs.drop(columns=["Cost Change"], inplace=True)

    # Add S.No
    triggered_costs.reset_index(drop=True, inplace=True)
    triggered_costs.index = triggered_costs.index + 1
    triggered_costs = triggered_costs.rename_axis("S.No").reset_index()

    # Return with dynamic column names
    return triggered_costs[
        ["S.No", "Group Description", f"Cost in {month_of_interest_name}", f"Cost in {compare_to_month_name}",
         cost_change_col_name]]