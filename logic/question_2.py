# logic/question_2.py

import pandas as pd
from datetime import datetime, timedelta
import re

def run_logic(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Analyzes cost changes month-over-month for a specific segment.
    Identifies 'Group Description' where cost increased from previous month to month of interest.
    """
    segment = filters.get("segment")
    period1_start = filters.get("period1_start")
    period1_end = filters.get("period1_end")
    period2_start = filters.get("period2_start")
    period2_end = filters.get("period2_end")

    if not period1_start or not period1_end or not period2_start or not period2_end:
        return pd.DataFrame({"Message": ["Please specify valid comparison periods for analysis (e.g., 'April 2025 vs May 2025')."]})

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
            return pd.DataFrame({"Message": [f"No data found for segment: '{segment}'. Please check the segment name in your data or try a broader term."]})
    else:
        df_filtered_segment = df
    # --- End Dynamic Segment Filtering Enhancement ---

    # Filter by Type = "Cost"
    df_costs = df_filtered_segment[df_filtered_segment["Type"] == "Cost"].copy()
    if df_costs.empty:
        return pd.DataFrame({"Message": [f"No cost data found for segment '{segment}'." if segment else "No cost data found."]})

    df_costs["Month"] = pd.to_datetime(df_costs["Month"], errors='coerce', dayfirst=True)
    df_costs = df_costs.dropna(subset=['Month'])

    df_period2 = df_costs[
        (df_costs["Month"].dt.date >= period2_start) &
        (df_costs["Month"].dt.date <= period2_end)
    ]

    df_period1 = df_costs[
        (df_costs["Month"].dt.date >= period1_start) &
        (df_costs["Month"].dt.date <= period1_end)
    ]

    if df_period2.empty and df_period1.empty:
        return pd.DataFrame({"Message": [f"No cost data available for {period2_start.strftime('%b %Y')} or {period1_start.strftime('%b %Y')} for segment '{segment}'."]})

    period2_costs = df_period2.groupby("Group Description")["Amount in USD"].sum().reset_index()
    period1_costs = df_period1.groupby("Group Description")["Amount in USD"].sum().reset_index()

    # Get month names for dynamic column headers
    current_month_name = period2_start.strftime('%b %Y')
    previous_month_name = period1_start.strftime('%b %Y')

    current_month_col_name = current_month_name + ' Cost'
    previous_month_col_name = previous_month_name + ' Cost'
    # New: Dynamic Cost Change column name
    cost_change_col_name = f'Cost Change ({current_month_name} vs {previous_month_name})'


    period2_costs.rename(columns={"Amount in USD": current_month_col_name}, inplace=True)
    period1_costs.rename(columns={"Amount in USD": previous_month_col_name}, inplace=True)


    merged_costs = pd.merge(
        period2_costs,
        period1_costs,
        on="Group Description",
        how="outer"
    ).fillna(0)

    # Calculate the change using the dynamically named columns
    merged_costs[cost_change_col_name] = merged_costs[current_month_col_name] - merged_costs[previous_month_col_name]

    # Filter for costs that increased
    triggered_costs = merged_costs[merged_costs[cost_change_col_name] > 0].sort_values(by=cost_change_col_name, ascending=False)

    if triggered_costs.empty:
        return pd.DataFrame({"Message": [f"No cost increases found in '{segment}' for {current_month_name} compared to {previous_month_name}."]})

    # Format for display
    triggered_costs[current_month_col_name] = triggered_costs[current_month_col_name].apply(lambda x: f"${x:,.2f}")
    triggered_costs[previous_month_col_name] = triggered_costs[previous_month_col_name].apply(lambda x: f"${x:,.2f}")
    # New: Format the dynamically named Cost Change column
    triggered_costs[cost_change_col_name] = triggered_costs[cost_change_col_name].apply(lambda x: f"${x:,.2f}")

    # Add S.No
    triggered_costs.reset_index(drop=True, inplace=True)
    triggered_costs.index = triggered_costs.index + 1
    triggered_costs = triggered_costs.rename_axis("S.No").reset_index()

    # Return with all dynamic column names
    return triggered_costs[["S.No", "Group Description", current_month_col_name, previous_month_col_name, cost_change_col_name]]