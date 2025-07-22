import pandas as pd
from datetime import datetime, timedelta
import re # Import regex module

def run_logic(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Analyzes cost changes month-over-month for a specific segment.
    Identifies 'Group Description' where cost increased from previous month to month of interest.
    """
    segment = filters.get("segment")
    month_of_interest_start = filters.get("month_of_interest_start")
    month_of_interest_end = filters.get("month_of_interest_end")
    compare_to_previous_month = filters.get("compare_to_previous_month", True)

    # --- Debugging print statement ---
    print(f"DEBUG: question_2 run_logic received segment: '{segment}'")
    print(f"DEBUG: month_of_interest_start: {month_of_interest_start}, month_of_interest_end: {month_of_interest_end}")
    # --- End Debugging print statement ---

    if not month_of_interest_start or not month_of_interest_end:
        return pd.DataFrame({"Message": ["Please specify a valid month for analysis (e.g., 'last month', 'July 2024')."]})

    # Calculate the previous month's dates
    prev_month_end = month_of_interest_start - timedelta(days=1)
    prev_month_start = prev_month_end.replace(day=1)

    # --- Start Dynamic Segment Filtering Enhancement ---
    if segment:
        # Step 1: Normalize the query segment for better matching
        # Convert to lowercase, remove common punctuation/extra spaces, handle 'and' vs '&'
        cleaned_query_segment = segment.lower()
        cleaned_query_segment = re.sub(r'\s+', ' ', cleaned_query_segment).strip() # Replace multiple spaces with single
        cleaned_query_segment = cleaned_query_segment.replace(' and ', ' & ') # Standardize 'and' to '&'
        cleaned_query_segment = cleaned_query_segment.replace('.', '') # Remove periods (e.g., "Enggr." vs "Enggr")
        cleaned_query_segment = cleaned_query_segment.replace('enggr', 'engineering') # Specific abbreviation handling

        # Step 2: Create a cleaned version of the 'Segment' column in the DataFrame
        # Apply the same cleaning logic to the DataFrame's 'Segment' column
        df['Cleaned_Segment'] = df["Segment"].astype(str).str.lower()
        df['Cleaned_Segment'] = df['Cleaned_Segment'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        df['Cleaned_Segment'] = df['Cleaned_Segment'].str.replace(' and ', ' & ')
        df['Cleaned_Segment'] = df['Cleaned_Segment'].str.replace('.', '')
        df['Cleaned_Segment'] = df['Cleaned_Segment'].str.replace('enggr', 'engineering') # Specific abbreviation handling

        # Step 3: Filter data using the cleaned segments
        # Use str.contains for flexible matching, as exact match might still be too strict
        # Ensure 'Segment' column is string type before using .str accessor
        df_filtered_segment = df[df['Cleaned_Segment'].str.contains(cleaned_query_segment, case=False, na=False)]

        # --- Debugging print statements for segment matching ---
        print(f"DEBUG: Cleaned Query Segment: '{cleaned_query_segment}'")
        print(f"DEBUG: Unique Cleaned Segments in DF: {df['Cleaned_Segment'].unique()}")
        print(f"DEBUG: Rows after segment filter: {len(df_filtered_segment)}")
        # --- End Debugging print statements ---

        if df_filtered_segment.empty:
            return pd.DataFrame({"Message": [f"No data found for segment: '{segment}'. Please check the segment name in your data or try a broader term."]})
    else:
        df_filtered_segment = df # If no segment specified, consider all data
    # --- End Dynamic Segment Filtering Enhancement ---

    # Filter by Type = "Cost"
    df_costs = df_filtered_segment[df_filtered_segment["Type"] == "Cost"]
    if df_costs.empty:
        return pd.DataFrame({"Message": [f"No cost data found for segment '{segment}'." if segment else "No cost data found."]})

    # Ensure 'Month' column is datetime objects for filtering
    df_costs["Month"] = pd.to_datetime(df_costs["Month"], errors='coerce', dayfirst=True, infer_datetime_format=True)
    df_costs = df_costs.dropna(subset=['Month']) # Drop rows where month parsing failed

    # Filter data for the month of interest
    df_current_month = df_costs[
        (df_costs["Month"].dt.date >= month_of_interest_start) &
        (df_costs["Month"].dt.date <= month_of_interest_end)
    ]

    # Filter data for the previous month
    df_previous_month = df_costs[
        (df_costs["Month"].dt.date >= prev_month_start) &
        (df_costs["Month"].dt.date <= prev_month_end)
    ]

    if df_current_month.empty and df_previous_month.empty:
        return pd.DataFrame({"Message": [f"No cost data available for {month_of_interest_start.strftime('%b %Y')} or {prev_month_start.strftime('%b %Y')} for segment '{segment}'."]})

    # Group by "Group Description" and sum "Amount in USD" for each month
    current_month_costs = df_current_month.groupby("Group Description")["Amount in USD"].sum().reset_index()
    current_month_costs.rename(columns={"Amount in USD": "Current Month Cost"}, inplace=True)

    previous_month_costs = df_previous_month.groupby("Group Description")["Amount in USD"].sum().reset_index()
    previous_month_costs.rename(columns={"Amount in USD": "Previous Month Cost"}, inplace=True)

    # Merge the two dataframes to compare costs
    merged_costs = pd.merge(
        current_month_costs,
        previous_month_costs,
        on="Group Description",
        how="outer"
    ).fillna(0)

    # Calculate the change
    merged_costs["Cost Change"] = merged_costs["Current Month Cost"] - merged_costs["Previous Month Cost"]

    # Filter for costs that increased (triggered margin drop)
    triggered_costs = merged_costs[merged_costs["Cost Change"] > 0].sort_values(by="Cost Change", ascending=False)

    if triggered_costs.empty:
        return pd.DataFrame({"Message": [f"No cost increases found in '{segment}' for {month_of_interest_start.strftime('%b %Y')} compared to {prev_month_start.strftime('%b %Y')}."]})

    # Format for display
    triggered_costs["Current Month Cost"] = triggered_costs["Current Month Cost"].apply(lambda x: f"${x:,.2f}")
    triggered_costs["Previous Month Cost"] = triggered_costs["Previous Month Cost"].apply(lambda x: f"${x:,.2f}")
    triggered_costs["Cost Change"] = triggered_costs["Cost Change"].apply(lambda x: f"${x:,.2f}")

    # Add S.No
    triggered_costs.reset_index(drop=True, inplace=True)
    triggered_costs.index = triggered_costs.index + 1
    triggered_costs = triggered_costs.rename_axis("S.No").reset_index()

    return triggered_costs[["S.No", "Group Description", "Current Month Cost", "Previous Month Cost", "Cost Change"]]

