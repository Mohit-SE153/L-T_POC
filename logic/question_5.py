import pandas as pd
from datetime import datetime, timedelta


def run_logic(df, parsed_filters):
    """
    Analyzes YoY, QoQ, MoM revenue trends based on parsed filters.
    Handles cases where start_date, end_date, period_type, or grouping_dimension are not provided by parse_query_filters.
    """
    # 1. Extract parsed filters, providing defaults if None
    start_date = parsed_filters.get("start_date")
    end_date = parsed_filters.get("end_date")

    # The 'grouping_dimension' from parsed_filters defaults to 'All'
    grouping_dimension = parsed_filters.get("grouping_dimension", "All")

    # Map friendly names to actual DataFrame column names
    grouping_column_df_name = None
    if grouping_dimension.lower() == 'du':
        grouping_column_df_name = 'PVDU'
    elif grouping_dimension.lower() == 'bu':
        grouping_column_df_name = 'Exec DG'
    elif grouping_dimension.lower() == 'account':
        grouping_column_df_name = 'FinalCustomerName'
    # If grouping_dimension is 'All', grouping_column_df_name remains None, and we'll group by overall total.

    # 2. Set default date range if no explicit dates are provided in parsed_filters
    if start_date is None or end_date is None:
        print(f"DEBUG Q5: No specific dates parsed from query. Setting default range to cover all available data.")

        # Ensure 'Month' column is in datetime format first to get min/max
        df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
        df.dropna(subset=['Month'], inplace=True)  # Remove rows where 'Month' is NaT after conversion

        # Determine the earliest and latest dates in your actual DataFrame
        # Provide robust fallbacks if DataFrame is empty for some reason
        min_df_date = df['Month'].min().date() if not df.empty else datetime(2023, 1, 1).date()  # Fallback date
        max_df_date = df['Month'].max().date() if not df.empty else datetime.now().date()  # Fallback date

        # Set the default range to cover ALL available data in your DataFrame
        start_date = min_df_date
        end_date = max_df_date
        print(f"DEBUG Q5: Defaulting analysis range to: {start_date} to {end_date}")

    # Ensure 'Month' column is in datetime format again, in case it was not processed before
    df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
    df.dropna(subset=['Month'], inplace=True)  # Remove rows where 'Month' is NaT after conversion

    # Filter data to the determined date range
    # Ensure both sides of comparison are datetime.date objects for consistency
    df_filtered = df[(df['Month'].dt.date >= start_date) & (df['Month'].dt.date <= end_date)].copy()

    print(f"DEBUG Q5: Shape of df_filtered after date filtering: {df_filtered.shape}")

    if df_filtered.empty:
        return {"Message": "No data available for the specified date range. Please try a different period."}

    # Ensure 'Amount in USD' is numeric for calculations
    if 'Amount in USD' not in df_filtered.columns:
        return {
            "Message": "'Amount in USD' column not found in data for revenue analysis. Please check your data source."}

    df_filtered['Amount in USD'] = pd.to_numeric(df_filtered['Amount in USD'], errors='coerce')
    df_filtered.dropna(subset=['Amount in USD'], inplace=True)  # Drop rows where Amount in USD is NaN

    # --- Perform Revenue Trend Calculations ---
    # We will prepare dataframes for each trend type and grouping

    all_mom_data = []
    all_qoq_data = []
    all_yoy_data = []

    # Determine the actual grouping values to iterate through
    current_grouping_values = ['Total Revenue']
    if grouping_column_df_name and grouping_column_df_name in df_filtered.columns:
        current_grouping_values = df_filtered[grouping_column_df_name].unique()

    # Store original df_filtered for re-grouping in app.py for YoY chart
    original_filtered_df_for_charts = df_filtered.copy()

    for g_val in current_grouping_values:
        temp_df = df_filtered.copy()
        if grouping_column_df_name and grouping_column_df_name in df_filtered.columns:
            temp_df = df_filtered[df_filtered[grouping_column_df_name] == g_val].copy()

        if temp_df.empty:
            continue  # Skip if no data for this group

        # Ensure 'Month' is set as index for time series operations, resample to monthly start for consistent periods
        temp_df_resampled = temp_df.set_index('Month').resample('MS').sum(numeric_only=True).reset_index()

        # --- MoM Calculations ---
        monthly_data = temp_df_resampled.sort_values('Month').copy()
        # Ensure sufficient data points for comparison
        if monthly_data.shape[0] > 1:
            monthly_data['MoM_Prev_Revenue'] = monthly_data['Amount in USD'].shift(1)
            # Calculate MoM change, handle division by zero and NaNs
            monthly_data['MoM_Change'] = monthly_data.apply(
                lambda row: (row['Amount in USD'] - row['MoM_Prev_Revenue']) / row['MoM_Prev_Revenue'] * 100
                if pd.notna(row['MoM_Prev_Revenue']) and row['MoM_Prev_Revenue'] != 0 else 0,
                axis=1
            )
            monthly_data['MoM_Change'] = monthly_data['MoM_Change'].replace([float('inf'), -float('inf')], 0).fillna(0)
            monthly_data['Grouping_Dimension'] = grouping_dimension
            monthly_data['Grouping_Value'] = g_val
            all_mom_data.append(monthly_data)

        # --- QoQ Calculations ---
        # Aggregate to quarterly data
        quarterly_data = temp_df.set_index('Month').resample('QS').sum(
            numeric_only=True).reset_index()  # Quarter start frequency
        quarterly_data = quarterly_data.sort_values('Month').copy()
        if quarterly_data.shape[0] > 1:
            quarterly_data['QoQ_Prev_Revenue'] = quarterly_data['Amount in USD'].shift(1)
            # Calculate QoQ change
            quarterly_data['QoQ_Change'] = quarterly_data.apply(
                lambda row: (row['Amount in USD'] - row['QoQ_Prev_Revenue']) / row['QoQ_Prev_Revenue'] * 100
                if pd.notna(row['QoQ_Prev_Revenue']) and row['QoQ_Prev_Revenue'] != 0 else 0,
                axis=1
            )
            quarterly_data['QoQ_Change'] = quarterly_data['QoQ_Change'].replace([float('inf'), -float('inf')],
                                                                                0).fillna(0)
            quarterly_data['Grouping_Dimension'] = grouping_dimension
            quarterly_data['Grouping_Value'] = g_val
            all_qoq_data.append(quarterly_data)

        # --- YoY Calculations ---
        # Aggregate to yearly data
        yearly_data = temp_df.set_index('Month').resample('YS').sum(
            numeric_only=True).reset_index()  # Year start frequency
        yearly_data = yearly_data.sort_values('Month').copy()
        if yearly_data.shape[0] > 1:
            yearly_data['YoY_Prev_Revenue'] = yearly_data['Amount in USD'].shift(1)
            # Calculate YoY change
            yearly_data['YoY_Change'] = yearly_data.apply(
                lambda row: (row['Amount in USD'] - row['YoY_Prev_Revenue']) / row['YoY_Prev_Revenue'] * 100
                if pd.notna(row['YoY_Prev_Revenue']) and row['YoY_Prev_Revenue'] != 0 else 0,
                axis=1
            )
            yearly_data['YoY_Change'] = yearly_data['YoY_Change'].replace([float('inf'), -float('inf')], 0).fillna(0)
            yearly_data['Grouping_Dimension'] = grouping_dimension
            yearly_data['Grouping_Value'] = g_val
            all_yoy_data.append(yearly_data)

    # Concatenate all dataframes if multiple groups exist, otherwise use the single one.
    final_mom_df = pd.concat(all_mom_data).reset_index(drop=True) if all_mom_data else pd.DataFrame()
    final_qoq_df = pd.concat(all_qoq_data).reset_index(drop=True) if all_qoq_data else pd.DataFrame()
    final_yoy_df = pd.concat(all_yoy_data).reset_index(drop=True) if all_yoy_data else pd.DataFrame()

    # Return a dictionary containing the DataFrames for each trend type and grouping info
    return {
        "mom_data": final_mom_df,
        "qoq_data": final_qoq_df,
        "yoy_data": final_yoy_df,
        "grouping_dimension_from_query": grouping_dimension,  # Original grouping requested by query
        "df_filtered_for_charts": original_filtered_df_for_charts
        # Pass the initially filtered DF for dynamic grouping in app.py
    }