# logic/question_4.py

import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import calendar

# Assume CB_GROUPS and REVENUE_GROUPS are imported or defined here for clarity,
# or ensure they are accessible globally if utils.py defines them.
# For now, let's explicitly import them from utils.py as it's cleaner.
from logic.utils import REVENUE_GROUPS, CB_GROUPS # Ensure these are exported from utils


def run_logic(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Calculates the Month-over-Month (M-o-M) trend of C&B cost as a percentage of total revenue.
    Applies optional date filtering.

    Args:
        df (pd.DataFrame): The main DataFrame containing financial data.
        filters (dict): Dictionary containing parsed date filter details (start_date, end_date, date_filter).

    Returns:
        pd.DataFrame: A DataFrame with the M-o-M trend, including Month, Total Revenue,
                      Total C&B Cost, and C&B % of Revenue.
                      Returns a DataFrame with a "Message" column if parsing failed or no data.
    """
    # Ensure 'Month' column is datetime (already done in app.py, but good to be safe if this function might be called independently)
    df["Month"] = pd.to_datetime(df["Month"], errors='coerce', dayfirst=True)
    df.dropna(subset=['Month'], inplace=True)

    # Apply date filter if present
    if filters.get("date_filter") and filters.get("start_date") and filters.get("end_date"):
        start_date = filters["start_date"]
        end_date = filters["end_date"]
        df_filtered_by_date = df[
            (df["Month"].dt.date >= start_date) &
            (df["Month"].dt.date <= end_date)
        ].copy()

        if df_filtered_by_date.empty:
            return pd.DataFrame({"Message": ["No data found for the specified date range for C&B trend analysis."]})
    else:
        df_filtered_by_date = df.copy() # Work on a copy if no date filter

    if df_filtered_by_date.empty:
        return pd.DataFrame({"Message": ["No data available to analyze C&B cost % of total revenue trend."]})

    # Ensure 'Amount in USD' is numeric
    df_filtered_by_date['Amount in USD'] = pd.to_numeric(df_filtered_by_date['Amount in USD'], errors='coerce')
    df_filtered_by_date.dropna(subset=['Amount in USD'], inplace=True)

    # Aggregate data month-by-month
    # First, get the month-level grouping column
    df_filtered_by_date['Month_Year'] = df_filtered_by_date['Month'].dt.to_period('M')

    # Calculate Total Revenue per month
    monthly_revenue = df_filtered_by_date[df_filtered_by_date["Group1"].isin(REVENUE_GROUPS)] \
        .groupby('Month_Year')["Amount in USD"].sum().reset_index()
    monthly_revenue.rename(columns={"Amount in USD": "Total Revenue"}, inplace=True)

    # Calculate Total C&B Cost per month
    monthly_cb_cost = df_filtered_by_date[df_filtered_by_date["Group Description"].isin(CB_GROUPS)] \
        .groupby('Month_Year')["Amount in USD"].sum().reset_index()
    monthly_cb_cost.rename(columns={"Amount in USD": "Total C&B Cost"}, inplace=True)

    # Merge revenue and C&B cost data
    merged_trend_df = pd.merge(
        monthly_revenue,
        monthly_cb_cost,
        on='Month_Year',
        how='outer' # Use outer to keep all months where either revenue or C&B exists
    ).fillna(0) # Fill NaN with 0 for months where one category might be missing

    # Convert Month_Year back to datetime for proper plotting
    merged_trend_df['Month'] = merged_trend_df['Month_Year'].dt.to_timestamp()
    merged_trend_df.sort_values(by='Month', inplace=True)

    # Calculate C&B Cost % of Revenue
    # Handle division by zero for revenue
    merged_trend_df['C&B % of Revenue'] = (
        (merged_trend_df['Total C&B Cost'] / merged_trend_df['Total Revenue'].replace(0, np.nan)) * 100
    ).fillna(0) # Fill NaN (from div by zero) with 0 or a placeholder as appropriate

    # Select and reorder columns for display
    result_for_display = merged_trend_df[['Month', 'Total Revenue', 'Total C&B Cost', 'C&B % of Revenue']].copy()

    # Format numeric columns for display
    result_for_display['Total Revenue'] = result_for_display['Total Revenue'].apply(lambda x: f"${x:,.2f}")
    result_for_display['Total C&B Cost'] = result_for_display['Total C&B Cost'].apply(lambda x: f"${x:,.2f}")
    result_for_display['C&B % of Revenue'] = result_for_display['C&B % of Revenue'].apply(lambda x: f"{x:,.2f}%")

    # Add S.No
    result_for_display.reset_index(drop=True, inplace=True)
    result_for_display.index = result_for_display.index + 1
    result_for_display = result_for_display.rename_axis("S.No").reset_index()

    return result_for_display