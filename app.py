import streamlit as st
import pandas as pd
import os
import pytz
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
import io

from dotenv import load_dotenv

load_dotenv()

# Import from your modules
from questions import get_question_id
from logic.utils import get_cm_query_details, get_cost_drop_query_details, parse_query_filters, REVENUE_GROUPS, \
    COST_GROUPS, CB_GROUPS # Ensure REVENUE_GROUPS, COST_GROUPS, CB_GROUPS are always imported if used

# Import specific logic functions with your preferred naming convention
from logic.question_1 import run_logic as run_question_1_logic
from logic.question_2 import run_logic as run_question_2_logic
from logic.question_3 import run_logic as run_question_3_logic
from logic.question_4 import run_logic as run_question_4_logic
from logic.question_5 import run_logic as run_question_5_logic # Your new question 5 logic


try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Visualizations disabled - Plotly not installed.")

# Azure Storage Configuration
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
AZURE_STORAGE_BLOB_NAME = os.getenv("AZURE_STORAGE_BLOB_NAME")

# Constructing the connection string (ensure this is secure in production)
AZURE_STORAGE_CONNECTION_STRING = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={AZURE_STORAGE_ACCOUNT_NAME};"
    f"AccountKey={AZURE_STORAGE_ACCOUNT_KEY};"
    f"EndpointSuffix=core.windows.net"
)


@st.cache_data
def load_data_from_azure(container_name, blob_name):
    """
    Loads financial data from an Azure Blob Storage CSV file.
    Caches the data to improve performance.
    """
    if not AZURE_STORAGE_ACCOUNT_NAME or not AZURE_STORAGE_ACCOUNT_KEY or not container_name or not blob_name:
        st.error("Azure Storage credentials or blob details are missing. Please set environment variables.")
        return pd.DataFrame({"Message": ["Azure Storage configuration error."]})
    try:
        # Initialize BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)

        # Download blob data into a BytesIO object
        csv_data = io.BytesIO(blob_client.download_blob().readall())
        csv_data.seek(0) # Reset stream position to the beginning

        # Read CSV data into a DataFrame
        df = pd.read_csv(
            csv_data, delimiter=',',
            dtype={'wbs id': str, 'Pulse Project ID': str, 'Contract ID': str, 'sales Region': str},
            low_memory=False
        )
        df = df.drop_duplicates() # Remove duplicate rows
        df["Amount in INR"] = df["Amount in INR"].round(2)
        df["Amount in USD"] = df["Amount in USD"].round(2)
        df["Month"] = pd.to_datetime(df["Month"], errors='coerce', dayfirst=True) # Convert 'Month' to datetime
        df.dropna(subset=['Month'], inplace=True) # Drop rows where 'Month' conversion failed
        return df
    except Exception as e:
        st.error(f"Error loading data from Azure Storage: {e}")
        st.info("Please check your Azure Storage settings. Ensure the blob is a valid CSV file.")
        return pd.DataFrame({"Message": [f"Failed to load data from Azure Storage: {e}"]})


def format_currency_dynamic(value):
    """
    Formats a numeric value into a dynamic currency string (e.g., $1.25M, $500K).
    """
    if pd.isna(value):
        return "N/A"
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    elif abs_value >= 1_000:
        return f"${value / 1_000:,.2f}K"
    else:
        return f"${value:,.2f}"


def display_kpi(col, label, value, delta_text=None, delta_color_class=""):
    """
    Displays a KPI using custom HTML/CSS for enhanced styling.
    """
    with col:
        st.markdown(f"""
            <div class="kpi-label-top">{label}</div>
            <div class="kpi-value-bottom">{value}</div>
            {f'<div class="kpi-delta-bottom {delta_color_class}">{delta_text}</div>' if delta_text else ''}
        """, unsafe_allow_html=True)


def main():
    st.set_page_config(layout="wide", page_title="CM Analyzer Pro", page_icon="📊")
    st.title("📊 L&T DataStream Financials")

    # Load data from Azure
    df = load_data_from_azure(AZURE_STORAGE_CONTAINER_NAME, AZURE_STORAGE_BLOB_NAME)
    if "Message" in df.columns and df["Message"].iloc[0].startswith("Azure Storage"):
        st.stop() # Stop if there's a configuration or loading error

    # --- Dummy Data for Local Testing without Azure (uncomment and replace df if needed) ---
    # Example dummy data structure, ensure it has columns used by your logic (Month, Amount in USD, Group1, etc.)
    # Make sure 'ONSITE' is in REVENUE_GROUPS for Q5 to work with dummy data
    # Make sure 'C&B Cost Onsite' is in CB_GROUPS for Q4 to work with dummy data
    #
    # dummy_data = {
    #     'Month': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01',
    #                              '2024-07-01', '2024-08-01', '2024-09-01', '2024-10-01', '2024-11-01', '2024-12-01',
    #                              '2025-01-01', '2025-02-01', '2025-03-01', '2025-04-01', '2025-05-01', '2025-06-01',
    #                              '2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01',
    #                              '2025-01-01', '2025-02-01', '2025-03-01', '2025-04-01', '2025-05-01', '2025-06-01'
    #                             ]),
    #     'Group1': ['ONSITE'] * 18 + ['Direct Expense'] * 12, # Assuming 'ONSITE' is a revenue group
    #     'Group Description': ['Revenue_Category'] * 18 + ['C&B Cost Onsite'] * 6 + ['Other Cost'] * 6, # Assuming 'C&B Cost Onsite' is a C&B group
    #     'Amount in USD': np.random.rand(30) * 100000,
    #     'FinalCustomerName': (['CustA'] * 3 + ['CustB'] * 3) * 3 + (['CustC'] * 3 + ['CustD'] * 3) * 2, # More diverse data
    #     'PVDU': (['DU1'] * 6 + ['DU2'] * 3 + ['DU1'] * 3) * 2 + (['DU3'] * 6 + ['DU4'] * 6), # More diverse data
    #     'Exec DG': (['BU_X'] * 6 + ['BU_Y'] * 6) * 2 + (['BU_Z'] * 6 + ['BU_W'] * 6), # More diverse data
    #     'Type': ['Revenue'] * 18 + ['Cost'] * 12, # Added 'Type' column
    #     'Segment': ['Tech'] * 15 + ['Finance'] * 15 # Added 'Segment' column
    # }
    # df = pd.DataFrame(dummy_data)
    # df["Month"] = pd.to_datetime(df["Month"], errors='coerce', dayfirst=True)
    # df['Amount in USD'] = pd.to_numeric(df['Amount in USD'], errors='coerce')
    # df.dropna(subset=['Month', 'Amount in USD'], inplace=True)
    # --- End Dummy Data ---

    # Custom CSS for KPI styling
    st.markdown("""
    <style>
    /* Global adjustments for the main content area */
    .block-container {
        padding-top: 3rem; /* Increased top padding to ensure title is not cropped */
        padding-left: 1.2 rem;
        padding-right: 1.2 rem;
        padding-bottom: 1.2 rem;
        max-width: unset; /* Ensure full width is used */
    }

    /* Adjustments for h2 and other headers */
    h2 {
        margin-top: 1rem !important; /* Standardize top margin for h2 */
        margin-bottom: 0.5rem !important;
        padding-top: 0.5 !important;
        padding-bottom: 0.5 !important;
    }

    /* Specific adjustment for the very first h2 (often the Key Metrics title) */
    div[data-testid="stVerticalBlock"] > h2:first-of-type {
        margin-left: 0rem !important; /* Align with the left edge of the main content */
        padding-left: 0rem !important;
    }

    /* Target the horizontal block containing the metrics (st.columns) */
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stHorizontalBlock"] {
        margin-top: 0.5 px !important;
        margin-bottom: 1rem !important;
        gap: 2 rem; /* Adjusted from 3rem to 1.5rem for tighter spacing between KPIs */
        width: 100%; /* Ensure the columns container takes full available width */
        justify-content: flex-start; /* Align items to the start of the container */
    }

    /* KPI Styling - No background, label on top, value on bottom */
    .kpi-label-top {
        font-size: 1rem !important;
        color: #b0b3b8 !important;
        margin-bottom: 0.2rem;
        text-align: left;
        width: 100%;
    }

    .kpi-value-bottom {
        font-size: 1.8rem !important;
        font-weight: bold !important;
        color: #fff !important;
        margin-bottom: 0.2rem;
        text-align: left;
        width: 100%;
    }

    .kpi-delta-bottom {
        font-size: 0.9rem !important;
        text-align: left;
        width: 100%;
    }

    /* Delta colors */
    .delta-green {
        color: #00FF00 !important;
        font-weight: bold !important;
    }
    .delta-red {
        color: #FF4B4B !important;
        font-weight: bold !important;
    }
    .delta-yellow {
        color: #FFD700 !important;
        font-weight: bold !important;
    }

    /* Hide Streamlit's default metric widget */
    .stMetric {
        display: none !important;
        visibility: hidden !important;
        height: 0;
        padding: 0;
        margin: 0;
    }
    .stMetric > div {
        display: none !important;
        visibility: hidden !important;
        height: 0;
        padding: 0;
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    user_query = st.text_input(
        "💬 Ask your question (e.g., 'List customers with CM > 90% last quarter', 'Which cost triggered the Margin drop last month in Transportation', 'What is M-o-M trend of C&B cost % w.r.t total revenue', 'What is the YoY, QoQ, MoM revenue for DU/BU/account')",
        placeholder="Enter your query here..."
    )

    if user_query:
        with st.spinner("Analyzing your question..."):
            question_id = get_question_id(user_input=user_query)
            # --- DEBUGGING: Display detected question_id ---
            #st.write(f"DEBUG: Detected question_id: **{question_id}**")

            parsed_filters = parse_query_filters(user_query, question_id)
            # --- DEBUGGING: Display parsed_filters ---
            #st.write(f"DEBUG: Parsed filters: {parsed_filters}")

        try:
            result_df = pd.DataFrame()
            if "Message" in parsed_filters and parsed_filters["Message"].startswith("API keys for Azure OpenAI are not configured"):
                st.error(parsed_filters["Message"])
                st.stop() # Stop execution if API keys are missing

            if question_id == "unknown":
                st.warning("Sorry, I couldn't understand your question or it's not yet supported.")
                st.info("Try queries like: 'List customers with CM > 50%' or 'What is M-o-M trend of C&B cost % w.r.t total revenue'...")
            elif question_id == "question_1":
                result_df = run_question_1_logic(df.copy(), parsed_filters) # Pass a copy
                if "Message" in result_df.columns:
                    st.warning(result_df["Message"].iloc[0])
                else:
                    cm_threshold = parsed_filters.get("lower", 0.0) * 100 if parsed_filters.get("type") in [
                        "greater_than", "between", "equals"] and parsed_filters.get("lower") is not None else 0.0

                    date_filter_msg = "📅 Showing all available data (no specific date filter applied from query)"
                    if parsed_filters.get("date_filter") and parsed_filters.get("start_date") and parsed_filters.get(
                            "end_date"):
                        date_filter_msg = f"📅 Date Filter: {parsed_filters['start_date'].strftime('%Y-%m-%d')} to {parsed_filters['end_date'].strftime('%Y-%m-%d')}"
                    st.success(date_filter_msg)

                    cm_filter_msg = "🔍 Showing all Contribution Margins (no specific CM filter applied from query)"
                    if parsed_filters["type"] == "greater_than":
                        cm_filter_msg = f"🔍 Applying CM Filter: CM > {parsed_filters['lower'] * 100:.2f}%"
                    elif parsed_filters["type"] == "less_than":
                        cm_filter_msg = f"🔍 Applying CM Filter: CM < {parsed_filters['lower'] * 100:.2f}%"
                    elif parsed_filters["type"] == "between":
                        cm_filter_msg = f"🔍 Applying CM Filter: CM between {parsed_filters['lower'] * 100:.2f}% and {parsed_filters['upper'] * 100:.2f}%"
                    elif parsed_filters["type"] == "equals":
                        cm_filter_msg = f"🔍 Applying CM Filter: CM = {parsed_filters['lower'] * 100:.2f}%"
                    st.write(cm_filter_msg)

                    st.markdown("## 📊 Key Metrics", unsafe_allow_html=True)
                    # Corrected order for display to match your image
                    col1, col2, col3, col4 = st.columns(4)
                    total_customers = len(result_df)
                    total_rev = result_df['Revenue'].str.replace(r'[$,]', '', regex=True).astype(float).sum()
                    total_cost = result_df['Cost'].str.replace(r'[$,]', '', regex=True).astype(float).sum()
                    avg_cm = result_df['CM_Value'].replace([np.inf, -np.inf], np.nan).mean()

                    # Re-ordered KPI display to match the image: Total Customers, Total Revenue, Total Cost, Average CM
                    display_kpi(col1, "Total Customers", total_customers)
                    display_kpi(col2, "Total Revenue", format_currency_dynamic(total_rev))
                    display_kpi(col3, "Total Cost", format_currency_dynamic(total_cost))
                    display_kpi(col4, "Average CM", f"{avg_cm:.2f}%" if pd.notna(avg_cm) else "N/A")

                    tab1, tab2 = st.tabs(["📋 Data Table", "📈 Visual Analysis"])
                    with tab1:
                        def color_cm(val):
                            try:
                                cm_val = float(str(val).replace('%', ''))
                                if cm_val > cm_threshold:
                                    return 'background-color: #d4edda; color: #155724;'
                                elif cm_val > 30:
                                    return 'background-color: #fff3cd; color: #856404;'
                                elif cm_val < 0:
                                    return 'background-color: #f8d7da; color: #721c24;'
                                return ''
                            except ValueError:
                                return ''

                        display_df = result_df.drop(columns=["CM_Value"])
                        styled_display_df = display_df.style.map(color_cm, subset=['CM (%)']).set_table_styles([
                            {'selector': 'th', 'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]}
                        ])
                        st.dataframe(styled_display_df, use_container_width=True)
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name='cm_analysis.csv',
                            mime='text/csv'
                        )
                    with tab2:
                        if PLOTLY_AVAILABLE:
                            st.subheader("CM Distribution")
                            hist_df = result_df.copy()
                            hist_df['CM_Value'] = hist_df['CM_Value'].clip(-100, 200) # Clip for better visualization
                            fig_hist = px.histogram(
                                hist_df,
                                x="CM_Value",
                                nbins=20,
                                title="Distribution of Contribution Margins",
                                labels={"CM_Value": "Contribution Margin (%)"},
                                color_discrete_sequence=['#636EFA']
                            )
                            fig_hist.update_layout(
                                bargap=0.1,
                                xaxis_title="Contribution Margin (%)",
                                yaxis_title="Number of Customers"
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)

                            st.subheader(f"Top Performers (CM > {cm_threshold:.0f}%)")
                            top_performers = result_df[result_df['CM_Value'] > cm_threshold]
                            if not top_performers.empty:
                                max_cm_value = top_performers['CM_Value'].max()
                                yaxis_max = max_cm_value * 1.15
                                if yaxis_max < 100:
                                    yaxis_max = 100

                                fig_top = px.bar(
                                    top_performers.nlargest(10, 'CM_Value'),
                                    x="FinalCustomerName",
                                    y="CM_Value",
                                    text="CM (%)",
                                    title=f"Top Performing Customers (CM > {cm_threshold:.0f}%)",
                                    color="CM_Value",
                                    color_continuous_scale='greens',
                                    hover_data={
                                        "FinalCustomerName": False,
                                        "CM_Value": False,
                                        "CM (%)": True,
                                    },
                                    labels={
                                        "FinalCustomerName": "Customer",
                                        "CM (%)": "CM"
                                    }
                                )
                                fig_top.update_layout(
                                    xaxis_title="Customer Name",
                                    yaxis_title="Contribution Margin (%)",
                                    xaxis={'categoryorder': 'total descending'},
                                    coloraxis_showscale=False,
                                    yaxis_range=[0, yaxis_max]
                                )
                                fig_top.update_traces(textposition='outside', texttemplate='%{text}')
                                st.plotly_chart(fig_top, use_container_width=True)
                            else:
                                st.warning(f"No customers found with CM > {cm_threshold:.0f}%")
                        else:
                            st.warning(
                                "Plotly is not installed. Install with: pip install plotly to enable visualizations.")
                            st.write("Here's a preview of the raw data:")
                            st.dataframe(
                                result_df.nlargest(10, 'CM_Value')[
                                    ['S.No', 'FinalCustomerName', 'CM (%)', 'Revenue', 'Cost']],
                                height=400
                            )
            elif question_id == "question_2":
                st.subheader("📉 Cost Drop Analysis")
                # Filters for Q2 are generally handled by get_cost_drop_query_details within utils
                # The parsed_filters already contain the info needed.
                # filters = get_cost_drop_query_details(user_query) # This line is redundant if parsed_filters is already from this function.

                if parsed_filters.get("Message"): # Use parsed_filters here
                    st.warning(parsed_filters["Message"])
                    return

                result_df = run_question_2_logic(df.copy(), parsed_filters) # Pass a copy

                if "Message" in result_df.columns:
                    st.warning(result_df["Message"].iloc[0])
                else:
                    current_month_display_name = parsed_filters.get("month_of_interest_name", "N/A Month")
                    compare_to_month_display_name = parsed_filters.get("compare_to_month_name", "N/A Month")

                    st.success(
                        f"Analysis for Segment: '{parsed_filters.get('segment', 'All')}' for {current_month_display_name} vs {compare_to_month_display_name}")

                    st.write("Costs that increased (potentially triggered margin drop):")
                    st.dataframe(result_df, use_container_width=True)
            elif question_id == "question_3":
                st.subheader("💰 C&B Cost Variation Analysis")
                result_df = run_question_3_logic(df.copy(), parsed_filters) # Pass a copy
                if "Message" in result_df.columns:
                    st.warning(result_df["Message"].iloc[0])
                else:
                    comparison_type = result_df["Comparison_Type"].iloc[0]

                    p1_name_display = parsed_filters.get("period1_name", "Period 1")
                    p2_name_display = parsed_filters.get("period2_name", "Period 2")

                    if comparison_type == "quarter":
                        p1_start_date = parsed_filters.get("period1_start")
                        p2_start_date = parsed_filters.get("period2_start")

                        # Corrected: Determine quarter string from the month of the start date
                        if p1_start_date:
                            p1_name_display = f"Q{(p1_start_date.month - 1) // 3 + 1} {p1_start_date.year}"
                        if p2_start_date:
                            p2_name_display = f"Q{(p2_start_date.month - 1) // 3 + 1} {p2_start_date.year}"

                        st.info(
                            f"Comparing C&B costs for **{p1_name_display}** vs **{p2_name_display}**")
                    else: # Monthly comparison
                        p1_start_date = parsed_filters.get("period1_start")
                        p2_start_date = parsed_filters.get("period2_start")

                        # Refine period names for display
                        if p1_start_date:
                            p1_name_display = f"{p1_start_date.strftime('%B %Y')}"
                        if p2_start_date:
                            p2_name_display = f"{p2_start_date.strftime('%B %Y')}"
                        st.info(
                            f"Comparing C&B costs for **{p1_name_display}** vs **{p2_name_display}**")


                    p1_cost_numeric = result_df["Period1_Cost_Numeric"].iloc[
                        0] if "Period1_Cost_Numeric" in result_df.columns else None
                    p2_cost_numeric = result_df["Period2_Cost_Numeric"].iloc[
                        0] if "Period2_Cost_Numeric" in result_df.columns else None
                    absolute_variation = p2_cost_numeric - p1_cost_numeric if pd.notna(p1_cost_numeric) and pd.notna(
                        p2_cost_numeric) else None
                    percentage_variation = result_df["Percentage_Variation"].iloc[
                        0] if "Percentage_Variation" in result_df.columns else None

                    st.markdown("## Key Metrics", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)

                    display_kpi(col1, f"C&B Cost {p1_name_display}", format_currency_dynamic(p1_cost_numeric))
                    display_kpi(col2, f"C&B Cost {p2_name_display}", format_currency_dynamic(p2_cost_numeric))

                    if pd.notna(absolute_variation):
                        variation_text = format_currency_dynamic(absolute_variation)
                        percentage_text = "N/A"
                        color_class = "" # Default color

                        if pd.notna(percentage_variation):
                            percentage_text = f"{percentage_variation:+.2f}%"
                            if percentage_variation > 0:
                                color_class = "delta-green" # Red for increase in cost
                            elif percentage_variation < 0:
                                color_class = "delta-red" # Green for decrease in cost
                            else:
                                color_class = "delta-yellow" # No change

                        display_kpi(col3, "Variation", variation_text, percentage_text, color_class)
                    else:
                        display_kpi(col3, "Variation", "N/A", "N/A", "delta-yellow")

                    if PLOTLY_AVAILABLE and pd.notna(p1_cost_numeric) and pd.notna(p2_cost_numeric):
                        st.subheader(
                            f"C&B Cost Comparison ({'Quarter' if comparison_type == 'quarter' else 'Month'}-over-{'Quarter' if comparison_type == 'quarter' else 'Month'})")
                        chart_data = pd.DataFrame({
                            "Period": [p1_name_display, p2_name_display],
                            "C&B Cost": [p1_cost_numeric, p2_cost_numeric]
                        })
                        fig = px.bar(
                            chart_data,
                            x="Period",
                            y="C&B Cost",
                            title=f"C&B Cost Comparison Between {p1_name_display} and {p2_name_display}",
                            text="C&B Cost",
                            color="Period",
                            color_discrete_map={
                                p1_name_display: "#636EFA",
                                p2_name_display: "#EF553B"
                            }
                        )
                        fig.update_traces(texttemplate='$%{y:,.2f}', textposition='outside')
                        fig.update_layout(
                            yaxis_title="C&B Cost (USD)",
                            yaxis_range=[0, max(p1_cost_numeric, p2_cost_numeric) * 1.15] # Adjust Y-axis for better fit
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    elif not PLOTLY_AVAILABLE:
                        st.warning(
                            "Plotly is not installed. Install with: pip install plotly to enable visualizations.")
                    else:
                        st.info("Not enough data to generate a comparison chart.")
            elif question_id == "question_4":
                st.subheader("📊 C&B Cost % of Total Revenue Trend (M-o-M)")
                result_df = run_question_4_logic(df.copy(), parsed_filters) # Pass a copy

                if "Message" in result_df.columns:
                    st.warning(result_df["Message"].iloc[0])
                else:
                    date_filter_msg = "📅 Showing all available data (no specific date filter applied from query)"
                    if parsed_filters.get("date_filter") and parsed_filters.get("start_date") and parsed_filters.get("end_date"):
                        date_filter_msg = f"📅 Date Filter: {parsed_filters['start_date'].strftime('%Y-%m-%d')} to {parsed_filters['end_date'].strftime('%Y-%m-%d')}"
                    st.success(date_filter_msg)

                    st.markdown("### Monthly Trend Data")
                    st.dataframe(result_df, use_container_width=True)

                    if PLOTLY_AVAILABLE:
                        st.markdown("### Visual Trend Analysis")
                        plot_df = result_df.copy()
                        plot_df['Total Revenue Numeric'] = plot_df['Total Revenue'].str.replace(r'[$,]', '', regex=True).astype(float)
                        plot_df['Total C&B Cost Numeric'] = plot_df['Total C&B Cost'].str.replace(r'[$,]', '', regex=True).astype(float)
                        plot_df['C&B % of Revenue Numeric'] = plot_df['C&B % of Revenue'].str.replace('%', '', regex=False).astype(float)

                        fig_trend_cb_revenue = px.line(
                            plot_df,
                            x='Month',
                            y='C&B % of Revenue Numeric',
                            title='C&B Cost as Percentage of Total Revenue (M-o-M Trend)',
                            labels={
                                'Month': 'Month',
                                'C&B % of Revenue Numeric': 'C&B % of Revenue'
                            },
                            line_shape='linear',
                            markers=True,
                            color_discrete_sequence=px.colors.qualitative.Plotly
                        )
                        fig_trend_cb_revenue.update_traces(mode='lines+markers', hovertemplate=
                            '<b>Month</b>: %{x|%B %Y}<br>' +
                            '<b>C&B % of Revenue</b>: %{y:,.2f}%<extra></extra>'
                        )
                        fig_trend_cb_revenue.update_layout(
                            xaxis_title='Month',
                            yaxis_title='C&B % of Revenue',
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig_trend_cb_revenue, use_container_width=True)

                        st.markdown("#### Monthly Total Revenue and C&B Cost")
                        fig_revenue_cb_bar = px.bar(
                            plot_df,
                            x='Month',
                            y=['Total Revenue Numeric', 'Total C&B Cost Numeric'],
                            title='Monthly Total Revenue vs. Total C&B Cost',
                            labels={
                                'Month': 'Month',
                                'value': 'Amount (USD)',
                                'variable': 'Category'
                            },
                            barmode='group',
                            color_discrete_sequence=px.colors.qualitative.D3
                        )
                        fig_revenue_cb_bar.update_layout(
                            xaxis_title='Month',
                            yaxis_title='Amount (USD)',
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig_revenue_cb_bar, use_container_width=True)
                    else:
                        st.warning("Plotly is not installed. Install with: pip install plotly to enable visualizations.")
                        st.write("Here's a preview of the raw data:")
                        st.dataframe(result_df, height=400)

            elif question_id == "question_5":  # --- Display for Question 5 ---
                st.subheader("📊 Revenue Trend Analysis (YoY, QoQ, MoM)")
                # Call the run_logic for question 5 which returns a dictionary of DataFrames
                q5_output_dict = run_question_5_logic(df.copy(), parsed_filters)

                # Check if q5_output_dict contains a "Message" key (indicating an error/no data)
                if isinstance(q5_output_dict, dict) and "Message" in q5_output_dict:
                    st.error(q5_output_dict["Message"])
                # Check if all returned dataframes are empty
                elif isinstance(q5_output_dict, dict) and \
                        q5_output_dict["mom_data"].empty and \
                        q5_output_dict["qoq_data"].empty and \
                        q5_output_dict["yoy_data"].empty:
                    st.info("No sufficient data to calculate trends for the selected criteria or periods.")
                elif isinstance(q5_output_dict, dict):
                    # DataFrames for charts are now in q5_output_dict dictionary
                    mom_df = q5_output_dict["mom_data"]
                    qoq_df = q5_output_dict["qoq_data"]
                    yoy_df_from_logic = q5_output_dict["yoy_data"]  # This is based on initial grouping
                    actual_grouping_dimension_from_query = q5_output_dict["grouping_dimension_from_query"]
                    df_filtered_for_charts = q5_output_dict[
                        "df_filtered_for_charts"]  # The filtered data for dynamic grouping in app.py

                    # Use the date filter message from parsed_filters if available
                    date_filter_msg = "📅 Showing all available data (no specific date filter applied from query)"
                    if parsed_filters.get("date_filter") and parsed_filters.get("start_date") and parsed_filters.get(
                            "end_date"):
                        # Check if start_date and end_date are actual datetime objects
                        if isinstance(parsed_filters['start_date'], datetime) and isinstance(parsed_filters['end_date'],
                                                                                             datetime):
                            date_filter_msg = f"📅 Date Filter: {parsed_filters['start_date'].strftime('%Y-%m-%d')} to {parsed_filters['end_date'].strftime('%Y-%m-%d')}"
                        else:
                            st.warning("Parsed dates are not valid datetime objects. Displaying all available data.")
                    st.success(date_filter_msg)

                    # --- Tabbed Interface for MoM, QoQ, YoY ---
                    tab1, tab2, tab3 = st.tabs(["MoM Trend", "QoQ Trend", "YoY Trend"])

                    # Mapping for slicer options to actual column names in your DataFrame
                    grouping_col_map = {
                        "DU": "PVDU",
                        "BU": "Exec DG",
                        "Account": "FinalCustomerName",
                        "All": None  # Special case for 'All'
                    }

                    # Determine available grouping columns based on the actual filtered data
                    available_grouping_options = ["All"]  # Always include 'All'
                    if 'PVDU' in df_filtered_for_charts.columns: available_grouping_options.append("DU")
                    if 'Exec DG' in df_filtered_for_charts.columns: available_grouping_options.append("BU")
                    if 'FinalCustomerName' in df_filtered_for_charts.columns: available_grouping_options.append(
                        "Account")

                    # Remove duplicates and ensure order
                    available_grouping_options = list(dict.fromkeys(available_grouping_options))

                    with tab1:
                        st.markdown("#### Month-over-Month Revenue Trend")

                        # Set default index based on the actual_grouping_dimension from the query
                        default_mom_index = 0
                        if actual_grouping_dimension_from_query in available_grouping_options:
                            default_mom_index = available_grouping_options.index(actual_grouping_dimension_from_query)

                        group_by_mom = st.radio(
                            "Group MoM Trend By:",
                            available_grouping_options,
                            index=default_mom_index,
                            key="mom_slicer_q5",
                            horizontal=True
                        )

                        selected_mom_dim_col = grouping_col_map.get(
                            group_by_mom)  # Get actual column name or None for 'All'

                        # Re-aggregate mom_df based on selected_mom_dim_col
                        if selected_mom_dim_col and selected_mom_dim_col in df_filtered_for_charts.columns:
                            temp_mom_data = \
                            df_filtered_for_charts.groupby([pd.Grouper(key='Month', freq='MS'), selected_mom_dim_col])[
                                'Amount in USD'].sum().reset_index()
                            temp_mom_data = temp_mom_data.sort_values('Month')
                            temp_mom_data['MoM_Prev_Revenue'] = temp_mom_data.groupby(selected_mom_dim_col)[
                                'Amount in USD'].shift(1)
                            temp_mom_data['MoM_Change'] = (temp_mom_data['Amount in USD'] - temp_mom_data[
                                'MoM_Prev_Revenue']) / temp_mom_data['MoM_Prev_Revenue'] * 100
                            temp_mom_data['MoM_Change'] = temp_mom_data['MoM_Change'].replace(
                                [float('inf'), -float('inf')], np.nan).fillna(0)
                            mom_plot_data = temp_mom_data
                            mom_plot_color_col = selected_mom_dim_col
                        else:  # Handle 'All' or no specific grouping column
                            overall_mom_data = df_filtered_for_charts.groupby(pd.Grouper(key='Month', freq='MS'))[
                                'Amount in USD'].sum().reset_index()
                            overall_mom_data = overall_mom_data.sort_values('Month')
                            overall_mom_data['MoM_Prev_Revenue'] = overall_mom_data['Amount in USD'].shift(1)
                            overall_mom_data['MoM_Change'] = (overall_mom_data['Amount in USD'] - overall_mom_data[
                                'MoM_Prev_Revenue']) / overall_mom_data['MoM_Prev_Revenue'] * 100
                            overall_mom_data['MoM_Change'] = overall_mom_data['MoM_Change'].replace(
                                [float('inf'), -float('inf')], np.nan).fillna(0)
                            overall_mom_data[
                                'Grouping_Value'] = 'Total Revenue'  # Add a common value for 'color' if not grouping
                            mom_plot_data = overall_mom_data
                            mom_plot_color_col = 'Grouping_Value'  # Use this generic column for coloring

                        if not mom_plot_data.empty and mom_plot_data.shape[0] > 1:
                            fig_mom_rev = px.line(
                                mom_plot_data,
                                x='Month',
                                y='Amount in USD',
                                color=mom_plot_color_col,  # Use the determined color column
                                title=f'MoM Revenue Trend by {group_by_mom}',
                                labels={
                                    'Amount in USD': 'Revenue (USD)',
                                    'Month': 'Month'
                                }
                            )
                            fig_mom_rev.update_traces(mode='lines+markers', hovertemplate=
                            '<b>%{x|%B %Y}</b><br>' +
                            'Revenue: %{y:,.2f}<br>' +
                            'MoM Change: %{customdata[0]:.2f}%<extra></extra>'
                                                      )
                            # Add MoM_Change to customdata for hover (assuming it's the first in custom_data_columns)
                            fig_mom_rev.update_traces(customdata=mom_plot_data[['MoM_Change']].values)
                            fig_mom_rev.update_layout(xaxis_title="Month", yaxis_title="Revenue (USD)")
                            st.plotly_chart(fig_mom_rev, use_container_width=True)

                            # MoM % Change Chart
                            fig_mom_pct = px.line(
                                mom_plot_data,
                                x='Month',
                                y='MoM_Change',  # Direct column name
                                color=mom_plot_color_col,
                                title=f'MoM Revenue Percentage Change by {group_by_mom}',
                                labels={
                                    'MoM_Change': 'MoM Change (%)',
                                    'Month': 'Month'
                                }
                            )
                            fig_mom_pct.update_layout(xaxis_title="Month", yaxis_title="MoM Change (%)",
                                                      yaxis_tickformat=".2f%")
                            st.plotly_chart(fig_mom_pct, use_container_width=True)

                            with st.expander("Show MoM Detailed Data"):
                                display_cols = ['Month', 'Amount in USD', 'MoM_Change']
                                if mom_plot_color_col and mom_plot_color_col in mom_plot_data.columns and mom_plot_color_col != 'Grouping_Value':
                                    display_cols.insert(1, mom_plot_color_col)  # Add grouping column if present

                                st.dataframe(mom_plot_data[display_cols].style.format({
                                    'Amount in USD': '$ {:,.2f}',
                                    'MoM_Change': '{:.2f}%'
                                }), use_container_width=True)
                        else:
                            st.info(
                                f"Not enough data points for Month-over-Month trend analysis grouped by {group_by_mom}.")

                    with tab2:
                        st.markdown("#### Quarter-over-Quarter Revenue Trend")

                        default_qoq_index = 0
                        if actual_grouping_dimension_from_query in available_grouping_options:
                            default_qoq_index = available_grouping_options.index(actual_grouping_dimension_from_query)

                        group_by_qoq = st.radio(
                            "Group QoQ Trend By:",
                            available_grouping_options,
                            index=default_qoq_index,
                            key="qoq_slicer_q5",
                            horizontal=True
                        )
                        selected_qoq_dim_col = grouping_col_map.get(group_by_qoq)

                        if selected_qoq_dim_col and selected_qoq_dim_col in df_filtered_for_charts.columns:
                            temp_qoq_data = \
                            df_filtered_for_charts.groupby([pd.Grouper(key='Month', freq='QS'), selected_qoq_dim_col])[
                                'Amount in USD'].sum().reset_index()
                            temp_qoq_data = temp_qoq_data.sort_values('Month')
                            temp_qoq_data['QoQ_Prev_Revenue'] = temp_qoq_data.groupby(selected_qoq_dim_col)[
                                'Amount in USD'].shift(1)
                            temp_qoq_data['QoQ_Change'] = (temp_qoq_data['Amount in USD'] - temp_qoq_data[
                                'QoQ_Prev_Revenue']) / temp_qoq_data['QoQ_Prev_Revenue'] * 100
                            temp_qoq_data['QoQ_Change'] = temp_qoq_data['QoQ_Change'].replace(
                                [float('inf'), -float('inf')], np.nan).fillna(0)
                            qoq_plot_data = temp_qoq_data
                            qoq_plot_color_col = selected_qoq_dim_col
                        else:
                            overall_qoq_data = df_filtered_for_charts.groupby(pd.Grouper(key='Month', freq='QS'))[
                                'Amount in USD'].sum().reset_index()
                            overall_qoq_data = overall_qoq_data.sort_values('Month')
                            overall_qoq_data['QoQ_Prev_Revenue'] = overall_qoq_data['Amount in USD'].shift(1)
                            overall_qoq_data['QoQ_Change'] = (overall_qoq_data['Amount in USD'] - overall_qoq_data[
                                'QoQ_Prev_Revenue']) / overall_qoq_data['QoQ_Prev_Revenue'] * 100
                            overall_qoq_data['QoQ_Change'] = overall_qoq_data['QoQ_Change'].replace(
                                [float('inf'), -float('inf')], np.nan).fillna(0)
                            overall_qoq_data['Grouping_Value'] = 'Total Revenue'
                            qoq_plot_data = overall_qoq_data
                            qoq_plot_color_col = 'Grouping_Value'

                        if not qoq_plot_data.empty and qoq_plot_data.shape[0] > 1:
                            fig_qoq_rev = px.line(
                                qoq_plot_data,
                                x='Month',  # Use actual datetime for correct x-axis
                                y='Amount in USD',
                                color=qoq_plot_color_col,
                                title=f'QoQ Revenue Trend by {group_by_qoq}',
                                labels={
                                    'Amount in USD': 'Revenue (USD)',
                                    'Month': 'Quarter'
                                }
                            )
                            fig_qoq_rev.update_traces(mode='lines+markers', hovertemplate=
                            '<b>%{x|%Y Q%q}</b><br>' +  # Format for quarter
                            'Revenue: %{y:,.2f}<br>' +
                            'QoQ Change: %{customdata[0]:.2f}%<extra></extra>'
                                                      )
                            fig_qoq_rev.update_traces(customdata=qoq_plot_data[['QoQ_Change']].values)
                            fig_qoq_rev.update_layout(xaxis_title="Quarter", yaxis_title="Revenue (USD)")
                            st.plotly_chart(fig_qoq_rev, use_container_width=True)

                            fig_qoq_pct = px.line(
                                qoq_plot_data,
                                x='Month',
                                y='QoQ_Change',
                                color=qoq_plot_color_col,
                                title=f'QoQ Revenue Percentage Change by {group_by_qoq}',
                                labels={
                                    'QoQ_Change': 'QoQ Change (%)',
                                    'Month': 'Quarter'
                                }
                            )
                            fig_qoq_pct.update_layout(xaxis_title="Quarter", yaxis_title="QoQ Change (%)",
                                                      yaxis_tickformat=".2f%")
                            st.plotly_chart(fig_qoq_pct, use_container_width=True)

                            with st.expander("Show QoQ Detailed Data"):
                                display_cols = ['Month', 'Amount in USD', 'QoQ_Change']
                                if qoq_plot_color_col and qoq_plot_color_col in qoq_plot_data.columns and qoq_plot_color_col != 'Grouping_Value':
                                    display_cols.insert(1, qoq_plot_color_col)
                                st.dataframe(qoq_plot_data[display_cols].style.format({
                                    'Amount in USD': '$ {:,.2f}',
                                    'QoQ_Change': '{:.2f}%'
                                }), use_container_width=True)
                        else:
                            st.info(
                                f"Not enough data points for Quarter-over-Quarter trend analysis grouped by {group_by_qoq}.")

                    with tab3:
                        st.markdown("#### Year-over-Year Revenue Trend")

                        default_yoy_index = 0
                        if actual_grouping_dimension_from_query in available_grouping_options:
                            default_yoy_index = available_grouping_options.index(actual_grouping_dimension_from_query)

                        group_by_yoy = st.radio(
                            "Group YoY Trend By:",
                            available_grouping_options,
                            index=default_yoy_index,
                            key="yoy_slicer_q5",
                            horizontal=True
                        )
                        selected_yoy_dim_col = grouping_col_map.get(group_by_yoy)

                        if selected_yoy_dim_col and selected_yoy_dim_col in df_filtered_for_charts.columns:
                            temp_yoy_data = \
                            df_filtered_for_charts.groupby([pd.Grouper(key='Month', freq='YS'), selected_yoy_dim_col])[
                                'Amount in USD'].sum().reset_index()
                            temp_yoy_data = temp_yoy_data.sort_values('Month')
                            temp_yoy_data['YoY_Prev_Revenue'] = temp_yoy_data.groupby(selected_yoy_dim_col)[
                                'Amount in USD'].shift(1)
                            temp_yoy_data['YoY_Change'] = (temp_yoy_data['Amount in USD'] - temp_yoy_data[
                                'YoY_Prev_Revenue']) / temp_yoy_data['YoY_Prev_Revenue'] * 100
                            temp_yoy_data['YoY_Change'] = temp_yoy_data['YoY_Change'].replace(
                                [float('inf'), -float('inf')], np.nan).fillna(0)
                            yoy_plot_data = temp_yoy_data
                            yoy_plot_color_col = selected_yoy_dim_col
                        else:
                            overall_yoy_data = df_filtered_for_charts.groupby(pd.Grouper(key='Month', freq='YS'))[
                                'Amount in USD'].sum().reset_index()
                            overall_yoy_data = overall_yoy_data.sort_values('Month')
                            overall_yoy_data['YoY_Prev_Revenue'] = overall_yoy_data['Amount in USD'].shift(1)
                            overall_yoy_data['YoY_Change'] = (overall_yoy_data['Amount in USD'] - overall_yoy_data[
                                'YoY_Prev_Revenue']) / overall_yoy_data['YoY_Prev_Revenue'] * 100
                            overall_yoy_data['YoY_Change'] = overall_yoy_data['YoY_Change'].replace(
                                [float('inf'), -float('inf')], np.nan).fillna(0)
                            overall_yoy_data['Grouping_Value'] = 'Total Revenue'
                            yoy_plot_data = overall_yoy_data
                            yoy_plot_color_col = 'Grouping_Value'

                        if not yoy_plot_data.empty and yoy_plot_data.shape[0] > 1:
                            fig_yoy_rev = px.line(
                                yoy_plot_data,
                                x='Month',  # Use actual datetime for correct x-axis
                                y='Amount in USD',
                                color=yoy_plot_color_col,
                                title=f'YoY Revenue Trend by {group_by_yoy}',
                                labels={
                                    'Amount in USD': 'Revenue (USD)',
                                    'Month': 'Year'
                                }
                            )
                            fig_yoy_rev.update_traces(mode='lines+markers', hovertemplate=
                            '<b>%{x|%Y}</b><br>' +  # Format for year
                            'Revenue: %{y:,.2f}<br>' +
                            'YoY Change: %{customdata[0]:.2f}%<extra></extra>'
                                                      )
                            fig_yoy_rev.update_traces(customdata=yoy_plot_data[['YoY_Change']].values)
                            fig_yoy_rev.update_layout(xaxis_title="Year", yaxis_title="Revenue (USD)")
                            st.plotly_chart(fig_yoy_rev, use_container_width=True)

                            fig_yoy_pct = px.line(
                                yoy_plot_data,
                                x='Month',
                                y='YoY_Change',
                                color=yoy_plot_color_col,
                                title=f'YoY Revenue Percentage Change by {group_by_yoy}',
                                labels={
                                    'YoY_Change': 'YoY Change (%)',
                                    'Month': 'Year'
                                }
                            )
                            fig_yoy_pct.update_layout(xaxis_title="Year", yaxis_title="YoY Change (%)",
                                                      yaxis_tickformat=".2f%")
                            st.plotly_chart(fig_yoy_pct, use_container_width=True)

                            with st.expander("Show YoY Detailed Data"):
                                display_cols = ['Month', 'Amount in USD', 'YoY_Change']
                                if yoy_plot_color_col and yoy_plot_color_col in yoy_plot_data.columns and yoy_plot_color_col != 'Grouping_Value':
                                    display_cols.insert(1, yoy_plot_color_col)
                                st.dataframe(yoy_plot_data[display_cols].style.format({
                                    'Amount in USD': '$ {:,.2f}',
                                    'YoY_Change': '{:.2f}%'
                                }), use_container_width=True)
                        else:
                            st.info(
                                f"Not enough data points for Year-over-Year trend analysis grouped by {group_by_yoy}.")
                else:  # If Plotly is not available for Question 5 or unexpected output
                    st.warning(
                        "Plotly is not installed. Install with: pip install plotly to enable visualizations for trends.")
                    # You might want to display a simplified table here if charts aren't available
                    if isinstance(q5_output_dict, dict) and 'mom_data' in q5_output_dict:
                        st.write("Here's a preview of the Month-over-Month data:")
                        st.dataframe(q5_output_dict['mom_data'], use_container_width=True)

            else:
                st.warning("Sorry, an internal routing error occurred or the question type is not fully supported.")

        except Exception as e:
            st.error(f"An error occurred while processing your request: {e}")
            st.error("Please check your query or contact support if the issue persists.")


if __name__ == "__main__":
    main()