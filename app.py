# In app.py

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import pandas as pd
import os
import pytz
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
import io

from questions import get_question_id
from logic.utils import get_cm_query_details, get_cost_drop_query_details, parse_query_filters, REVENUE_GROUPS, \
    COST_GROUPS, CB_GROUPS
from logic.question_1 import run_logic as run_question_1_logic
from logic.question_2 import run_logic as run_question_2_logic
from logic.question_3 import run_logic as run_question_3_logic
from logic.question_4 import run_logic as run_question_4_logic

try:
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Visualizations disabled - Plotly not installed.")

AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
AZURE_STORAGE_BLOB_NAME = os.getenv("AZURE_STORAGE_BLOB_NAME")
AZURE_STORAGE_CONNECTION_STRING = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={AZURE_STORAGE_ACCOUNT_NAME};"
    f"AccountKey={AZURE_STORAGE_ACCOUNT_KEY};"
    f"EndpointSuffix=core.windows.net"
)


@st.cache_data
def load_data_from_azure(container_name, blob_name):
    if not AZURE_STORAGE_ACCOUNT_NAME or not AZURE_STORAGE_ACCOUNT_KEY or not container_name or not blob_name:
        st.error("Azure Storage credentials or blob details are missing. Please set environment variables.")
        return pd.DataFrame({"Message": ["Azure Storage configuration error."]})
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        csv_data = io.BytesIO(blob_client.download_blob().readall())
        csv_data.seek(0)
        df = pd.read_csv(
            csv_data, delimiter=',',
            dtype={'wbs id': str, 'Pulse Project ID': str, 'Contract ID': str, 'sales Region': str},
            low_memory=False
        )
        df = df.drop_duplicates()
        df["Amount in INR"] = df["Amount in INR"].round(2)
        df["Amount in USD"] = df["Amount in USD"].round(2)
        df["Month"] = pd.to_datetime(df["Month"], errors='coerce', dayfirst=True)
        df.dropna(subset=['Month'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data from Azure Storage: {e}")
        st.info("Please check your Azure Storage settings. Ensure the blob is a valid CSV file.")
        return pd.DataFrame({"Message": [f"Failed to load data from Azure Storage: {e}"]})


def format_currency_dynamic(value):
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
    with col:
        st.markdown(f"""
            <div class="kpi-label-top">{label}</div>
            <div class="kpi-value-bottom">{value}</div>
            {f'<div class="kpi-delta-bottom {delta_color_class}">{delta_text}</div>' if delta_text else ''}
        """, unsafe_allow_html=True)


def main():
    st.set_page_config(layout="wide", page_title="CM Analyzer Pro", page_icon="📊")
    st.title("📊 L&T DataStream Financials")

    df = load_data_from_azure(AZURE_STORAGE_CONTAINER_NAME, AZURE_STORAGE_BLOB_NAME)
    if "Message" in df.columns and df["Message"].iloc[0].startswith("Azure Storage configuration error."):
        st.stop()
    elif "Message" in df.columns and df["Message"].iloc[0].startswith("Failed to load data from Azure Storage:"):
        st.stop()

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
    /* This ensures it aligns with other content on the left */
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
        /* Aligning the start of this block with other content */
        justify-content: flex-start; /* Align items to the start of the container */
    }

    /* KPI Styling - No background, label on top, value on bottom */
    /* Ensure these elements align left within their columns */

    .kpi-label-top {
        font-size: 1rem !important;
        color: #b0b3b8 !important;
        margin-bottom: 0.2rem;
        text-align: left;
        width: 100%; /* Ensure it takes full width of its column for alignment */
    }

    .kpi-value-bottom {
        font-size: 1.8rem !important;
        font-weight: bold !important;
        color: #fff !important;
        margin-bottom: 0.2rem;
        text-align: left;
        width: 100%; /* Ensure it takes full width of its column for alignment */
    }

    .kpi-delta-bottom {
        font-size: 0.9rem !important;
        text-align: left;
        width: 100%; /* Ensure it takes full width of its column for alignment */
    }

    /* Delta colors (can be customized further) */
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
        "💬 Ask your question (e.g., 'List customers with CM > 90% last quarter', 'Which cost triggered the Margin drop last month in Transportation', 'What is M-o-M trend of C&B cost % w.r.t total revenue', ...)",
        placeholder="Enter your query here..."
    )

    if user_query:
        with st.spinner("Analyzing your question..."):
            question_id = get_question_id(user_input=user_query)
            parsed_filters = parse_query_filters(user_query, question_id)

        try:
            result_df = pd.DataFrame()
            if "Message" in parsed_filters and parsed_filters["Message"].startswith("API keys for Azure OpenAI are not configured"):
                st.error(parsed_filters["Message"])
                st.stop()
            if question_id == "unknown":
                st.warning("Sorry, I couldn't understand your question or it's not yet supported.")
                st.info("Try queries like: 'List customers with CM > 50%' or 'What is M-o-M trend of C&B cost % w.r.t total revenue'...")
            elif question_id == "question_1":
                result_df = run_question_1_logic(df, parsed_filters)
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
                            hist_df['CM_Value'] = hist_df['CM_Value'].clip(-100, 200)
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
                filters = get_cost_drop_query_details(user_query)

                if filters.get("Message"):
                    st.warning(filters["Message"])
                    return

                result_df = run_question_2_logic(df, filters)

                if "Message" in result_df.columns:
                    st.warning(result_df["Message"].iloc[0])
                else:
                    current_month_display_name = filters.get("month_of_interest_name", "N/A Month")
                    compare_to_month_display_name = filters.get("compare_to_month_name", "N/A Month")

                    st.success(
                        f"Analysis for Segment: '{filters.get('segment', 'All')}' for {current_month_display_name} vs {compare_to_month_display_name}")

                    st.write("Costs that increased (potentially triggered margin drop):")
                    st.dataframe(result_df, use_container_width=True)
            elif question_id == "question_3":
                st.subheader("💰 C&B Cost Variation Analysis")
                result_df = run_question_3_logic(df, parsed_filters)
                if "Message" in result_df.columns:
                    st.warning(result_df["Message"].iloc[0])
                else:
                    comparison_type = result_df["Comparison_Type"].iloc[0]

                    p1_name_display = parsed_filters.get("period1_name", "Period 1")
                    p2_name_display = parsed_filters.get("period2_name", "Period 2")

                    if comparison_type == "quarter":
                        p1_start_date = parsed_filters.get("period1_start")
                        p1_end_date = parsed_filters.get("period1_end")
                        p2_start_date = parsed_filters.get("period2_start")
                        p2_end_date = parsed_filters.get("period2_end")

                        if p1_start_date and p1_end_date:
                            p1_name_display = f"{p1_start_date.strftime('%b')}-{p1_end_date.strftime('%b %Y')}"
                        if p2_start_date and p2_end_date:
                            p2_name_display = f"{p2_start_date.strftime('%b')}-{p2_end_date.strftime('%b %Y')}"

                        st.info(
                            f"Comparing C&B costs for **{p1_name_display}** vs **{p2_name_display}**")
                    else:
                        p1_start_date = parsed_filters.get("period1_start")
                        p1_end_date = parsed_filters.get("period1_end")
                        p2_start_date = parsed_filters.get("period2_start")
                        p2_end_date = parsed_filters.get("period2_end")
                        st.info(
                            f"Comparing C&B costs for **{p1_name_display} ({p1_start_date.strftime('%b %Y')})** vs **{p2_name_display} ({p2_start_date.strftime('%b %Y')})**")

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
                        color_class = "delta-yellow"

                        if pd.notna(percentage_variation):
                            percentage_text = f"{percentage_variation:+.2f}%"
                            if percentage_variation > 0:
                                color_class = "delta-green"
                            elif percentage_variation < 0:
                                color_class = "delta-red"
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
                            yaxis_range=[0, max(p1_cost_numeric, p2_cost_numeric) * 1.15]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    elif not PLOTLY_AVAILABLE:
                        st.warning(
                            "Plotly is not installed. Install with: pip install plotly to enable visualizations.")
                    else:
                        st.info("Not enough data to generate a comparison chart.")
            elif question_id == "question_4":
                st.subheader("📊 C&B Cost % of Total Revenue Trend (M-o-M)")
                result_df = run_question_4_logic(df, parsed_filters)

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

            else:
                st.warning("Sorry, an internal routing error occurred or the question type is not fully supported.")

        except Exception as e:
            st.error(f"An error occurred while processing your request: {e}")
            st.error("Please check your query or contact support if the issue persists.")


if __name__ == "__main__":
    main()