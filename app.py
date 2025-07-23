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
from logic.utils import get_cm_query_details, get_cost_drop_query_details, parse_query_filters, REVENUE_GROUPS, COST_GROUPS
from logic.question_1 import run_logic as run_question_1_logic
from logic.question_2 import run_logic as run_question_2_logic
from logic.question_3 import run_logic as run_question_3_logic

# Check for Plotly for visuals
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
        download_stream = blob_client.download_blob()
        csv_data = io.BytesIO(download_stream.readall())

        csv_data.seek(0)
        print(f"DEBUG: First 500 bytes of downloaded CSV data:\n{csv_data.read(500).decode('utf-8', errors='ignore')}")
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

# --- Helper function for dynamic number formatting ---
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

def main():
    st.set_page_config(layout="wide", page_title="CM Analyzer Pro", page_icon="📊")
    st.title("📊 Contribution Margin Analyzer")

    df = load_data_from_azure(AZURE_STORAGE_CONTAINER_NAME, AZURE_STORAGE_BLOB_NAME)
    if "Message" in df.columns and df["Message"].iloc[0].startswith("Azure Storage configuration error."):
        st.stop()
    elif "Message" in df.columns and df["Message"].iloc[0].startswith("Failed to load data from Azure Storage:"):
        st.stop()

    # --- STRONGER CSS: Fixes extra vertical spacing between h2 and columns/kpi cards ---
    st.markdown("""
    <style>
    /* Tighter spacing for h2 heading and horizontal block of metrics */
    h2, h2 + div {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    div[data-testid="stVerticalBlock"] > h2 {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stHorizontalBlock"] {
        margin-top: -150px !important;   /* Change this (more negative) to reduce gap further */
    }
    .kpi-container {
        padding:10px;
        margin-bottom:8px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: center;
        min-height: 25px;
    }
    .delta-green { color: #155724 !important; font-weight: bold !important; }
    .delta-red { color: #721c24 !important; font-weight: bold !important; }
    .delta-yellow { color: #856404 !important; font-weight: bold !important; }
    .custom-metric-label { font-size: 2.4 rem !important; color: #b0b3b8 !important; margin-bottom: 2px; }
    .custom-metric-value { font-size: 2 rem !important; font-weight: bold !important; color: #fff !important; margin-bottom: 2px; }
    .custom-metric-delta { font-size: 1,8 rem !important; }
    .stMetric { display: none !important; visibility: hidden !important; height: 0; padding: 0; margin: 0; }
    .stMetric > div { display: none !important; visibility: hidden !important; height: 0; padding: 0; margin: 0; }
    </style>
    """, unsafe_allow_html=True)

    user_query = st.text_input(
        "💬 Ask your question (e.g., 'List customers with CM > 90% last quarter', 'Which cost triggered the Margin drop last month in Transportation', ...)",
        placeholder="Enter your query here..."
    )

    if user_query:
        with st.spinner("Analyzing your question..."):
            question_id = get_question_id(user_input=user_query)
            parsed_filters = parse_query_filters(user_query, question_id)

        try:
            result_df = pd.DataFrame()
            if question_id == "unknown":
                st.warning("Sorry, I couldn't understand your question or it's not yet supported.")
                st.info("Try queries like: 'List customers with CM > 50%' ...")
            elif question_id == "question_1":
                result_df = run_question_1_logic(df, parsed_filters)
                if "Message" in result_df.columns:
                    st.warning(result_df["Message"].iloc[0])
                else:
                    cm_threshold = parsed_filters.get("lower", 0.0) * 100 if parsed_filters.get("type") in ["greater_than", "between"] and parsed_filters.get("lower") is not None else 0.0
                    if parsed_filters.get("date_filter") and parsed_filters.get("start_date") and parsed_filters.get("end_date"):
                        st.success(
                            f"📅 Date Filter: {parsed_filters['start_date'].strftime('%Y-%m-%d')} to {parsed_filters['end_date'].strftime('%Y-%m-%d')}")
                    else:
                        st.info("📅 Showing all available data (no specific date filter applied from query)")

                    if parsed_filters["type"] == "none":
                        st.info("🔍 Showing all Contribution Margins (no specific CM filter applied from query)")
                    elif parsed_filters["type"] == "greater_than":
                        st.write(f"🔍 Applying CM Filter: CM > {parsed_filters['lower'] * 100:.2f}%")
                    elif parsed_filters["type"] == "less_than":
                        st.write(f"🔍 Applying CM Filter: CM < {parsed_filters['lower'] * 100:.2f}%")
                    elif parsed_filters["type"] == "between":
                        st.write(f"🔍 Applying CM Filter: CM between {parsed_filters['lower'] * 100:.2f}% and {parsed_filters['upper'] * 100:.2f}%")

                    st.markdown("## 📊 Key Metrics", unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    total_customers = len(result_df)
                    total_rev = result_df['Revenue'].str.replace(r'[$,]', '', regex=True).astype(float).sum()
                    total_cost = result_df['Cost'].str.replace(r'[$,]', '', regex=True).astype(float).sum()
                    avg_cm = result_df['CM_Value'].replace([np.inf, -np.inf], np.nan).mean()
                    with col1:
                        st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="custom-metric-label">Total Customers</div>
                            <div class="custom-metric-value">{total_customers}</div>
                        """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="custom-metric-label">Total Revenue</div>
                            <div class="custom-metric-value">{format_currency_dynamic(total_rev)}</div>
                        """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="custom-metric-label">Total Cost</div>
                            <div class="custom-metric-value">{format_currency_dynamic(total_cost)}</div>
                        """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col4:
                        st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="custom-metric-label">Average CM</div>
                            <div class="custom-metric-value">{f"{avg_cm:.2f}%" if pd.notna(avg_cm) else "N/A"}</div>
                        """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # ========== VISUALIZATIONS & DATA TABLE TABS ==========
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
                            except:
                                return ''

                        display_df = result_df.drop(columns=["CM_Value"])
                        styled_display_df = display_df.style.applymap(color_cm, subset=['CM (%)']).set_table_styles([
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
                            st.subheader("Monthly Unique Customer Count")
                            trend_data = []
                            all_unique_customers = set()
                            trend_start_date = parsed_filters["start_date"] if parsed_filters.get("date_filter") and parsed_filters.get("start_date") else df["Month"].min().date()
                            trend_end_date = parsed_filters["end_date"] if parsed_filters.get("date_filter") and parsed_filters.get("end_date") else df["Month"].max().date()
                            current_month_iter = datetime(trend_start_date.year, trend_start_date.month, 1)
                            while current_month_iter.date() <= trend_end_date:
                                next_month_iter = (current_month_iter.replace(day=28) + pd.Timedelta(days=4)).replace(day=1)
                                monthly_df = df[
                                    (df["Month"].dt.date >= current_month_iter.date()) &
                                    (df["Month"].dt.date < next_month_iter.date())
                                    ].copy()
                                monthly_grouped = monthly_df.groupby("FinalCustomerName", as_index=False).apply(
                                    lambda x: pd.Series({
                                        "Revenue": x[x["Group1"].isin(REVENUE_GROUPS)]["Amount in USD"].sum(),
                                        "Cost": x[x["Group1"].isin(COST_GROUPS)]["Amount in USD"].sum()
                                    }), include_groups=False)
                                revenue_abs = monthly_grouped["Revenue"].abs()
                                monthly_grouped["CM_Ratio"] = (monthly_grouped["Revenue"] - monthly_grouped["Cost"]) / revenue_abs.replace(0, float('nan'))
                                monthly_grouped = monthly_grouped.dropna(subset=["CM_Ratio"])

                                filtered_monthly_customers = pd.DataFrame()
                                if parsed_filters["type"] == "less_than" and parsed_filters["lower"] is not None:
                                    filtered_monthly_customers = monthly_grouped[monthly_grouped["CM_Ratio"] < parsed_filters["lower"]]
                                elif parsed_filters["type"] == "greater_than" and parsed_filters["lower"] is not None:
                                    filtered_monthly_customers = monthly_grouped[monthly_grouped["CM_Ratio"] > parsed_filters["lower"]]
                                elif parsed_filters["type"] == "between":
                                    filtered_monthly_customers = monthly_grouped[(monthly_grouped["CM_Ratio"] >= parsed_filters["lower"]) & (monthly_grouped["CM_Ratio"] <= parsed_filters["upper"])]
                                else:
                                    filtered_monthly_customers = monthly_grouped

                                unique_customers_this_month = set(filtered_monthly_customers.index)
                                all_unique_customers.update(unique_customers_this_month)
                                trend_data.append({
                                    "Month": current_month_iter.strftime("%b-%Y"),
                                    "Unique Customers": len(unique_customers_this_month)
                                })
                                current_month_iter = next_month_iter

                            st.caption(f"✅ Total unique customers across all months: {len(all_unique_customers)}")
                            trend_df = pd.DataFrame(trend_data)
                            if not trend_df.empty:
                                fig = px.line(
                                    trend_df,
                                    x="Month",
                                    y="Unique Customers",
                                    title=f"Unique Customers (CM {parsed_filters['type']} {parsed_filters.get('lower', '') if parsed_filters.get('type') != 'none' else ''})",
                                    markers=True,
                                    text="Unique Customers"
                                )
                                fig.update_traces(textposition="top center")
                                fig.update_layout(
                                    xaxis_title="Month",
                                    yaxis_title="Unique Customers",
                                    hovermode="x unified"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("No trend data available for the selected period and CM criteria.")

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

                        else:
                            st.warning("Plotly is not installed. Install with: pip install plotly to enable visualizations.")
                            st.write("Here's a preview of the raw data:")
                            st.dataframe(
                                result_df.nlargest(10, 'CM_Value')[['S.No', 'FinalCustomerName', 'CM (%)', 'Revenue', 'Cost']],
                                height=400
                            )
            elif question_id == "question_2":
                st.subheader("📉 Cost Drop Analysis")
                result_df = run_question_2_logic(df, parsed_filters)
                if "Message" in result_df.columns:
                    st.warning(result_df["Message"].iloc[0])
                else:
                    month_of_interest_start = parsed_filters.get('month_of_interest_start')
                    current_month_str = month_of_interest_start.strftime('%b %Y') if month_of_interest_start else 'N/A'
                    prev_month_str = 'N/A'
                    if month_of_interest_start:
                        try:
                            prev_month_date = month_of_interest_start.replace(day=1) - timedelta(days=1)
                            prev_month_str = prev_month_date.strftime('%b %Y')
                        except Exception as e:
                            prev_month_str = 'N/A'

                    st.success(f"Analysis for Segment: '{parsed_filters.get('segment', 'All')}' for {current_month_str} vs {prev_month_str}")
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
                    p1_start_date = parsed_filters.get("period1_start")
                    p1_end_date = parsed_filters.get("period1_end")
                    p2_start_date = parsed_filters.get("period2_start")
                    p2_end_date = parsed_filters.get("period2_end")

                    if comparison_type == "quarter":
                        st.info(
                            f"Comparing C&B costs for **{p1_name_display} ({p1_start_date.strftime('%b %Y')}-{p1_end_date.strftime('%b %Y')})** vs **{p2_name_display} ({p2_start_date.strftime('%b %Y')}-{p2_end_date.strftime('%b %Y')})**")
                    else:
                        st.info(
                            f"Comparing C&B costs for **{p1_name_display} ({p1_start_date.strftime('%b %Y')})** vs **{p2_name_display} ({p2_start_date.strftime('%b %Y')})**")

                    p1_cost_numeric = result_df["Period1_Cost_Numeric"].iloc[0] if "Period1_Cost_Numeric" in result_df.columns else None
                    p2_cost_numeric = result_df["Period2_Cost_Numeric"].iloc[0] if "Period2_Cost_Numeric" in result_df.columns else None
                    absolute_variation = p2_cost_numeric - p1_cost_numeric if pd.notna(p1_cost_numeric) and pd.notna(p2_cost_numeric) else None
                    percentage_variation = result_df["Percentage_Variation"].iloc[0] if "Percentage_Variation" in result_df.columns else None

                    st.markdown("## Key Metrics", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="custom-metric-label">C&B Cost {p1_name_display}</div>
                            <div class="custom-metric-value">{format_currency_dynamic(p1_cost_numeric)}</div>
                        """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="custom-metric-label">C&B Cost {p2_name_display}</div>
                            <div class="custom-metric-value">{format_currency_dynamic(p2_cost_numeric)}</div>
                        """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
                        if pd.notna(absolute_variation):
                            variation_text = format_currency_dynamic(absolute_variation)
                            percentage_text = ""
                            color_class = ""
                            if pd.notna(percentage_variation):
                                percentage_text = f"{percentage_variation:+.2f}%"
                                # Corrected coloring logic for cost variation:
                                # Cost INCREASE (positive variation) is generally bad, so RED.
                                # Cost DECREASE (negative variation) is generally good, so GREEN.
                                # Zero/N/A is YELLOW.
                                if percentage_variation > 0: # Cost increased
                                    color_class = "delta-green"
                                elif percentage_variation < 0: # Cost decreased
                                    color_class = "delta-red"
                                else: # No change
                                    color_class = "delta-yellow"
                            st.markdown(f"""
                                <div class="custom-metric-label">Variation</div>
                                <div class="custom-metric-value">{variation_text}</div>
                                <div class="custom-metric-delta {color_class}">{percentage_text}</div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class="custom-metric-label">Variation</div>
                                <div class="custom-metric-value">N/A</div>
                                <div class="custom-metric-delta delta-yellow">N/A</div>
                            """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
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
                        st.warning("Plotly is not installed. Install with: pip install plotly to enable visualizations.")
                    else:
                        st.info("Not enough data to generate a comparison chart.")
        except Exception as e:
            st.error(f"An error occurred while processing your request: {e}")
            st.error("Please check your query or contact support if the issue persists.")

if __name__ == "__main__":
    main()
