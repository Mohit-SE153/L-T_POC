import streamlit as st
import pandas as pd
import os
import pytz
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta  # Ensure timedelta is also imported for date calculations

# Import the question routing function
from questions import get_question_id

# Import the utility functions and the specific question logic
from logic.utils import get_cm_query_details, get_cost_drop_query_details, parse_query_filters, REVENUE_GROUPS, \
    COST_GROUPS
from logic.question_1 import run_logic as run_question_1_logic  # Alias for question_1 logic
from logic.question_2 import run_logic as run_question_2_logic  # Alias for question_2 logic

# Check for Plotly availability for visualizations
try:
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Visualizations disabled - Plotly not installed. Install with: pip install plotly")

# --- CONFIGURATION ---
# IMPORTANT: Update this path to your actual Excel file location
LOCAL_FILE_PATH = r"D:\L&T POC\OPS MIS_BRD 3_V1.1.xlsx"
SHEET_NAME = "P&L"


# --- DATA LOADING ---
@st.cache_data
def load_data(path, sheet_name):
    """
    Loads data from the specified Excel file and performs initial cleaning and formatting.
    Caches the data for performance.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.drop_duplicates()
    df["Amount in INR"] = df["Amount in INR"].round(2)
    df["Amount in USD"] = df["Amount in USD"].round(2)
    # Ensure 'Month' is datetime. The format infer_datetime_format=True helps.
    df["Month"] = pd.to_datetime(df["Month"], errors='coerce', dayfirst=True, infer_datetime_format=True)
    return df


# --- MAIN STREAMLIT APP LAYOUT ---
def main():
    st.set_page_config(layout="wide", page_title="CM Analyzer Pro", page_icon="📊")
    st.title("📊 Contribution Margin Analyzer")

    # Load the data
    df = load_data(LOCAL_FILE_PATH, SHEET_NAME)

    # Custom CSS for styling (copied from original L&T POC.py)
    st.markdown("""
    <style>
        .metric-card {
            padding: 15px;
            border-radius: 10px;
            background-color: #f0f2f6;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        .positive-high {
            background-color: #d4edda !important;
            color: #155724 !important;
        }
        .positive-med {
            background-color: #fff3cd !important;
            color: #856404 !important;
        }
        .negative {
            background-color: #f8d7da !important;
            color: #721c24 !important;
        }
        .table-header {
            font-weight: bold !important;
            background-color: #f0f2f6 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # User input text box
    user_query = st.text_input(
        "💬 Ask your question (e.g., 'List customers with CM > 90% last quarter', 'Which cost triggered the Margin drop last month in Transportation')",
        placeholder="Enter your query here..."
    )

    if user_query:
        with st.spinner("Analyzing your question..."):
            # Step 1: Route the question to the appropriate logic
            question_id = get_question_id(user_query)

            # Step 2: Parse filters based on the identified question_id
            parsed_filters = parse_query_filters(user_query, question_id)

        try:
            if question_id == "unknown":
                st.warning("Sorry, I couldn't understand your question or it's not yet supported.")
                st.info(
                    "Try queries like: 'List customers with CM > 50%', 'Show CM for last month', 'What is CM between 10% and 30% in Q1 2023?', or 'Which cost triggered the Margin drop last month in Transportation'.")
            elif question_id == "question_1":  # Handles all CM related queries
                result_df = run_question_1_logic(df, parsed_filters)

                if "Message" in result_df.columns:
                    st.warning(result_df["Message"].iloc[0])
                else:
                    # Determine the CM threshold for display and conditional formatting
                    cm_threshold = parsed_filters.get("lower", 0.0) * 100 if parsed_filters.get("type") in [
                        "greater_than", "between"] and parsed_filters.get("lower") is not None else 0.0

                    # Display date filter information
                    if parsed_filters.get("date_filter") and parsed_filters.get("start_date") and parsed_filters.get(
                            "end_date"):
                        st.success(
                            f"📅 Date Filter: {parsed_filters['start_date'].strftime('%Y-%m-%d')} to {parsed_filters['end_date'].strftime('%Y-%m-%d')}")
                    else:
                        st.info("📅 Showing all available data (no specific date filter applied from query)")

                    # Display CM filter information
                    if parsed_filters["type"] == "none":
                        st.info("🔍 Showing all Contribution Margins (no specific CM filter applied from query)")
                    elif parsed_filters["type"] == "greater_than":
                        st.write(f"🔍 Applying CM Filter: CM > {parsed_filters['lower'] * 100:.2f}%")
                    elif parsed_filters["type"] == "less_than":
                        st.write(f"🔍 Applying CM Filter: CM < {parsed_filters['lower'] * 100:.2f}%")
                    elif parsed_filters["type"] == "between":
                        st.write(
                            f"🔍 Applying CM Filter: CM between {parsed_filters['lower'] * 100:.2f}% and {parsed_filters['upper'] * 100:.2f}%")

                    # ========== METRIC CARDS ==========
                    st.subheader("📊 Key Metrics")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Total Customers", len(result_df))
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col2:
                        avg_cm = result_df['CM_Value'].replace([np.inf, -np.inf], np.nan).mean()
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Average CM", f"{avg_cm:.2f}%" if pd.notna(avg_cm) else "N/A")
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col3:
                        total_rev = result_df['Revenue'].str.replace(r'[$,]', '', regex=True).astype(float).sum()
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Total Revenue", f"${total_rev:,.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # ========== VISUALIZATIONS & DATA TABLE TABS ==========
                    tab1, tab2 = st.tabs(["📋 Data Table", "📈 Visual Analysis"])

                    with tab1:
                        # ========== CONDITIONAL FORMATTING FOR TABLE ==========
                        def color_cm(val):
                            """Applies conditional formatting based on CM value."""
                            try:
                                cm_val = float(val.replace('%', ''))
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

                        styled_display_df = display_df.style \
                            .applymap(color_cm, subset=['CM (%)']) \
                            .set_table_styles([
                            {'selector': 'th', 'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]}
                        ])

                        st.dataframe(
                            styled_display_df,
                            height=600,
                            use_container_width=True
                        )

                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name='cm_analysis.csv',
                            mime='text/csv'
                        )

                    with tab2:
                        if PLOTLY_AVAILABLE:
                            # Monthly Unique Customer Trend
                            st.subheader("Monthly Unique Customer Count")
                            trend_data = []
                            all_unique_customers = set()

                            trend_start_date = parsed_filters["start_date"] if parsed_filters.get(
                                "date_filter") and parsed_filters.get("start_date") else df["Month"].min().date()
                            trend_end_date = parsed_filters["end_date"] if parsed_filters.get(
                                "date_filter") and parsed_filters.get("end_date") else df["Month"].max().date()

                            current_month_iter = datetime(trend_start_date.year, trend_start_date.month, 1)
                            while current_month_iter.date() <= trend_end_date:
                                next_month_iter = (current_month_iter.replace(day=28) + pd.Timedelta(days=4)).replace(
                                    day=1)

                                monthly_df = df[
                                    (df["Month"].dt.date >= current_month_iter.date()) &
                                    (df["Month"].dt.date < next_month_iter.date())
                                    ]

                                monthly_grouped = monthly_df.groupby("FinalCustomerName").apply(lambda x: pd.Series({
                                    "Revenue": x[x["Group1"].isin(REVENUE_GROUPS)]["Amount in USD"].sum(),
                                    "Cost": x[x["Group1"].isin(COST_GROUPS)]["Amount in USD"].sum()
                                }))

                                revenue_abs = monthly_grouped["Revenue"].abs()
                                monthly_grouped["CM_Ratio"] = (monthly_grouped["Revenue"] - monthly_grouped[
                                    "Cost"]) / revenue_abs.replace(0, float('nan'))
                                monthly_grouped = monthly_grouped.dropna(subset=["CM_Ratio"])

                                filtered_monthly_customers = pd.DataFrame()
                                if parsed_filters["type"] == "less_than" and parsed_filters["lower"] is not None:
                                    filtered_monthly_customers = monthly_grouped[
                                        monthly_grouped["CM_Ratio"] < parsed_filters["lower"]]
                                elif parsed_filters["type"] == "greater_than" and parsed_filters["lower"] is not None:
                                    filtered_monthly_customers = monthly_grouped[
                                        monthly_grouped["CM_Ratio"] > parsed_filters["lower"]]
                                elif parsed_filters["type"] == "between":
                                    filtered_monthly_customers = monthly_grouped[
                                        (monthly_grouped["CM_Ratio"] >= parsed_filters["lower"]) &
                                        (monthly_grouped["CM_Ratio"] <= parsed_filters["upper"])
                                        ]
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

                            # CM Distribution Histogram
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

                            # Top Performers Bar Chart
                            st.subheader(f"Top Performers (CM > {cm_threshold:.0f}%)")

                            top_performers = result_df[result_df['CM_Value'] > cm_threshold]

                            if not top_performers.empty:
                                fig_top = px.bar(
                                    top_performers.nlargest(10, 'CM_Value'),
                                    x="FinalCustomerName",
                                    y="CM_Value",
                                    text="CM (%)",
                                    title=f"Top Performing Customers (CM > {cm_threshold:.0f}%)",
                                    color="CM_Value",
                                    color_continuous_scale='greens'
                                )
                                fig_top.update_layout(
                                    xaxis_title="Customer Name",
                                    yaxis_title="Contribution Margin (%)",
                                    xaxis={'categoryorder': 'total descending'},
                                    coloraxis_showscale=False
                                )
                                fig_top.update_traces(texttemplate='%{text}', textposition='outside')
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
            elif question_id == "question_2":  # New block for question_2
                st.subheader("📉 Cost Drop Analysis")
                # Call the logic for question_2
                result_df = run_question_2_logic(df, parsed_filters)

                if "Message" in result_df.columns:
                    st.warning(result_df["Message"].iloc[0])
                else:
                    # Retrieve month_of_interest_start from parsed_filters
                    month_of_interest_start = parsed_filters.get('month_of_interest_start')

                    # Safely format the date strings for display
                    current_month_str = month_of_interest_start.strftime('%b %Y') if month_of_interest_start else 'N/A'

                    # Calculate previous month's date string safely
                    prev_month_str = 'N/A'
                    if month_of_interest_start:
                        try:
                            # Go to the first day of the month of interest, then subtract one day to get to the end of the previous month
                            prev_month_date = month_of_interest_start.replace(day=1) - timedelta(days=1)
                            prev_month_str = prev_month_date.strftime('%b %Y')
                        except Exception as e:
                            print(f"Error calculating previous month for display: {e}")
                            prev_month_str = 'N/A'  # Fallback if calculation fails

                    st.success(f"Analysis for Segment: '{parsed_filters.get('segment', 'All')}' for "
                               f"{current_month_str} vs {prev_month_str}")
                    st.write("Costs that increased (potentially triggered margin drop):")
                    st.dataframe(result_df, use_container_width=True)

            else:  # For future question_3, etc.
                st.warning(
                    f"Question type '{question_id}' is recognized but its logic is not yet fully implemented or displayed in the app.")
                st.info("Please try a CM-related query for now.")

        except Exception as e:
            st.error(f"An error occurred while processing your request: {e}")
            st.error("Please check your query or contact support if the issue persists.")


if __name__ == "__main__":
    main()

