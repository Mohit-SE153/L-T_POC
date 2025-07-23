import pandas as pd
import re
import numpy as np
from dateutil import parser
from datetime import datetime, timedelta
from typing import Tuple, Optional
import json
import pytz
from openai import AzureOpenAI
import os  # Ensure os is imported here too for getenv

# ---------- CONFIGURATION ----------
# IMPORTANT: Read API key and endpoint from environment variables for deployment.
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
# Get raw endpoint, then strip if it's not None
raw_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_ENDPOINT = raw_endpoint.strip() if raw_endpoint else None

AZURE_OPENAI_DEPLOYMENT = "gpt-35-turbo"
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"

# --- DEBUGGING PRINTS (for local testing) ---
print(
    f"DEBUG: logic/utils.py - AZURE_OPENAI_KEY (first 5 chars): {AZURE_OPENAI_KEY[:5] if AZURE_OPENAI_KEY else 'None'}")
print(f"DEBUG: logic/utils.py - AZURE_OPENAI_ENDPOINT: '{AZURE_OPENAI_ENDPOINT}'")
# --- END DEBUGGING PRINTS ---

# ---------- GLOBAL AZURE OPENAI CLIENT INITIALIZATION ----------
# Initialize client globally, only if environment variables are available
utils_openai_client = None
try:
    if AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT:
        utils_openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        # Optional: A small test call to verify connectivity
        # test_response = utils_openai_client.chat.completions.create(
        #     model=AZURE_OPENAI_DEPLOYMENT,
        #     messages=[{"role": "user", "content": "Test connection"}],
        #     max_tokens=5
        # )
        # print("DEBUG: utils.py - Azure OpenAI client initialized and tested successfully.")
    else:
        print("WARNING: utils.py - Azure OpenAI client not initialized due to missing environment variables.")
except Exception as e:
    print(f"CRITICAL ERROR: utils.py - Failed to initialize Azure OpenAI client: {e}")
    utils_openai_client = None  # Ensure client is None if initialization fails

# Column alias mapping
column_mapping = {
    "customer": "FinalCustomerName",
    "type": "Type",
    "project": "Pulse Project ID",
    "month": "Month",
    "region": "sales Region",
    "amount": "Amount in USD",
    "group1": "Group1",
    "group": "Group1",
    "segment": "Segment"
}


# ---------- DATE RANGE PARSING ----------
def parse_date_range_from_query_llm(query):
    """
    Uses LLM to extract precise date ranges from natural language queries.
    Returns a dictionary with date_filter, start_date, end_date, and description.
    """
    today = datetime.now(pytz.timezone("Asia/Kolkata")).date()

    system_prompt = f"""Today's date is {today}. You are an expert at extracting precise date ranges from natural language queries.
Return a JSON object with:
- 'date_filter': boolean indicating if a date filter was requested
- 'start_date': YYYY-MM-DD format (first day of period)
- 'end_date': YYYY-MM-DD format (last day of period)
- 'description': natural language description of the period

Rules:
1. For relative periods (like "last month", "previous month", "last quarter"), calculate exact dates based on today's date.
2. For absolute periods (like "January 2023"), use exact dates.
3. For ranges (like "from March to May 2023"), use exact start/end dates.
4. For quarters, use exact quarter boundaries (e.g., Q1 = Jan 1 to Mar 31).
5. If no date filter, set date_filter=false and return null for dates.
6. Ensure the JSON output is perfectly valid, with no trailing commas.
"""
    try:
        # Use the globally initialized client
        if utils_openai_client:
            response = utils_openai_client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0
            )
            llm_response_content = response.choices[0].message.content.strip()
            print(f"DEBUG: LLM response for date parsing: '{llm_response_content}'")  # Debug print for LLM response
        else:
            print("ERROR: utils.py - Azure OpenAI client is not available for parse_date_range_from_query_llm.")
            return {"date_filter": False, "start_date": None, "end_date": None, "description": "all available data"}

        # Safely extract and parse JSON using regex to handle potential extra text/trailing commas
        json_str_match = re.search(r"\{.*\}", llm_response_content, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group()
            result = json.loads(json_str)
        else:
            print(f"WARNING: No valid JSON found in LLM date parsing response: '{llm_response_content}'")
            return {"date_filter": False, "start_date": None, "end_date": None, "description": "all available data"}

        if result.get("date_filter"):
            try:
                # Ensure start_date and end_date are not None before parsing
                start_date = parser.parse(result["start_date"]).date() if result.get("start_date") else None
                end_date = parser.parse(result["end_date"]).date() if result.get("end_date") else None
                result["start_date"] = start_date
                result["end_date"] = end_date
                if not (start_date and end_date):
                    result["date_filter"] = False
                    print(f"WARNING: Date parsing resulted in invalid start/end dates for query: '{query}'")
            except Exception as parse_e:
                result["date_filter"] = False
                result["start_date"] = None
                result["end_date"] = None
                print(f"ERROR: Failed to parse dates from LLM response for query '{query}': {parse_e}")

        print(f"DEBUG: Final parsed date info: {result}")  # Debug print for final parsed dates
        return result
    except Exception as e:
        print(f"Error in parse_date_range_from_query_llm API call or parsing: {e}")
        return {"date_filter": False, "start_date": None, "end_date": None, "description": "all available data"}


# ---------- CM QUERY PARSER ----------
def parse_percentage(val):
    """Converts a value to a float, handling percentages."""
    try:
        if isinstance(val, str) and "%" in val:
            val = float(val.replace("%", "").strip()) / 100
        elif isinstance(val, str):
            val = float(val.strip())
            if val > 1:  # Assume if > 1, it's a percentage entered as a whole number (e.g., 30 instead of 0.3)
                val = val / 100
        elif isinstance(val, (int, float)) and val > 1:  # Same for numerical values
            val = val / 100
        return float(val)
    except:
        return None


def get_cm_query_details(prompt):
    """
    Uses LLM to extract CM filter details (less_than, greater_than, between)
    and integrates date parsing.
    """
    date_info = parse_date_range_from_query_llm(prompt)

    system_prompt = """You are a CM filter extraction assistant. From the user query, extract:
    - Filter type: 'less_than', 'greater_than', 'between', or 'none' if no CM filter is specified.
    - Lower bound (convert percentages to decimals: 30% -> 0.3)
    - Upper bound (if 'between' filter)

    Return ONLY valid JSON, nothing else.
    Example outputs:
    - "CM < 30%" -> {"type": "less_than", "lower": 0.3}
    - "CM > 20%" -> {"type": "greater_than", "lower": 0.2}
    - "between 10% and 15%" -> {"type": "between", "lower": 0.1, "upper": 0.15}
    - "Show me all customers" -> {"type": "none", "lower": null, "upper": null}
    """

    try:
        # Use the globally initialized client
        if utils_openai_client:
            response = utils_openai_client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            llm_response_content = response.choices[0].message.content.strip()
            print(f"DEBUG: LLM response for CM parsing: '{llm_response_content}'")  # Debug print for LLM response
        else:
            print("ERROR: utils.py - Azure OpenAI client is not available for get_cm_query_details.")
            return {
                "type": "none",
                "lower": None,
                "upper": None,
                **date_info
            }

        # Use regex to safely extract JSON from the response
        json_str_match = re.search(r"\{.*\}", llm_response_content, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group()
            result = json.loads(json_str)
        else:
            print(f"WARNING: No valid JSON found in LLM CM parsing response: '{llm_response_content}'")
            # Fallback if JSON is not found, assume no specific CM filter
            result = {"type": "none", "lower": None, "upper": None}

        result["lower"] = parse_percentage(result.get("lower"))
        result["upper"] = parse_percentage(result.get("upper"))
        result.update(date_info)  # Merge date info

        print(f"DEBUG: Final parsed CM details: {result}")  # Debug print for final parsed CM details
        return result

    except Exception as e:
        print(f"Failed to parse CM filters with LLM: {e}")
        # Default to showing all CMs if parsing fails, but respect date filter if present
        return {
            "type": "none",  # New type to indicate no specific CM filter
            "lower": None,
            "upper": None,
            **date_info
        }


# ---------- Parsing for Question 2 (Cost Drop Analysis) ----------
def get_cost_drop_query_details(prompt):
    """
    Uses LLM to extract segment and specific month for cost drop analysis.
    """
    today = datetime.now(pytz.timezone("Asia/Kolkata")).date()
    # Calculate last month's and previous month's dates for context
    last_month_end = today.replace(day=1) - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)

    prev_month_end = last_month_start - timedelta(days=1)
    prev_month_start = prev_month_end.replace(day=1)

    system_prompt = f"""Today's date is {today}.
You are an expert at extracting details for cost drop analysis queries.
Return a JSON object with:
- 'segment': The exact segment name as it appears in the data (e.g., 'Transportation', 'Media and technology', 'Healthcare'). If no specific segment is mentioned, return null.
- 'month_of_interest_start': YYYY-MM-DD format, the first day of the month the user is asking about (e.g., for "last month", this would be {last_month_start.strftime('%Y-%m-%d')}).
- 'month_of_interest_end': YYYY-MM-DD format, the last day of the month the user is asking about (e.g., for "last month", this would be {last_month_end.strftime('%Y-%m-%d')}).
- 'compare_to_previous_month': boolean, true if the query asks to compare to the previous month, false otherwise. (For this question type, it will mostly be true).

Rules for month_of_interest:
1. If "last month", calculate based on today's date.
2. If specific month/year (e.g., "July 2024"), use that month.
3. If "last quarter", use the last month of the last quarter.
4. If no specific month is mentioned but "last month" is implied by context (e.g., "cost drop in Transportation"), default to "last month".
5. Ensure the JSON output is perfectly valid, with no trailing commas.
"""
    try:
        # Use the globally initialized client
        if utils_openai_client:
            response = utils_openai_client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            llm_response_content = response.choices[0].message.content.strip()
            print(
                f"DEBUG: LLM response for cost drop parsing: '{llm_response_content}'")  # Debug print for LLM response
        else:
            print("ERROR: utils.py - Azure OpenAI client is not available for get_cost_drop_query_details.")
            return {
                "segment": None,
                "month_of_interest_start": last_month_start.strftime('%Y-%m-%d'),
                "month_of_interest_end": last_month_end.strftime('%Y-%m-%d'),
                "compare_to_previous_month": True
            }

        json_str_match = re.search(r"\{.*\}", llm_response_content, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group()
            result = json.loads(json_str)
        else:
            print(f"WARNING: No valid JSON found in LLM cost drop parsing response: '{llm_response_content}'")
            # Fallback to default values if JSON is not found
            result = {
                "segment": None,
                "month_of_interest_start": last_month_start.strftime('%Y-%m-%d'),
                "month_of_interest_end": last_month_end.strftime('%Y-%m-%d'),
                "compare_to_previous_month": True
            }

        # Convert date strings to datetime.date objects
        if result.get("month_of_interest_start"):
            result["month_of_interest_start"] = parser.parse(result["month_of_interest_start"]).date()
        if result.get("month_of_interest_end"):
            result["month_of_interest_end"] = parser.parse(result["month_of_interest_end"]).date()

        print(f"DEBUG: Final parsed cost drop details: {result}")  # Debug print for final parsed cost drop details
        return result
    except Exception as e:
        print(f"Error parsing cost drop query details with LLM: {e}")
        # Fallback to default for "last month" if parsing fails
        return {
            "segment": None,
            "month_of_interest_start": last_month_start,
            "month_of_interest_end": last_month_end,
            "compare_to_previous_month": True
        }


# ---------- General Filter Parsing Dispatcher ----------
def parse_query_filters(user_query: str, question_id: str) -> dict:
    """
    Dispatches to the appropriate filter parsing function based on question_id.
    """
    if question_id == "question_1":
        return get_cm_query_details(user_query)
    elif question_id == "question_2":
        return get_cost_drop_query_details(user_query)
    # Add more question types here as they are implemented
    else:
        # Default empty filters for unknown types or future types
        return {
            "date_filter": False,
            "start_date": None,
            "end_date": None,
            "type": "none",
            "lower": None,
            "upper": None,
            "segment": None,
            "month_of_interest_start": None,
            "month_of_interest_end": None,
            "compare_to_previous_month": False
        }


# CM Calculation and Filtering logic
REVENUE_GROUPS = ["ONSITE", "OFFSHORE", "INDIRECT REVENUE"]
COST_GROUPS = [
    "Direct Expense", "OWN OVERHEADS", "Indirect Expense",
    "Project Level Depreciation", "Direct Expense - DU Block Seats Allocation",
    "Direct Expense - DU Pool Allocation", "Establishment Expenses"
]


def calculate_cm(df: pd.DataFrame, cm_filters: dict) -> pd.DataFrame:
    """
    Calculates Contribution Margin (CM) per customer and applies CM filters.
    """
    # Group by customer
    # Added include_groups=False to silence FutureWarning
    grouped = df.groupby("FinalCustomerName", as_index=False).apply(lambda x: pd.Series({
        "Revenue": x[x["Group1"].isin(REVENUE_GROUPS)]["Amount in USD"].sum(),
        "Cost": x[x["Group1"].isin(COST_GROUPS)]["Amount in USD"].sum()
    }), include_groups=False)

    # Calculate CM ratio
    revenue_abs = grouped["Revenue"].abs()
    grouped["CM_Ratio"] = (grouped["Revenue"] - grouped["Cost"]) / revenue_abs.replace(0, float('nan'))
    grouped["CM_Ratio"] = grouped["CM_Ratio"].replace([float('inf'), -float('inf')], float('nan'))

    # Drop rows where CM_Ratio is NaN
    grouped = grouped.dropna(subset=["CM_Ratio"])

    # Apply CM filter
    # This is the corrected filtering logic
    filtered = grouped.copy()  # Start with a copy to apply filters iteratively
    if cm_filters["type"] == "less_than" and cm_filters["lower"] is not None:
        filtered = filtered[filtered["CM_Ratio"] < cm_filters["lower"]]
    elif cm_filters["type"] == "greater_than" and cm_filters["lower"] is not None:
        filtered = filtered[filtered["CM_Ratio"] > cm_filters["lower"]]
    elif cm_filters["type"] == "between":
        if cm_filters["lower"] is not None and cm_filters["upper"] is not None:
            filtered = filtered[
                (filtered["CM_Ratio"] >= cm_filters["lower"]) &
                (filtered["CM_Ratio"] <= cm_filters["upper"])
                ]
        else:  # If 'between' but bounds are missing, treat as 'none'
            filtered = grouped  # No filter applied
    # If type is 'none', no CM filter is applied, so 'filtered' remains 'grouped'

    # Format results for display
    # Sort order depends on filter type for better display
    # Only sort if there's a specific CM filter type, otherwise default sort
    if cm_filters["type"] != "none":
        filtered = filtered.sort_values(
            by="CM_Ratio",
            ascending=(cm_filters["type"] != "greater_than")
            # Ascending for less_than and between, Descending for greater_than
        ).reset_index(drop=True)
    else:
        # Default sort if no specific CM filter
        filtered = filtered.sort_values(by="CM_Ratio", ascending=False).reset_index(drop=True)

    filtered["CM (%)"] = filtered["CM_Ratio"].apply(
        lambda x: "N/A" if pd.isna(x) else f"{x * 100:.2f}%"
    )
    filtered["Revenue"] = filtered["Revenue"].apply(lambda x: f"${x:,.2f}")
    filtered["Cost"] = filtered["Cost"].apply(lambda x: f"${x:,.2f}")

    # Add CM_Value for visualizations (percentage as float)
    filtered["CM_Value"] = filtered["CM_Ratio"] * 100

    # Reset index with serial numbers
    filtered.index = filtered.index + 1
    filtered = filtered.rename_axis("S.No").reset_index()

    return filtered[["S.No", "FinalCustomerName", "Revenue", "Cost", "CM (%)", "CM_Value"]]

