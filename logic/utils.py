# In logic/utils.py

# ... (existing imports) ...
import pandas as pd
import re
import numpy as np # Ensure numpy is imported
from dateutil import parser
from datetime import datetime, timedelta
from typing import Tuple, Optional
import json
import pytz
import calendar
from openai import AzureOpenAI
import os

# ---------- CONFIGURATION ----------
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
raw_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_ENDPOINT = raw_endpoint.strip() if raw_endpoint else None

utils_openai_client = None
try:
    if AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT:
        utils_openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            api_version="2025-01-01-preview", # Use the specified API version
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
    else:
        print("WARNING: utils.py - Azure OpenAI client not initialized due to missing environment variables.")
except Exception as e:
    print(f"CRITICAL ERROR: utils.py - Failed to initialize Azure OpenAI client: {e}")
    utils_openai_client = None

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

# --- MISSING FUNCTION: parse_percentage ---
def parse_percentage(value) -> Optional[float]:
    """
    Parses a string or numeric value to a float percentage (decimal).
    Handles 'null', None, and values with '%' sign.
    """
    if value is None or (isinstance(value, str) and value.lower() == 'null'):
        return None
    try:
        if isinstance(value, str):
            value = value.strip()
            if value.endswith('%'):
                return float(value[:-1]) / 100.0
            else:
                return float(value) # Assume it's already a decimal if no %
        return float(value)
    except ValueError:
        return None
# --- END MISSING FUNCTION ---


# (Your existing parse_date_range_from_query_llm function)
# ... (It should remain as is) ...

def parse_date_range_from_query_llm(query):
    today = datetime.now(pytz.timezone("Asia/Kolkata")).date()

    system_prompt = f"""Today's date is {today}. You are an expert at extracting precise date ranges from natural language queries for financial data.
Follow these rules strictly:

**1. Fiscal Year (FY) Quarters and their Calendar Date Ranges:**
   (ONLY use these definitions if 'FY', 'fiscal year', or 'financial year' is explicitly mentioned in the query)
- **FY24 Q1 (April 2023 - June 2023):** Apr 1, 2023 to Jun 30, 2023
- **FY24 Q2 (July 2023 - Sep 2023):** Jul 1, 2023 to Sep 30, 2023
- **FY24 Q3 (Oct 2023 - Dec 2023):** Oct 1, 2023 to Dec 31, 2023
- **FY24 Q4 (Jan 2024 - Mar 2024):** Jan 1, 2024 to Mar 31, 2024
- **FY25 Q1 (April 2024 - June 2024):** Apr 1, 2024 to Jun 30, 2024
- **FY25 Q2 (July 2024 - Sep 2024):** Jul 1, 2024 to Sep 30, 2024
- **FY25 Q3 (Oct 2024 - Dec 2024):** Oct 1, 2024 to Dec 31, 2024
- **FY25 Q4 (Jan 2025 - Mar 2025):** Jan 1, 2025 to Mar 31, 2025
- **FY26 Q1 (April 2025 - June 2025):** Apr 1, 2025 to Jun 30, 2025 (This is the current fiscal quarter if today is in Apr-Jun 2025)
- **FY26 Q2 (July 2025 - Sep 2025):** Jul 1, 2025 to Sep 30, 2025
- **FY26 Q3 (Oct 2025 - Dec 2025):** Oct 1, 2025 to Dec 31, 2025
- **FY26 Q4 (Jan 2026 - Mar 2026):** Jan 1, 2026 to Mar 31, 2026

**2. Standard Calendar Year Quarters (CY Q) and their Calendar Date Ranges:**
   (Use these definitions if a year and 'Q'/'quarter' are mentioned WITHOUT 'FY', 'fiscal year', or 'financial year')
- **CY Q1:** Jan 1 to Mar 31
- **CY Q2:** Apr 1 to Jun 30
- **CY Q3:** Jul 1 to Sep 30
- **CY Q4:** Oct 1 to Dec 31

Return a JSON object with:
- 'date_filter': boolean indicating if a date filter was requested
- 'start_date': YYYY-MM-DD format (first day of period)
- 'end_date': YYYY-MM-DD format (last day of period)
- 'description': natural language description of the period

General Rules:
- **Prioritize explicit 'FY' prefix for fiscal quarters.**
- **For quarters mentioned without 'FY' (e.g., '2025 Q1', 'Q2 2024'), strictly use the Standard Calendar Year (CY) definitions.** If the year is ambiguous, assume the most relevant past or current calendar year.
- **For relative periods (e.g., "last month", "previous month", "last quarter"), calculate exact dates based on today's date.** "last quarter" or "this quarter" should refer to the *current fiscal quarter* based on the application's primary financial context, unless explicitly stated as "last *calendar* quarter".
- For absolute periods, handle both full month names (e.g., "January 2023") and short forms (e.g., "Jan 2023", "Apr 25").
- For ranges (like "from March to May 2023"), use exact start/end dates.
- If no date filter, set date_filter=false and return null for dates.
- Ensure the JSON output is perfectly valid, with no trailing commas.

Examples (Critical for LLM training):
- "List customers with CM > 90% in **fy 2026 q1**" -> {{"date_filter": true, "start_date": "2025-04-01", "end_date": "2025-06-30", "description": "FY26 Q1"}}
- "List customers with CM > 90% in **2026 q1**" -> {{"date_filter": true, "start_date": "2026-01-01", "end_date": "2026-03-31", "description": "2026 Q1 Calendar"}}
- "List customers with CM > 90% in **2025 q2**" -> {{"date_filter": true, "start_date": "2025-04-01", "end_date": "2025-06-30", "description": "2025 Q2 Calendar"}}
- "List customers with CM > 90% in **fy 2025 q1**" -> {{"date_filter": true, "start_date": "2024-04-01", "end_date": "2024-06-30", "description": "FY25 Q1"}}
- "last quarter" -> automatically calculate exact dates for last fiscal quarter.
- "2025 apr" or "April 2025" -> calculate exact dates for April 2025.
- "FY25 Q4" -> 2025-01-01 to 2025-03-31 (Fiscal Q4)
- "Q3 2024" -> 2024-07-01 to 2024-09-30 (Calendar Q3)
- "What is CM in 2025 quarter 1?" -> {{"date_filter": true, "start_date": "2025-01-01", "end_date": "2025-03-31", "description": "2025 Q1 Calendar"}}
"""
    try:
        if utils_openai_client:
            response = utils_openai_client.chat.completions.create(
                model="gpt-35-turbo", # Use the deployment name
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0
            )
            llm_response_content = response.choices[0].message.content.strip()
        else:
            return {"date_filter": False, "start_date": None, "end_date": None, "description": "all available data"}

        json_str_match = re.search(r"\{.*\}", llm_response_content, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group()
            result = json.loads(json_str)
        else:
            # Fallback if LLM doesn't return valid JSON, or if it indicates no date filter
            return {"date_filter": False, "start_date": None, "end_date": None, "description": "all available data"}

        if result.get("date_filter"):
            try:
                start_date = parser.parse(result["start_date"]).date() if result.get("start_date") else None
                end_date = parser.parse(result["end_date"]).date() if result.get("end_date") else None
                result["start_date"] = start_date
                result["end_date"] = end_date
                if not (start_date and end_date): # If dates couldn't be parsed or are incomplete
                    result["date_filter"] = False
            except Exception as parse_e:
                print(f"Error parsing date from LLM response: {parse_e}")
                result["date_filter"] = False
                result["start_date"] = None
                result["end_date"] = None

        return result
    except Exception as e:
        print(f"Error in LLM call for date parsing: {e}")
        return {"date_filter": False, "start_date": None, "end_date": None, "description": "all available data"}


def get_cm_query_details(prompt):
    date_info = parse_date_range_from_query_llm(prompt)

    system_prompt = """You are a CM filter extraction assistant. From the user query, extract:
    - Filter type: 'less_than', 'greater_than', 'between', 'equals', or 'none' if no CM filter is specified.
    - Lower bound (convert percentages to decimals: 30% -> 0.3, 215% -> 2.15)
    - Upper bound (if 'between' filter)

    Return ONLY valid JSON, nothing else.
    Example outputs:
    - "CM < 30%" -> {"type": "less_than", "lower": 0.3}
    - "CM > 20%" -> {"type": "greater_than", "lower": 0.2}
    - "between 10% and 15%" -> {"type": "between", "lower": 0.1, "upper": 0.15}
    - "CM = 90%" -> {"type": "equals", "lower": 0.9}
    - "CM = 215%" -> {"type": "equals", "lower": 2.15}
    - "Show me all customers" -> {"type": "none", "lower": null, "upper": null}
    """

    try:
        if utils_openai_client:
            response = utils_openai_client.chat.completions.create(
                model="gpt-35-turbo", # Use the deployment name
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            llm_response_content = response.choices[0].message.content.strip()
        else:
            return {
                "type": "none",
                "lower": None,
                "upper": None,
                **date_info # Ensure date info is still passed
            }

        json_str_match = re.search(r"\{.*\}", llm_response_content, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group()
            result = json.loads(json_str)
        else:
            # Fallback if LLM doesn't return valid JSON
            result = {"type": "none", "lower": None, "upper": None}

        result["lower"] = parse_percentage(result.get("lower"))
        result["upper"] = parse_percentage(result.get("upper"))

        if result.get("type") == "equals" and result.get("lower") is not None:
            result["upper"] = result["lower"]

        result.update(date_info) # Merge date info parsed previously
        return result

    except Exception as e:
        print(f"Error in LLM call for CM filter parsing: {e}")
        return {
            "type": "none",
            "lower": None,
            "upper": None,
            **date_info # Ensure date info is still passed on error
        }


def get_cost_drop_query_details(prompt):
    today = datetime.now(pytz.timezone("Asia/Kolkata")).date()

    # Calculate last month's dates for default/fallback
    last_month_end = today.replace(day=1) - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    last_month_name = last_month_start.strftime("%B %Y")

    # Calculate month prior to last month for default/fallback
    month_prior_to_last_end = last_month_start - timedelta(days=1)
    month_prior_to_last_start = month_prior_to_last_end.replace(day=1)
    month_prior_to_last_name = month_prior_to_last_start.strftime("%B %Y")

    system_prompt = f"""Today's date is {today.strftime('%Y-%m-%d')}.
You are an expert at extracting details for cost drop analysis queries, specifically identifying two months for comparison.
Return a JSON object with:
- 'segment': The exact segment name (e.g., 'Transportation', 'Media and technology', 'Healthcare'). If no specific segment, return null.
- 'month1_name': The natural language name of the FIRST month mentioned or implied (e.g., "April 2025", "last month", "current month"). If implied as "last month", use "{last_month_name}".
- 'month2_name': The natural language name of the SECOND month mentioned or implied for comparison (e.g., "March 2025", "previous month"). If implied as "previous month" relative to "last month", use "{month_prior_to_last_name}". If only one month is explicitly mentioned (e.g., "cost drop in July 2024"), infer the second month as the one chronologically *before* month1.
- 'month_of_interest_name': The name of the month that the user is primarily interested in seeing data for (the later month in the comparison).
- 'compare_to_month_name': The name of the month being compared against (the earlier month in the comparison).

Rules for determining 'month_of_interest_name' and 'compare_to_month_name':
- If the query is "from X to Y", then X is 'compare_to_month_name' and Y is 'month_of_interest_name'.
- If the query is "X compared to Y", then X is 'month_of_interest_name' and Y is 'compare_to_month_name'.
- If the query is "last month compared to previous month", then 'last month' is 'month_of_interest_name' and 'previous month' is 'compare_to_month_name'.
- If only one month is mentioned (e.g., "cost drop in July 2024"), then that month is 'month_of_interest_name', and the month immediately preceding it is 'compare_to_month_name'.
- If the query is vague (e.g., "cost drop in Transportation"), default to 'month1_name': "{last_month_name}" and 'month2_name': "{month_prior_to_last_name}". In this default case, '{last_month_name}' is 'month_of_interest_name' and '{month_prior_to_last_name}' is 'compare_to_month_name'.

Ensure the JSON output is perfectly valid, with no trailing commas.

Examples:
- "Which cost triggered the Margin drop from march 2025 to april 2025 in Transportation" -> {{"segment": "Transportation", "month1_name": "March 2025", "month2_name": "April 2025", "month_of_interest_name": "April 2025", "compare_to_month_name": "March 2025"}}
- "Which cost triggered the Margin drop last month in Transportation" -> {{"segment": "Transportation", "month1_name": "{last_month_name}", "month2_name": "{month_prior_to_last_name}", "month_of_interest_name": "{last_month_name}", "compare_to_month_name": "{month_prior_to_last_name}"}}
- "Show me cost increases in Healthcare for July 2024 compared to June" -> {{"segment": "Healthcare", "month1_name": "July 2024", "month2_name": "June 2024", "month_of_interest_name": "July 2024", "compare_to_month_name": "June 2024"}}
- "What expenses went up in Retail in August 2025" -> {{"segment": "Retail", "month1_name": "August 2025", "month2_name": "July 2025", "month_of_interest_name": "August 2025", "compare_to_month_name": "July 2025"}}
- "cost drop for Media and Technology" -> {{"segment": "Media and technology", "month1_name": "{last_month_name}", "month2_name": "{month_prior_to_last_name}", "month_of_interest_name": "{last_month_name}", "compare_to_month_name": "{month_prior_to_last_name}"}}
"""
    try:
        if utils_openai_client:
            response = utils_openai_client.chat.completions.create(
                model="gpt-35-turbo",  # Use the deployment name
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            llm_response_content = response.choices[0].message.content.strip()
        else:
            return {
                "segment": None,
                "month1_name": last_month_name,
                "month2_name": month_prior_to_last_name,
                "month_of_interest_name": last_month_name,
                "compare_to_month_name": month_prior_to_last_name,
                "month_of_interest_start": last_month_start,  # Keep for run_logic
                "month_of_interest_end": last_month_end,
                "compare_to_month_start": month_prior_to_last_start,  # Keep for run_logic
                "compare_to_month_end": month_prior_to_last_end,
                "compare_to_previous_month": True  # This can largely be inferred/removed now
            }

        json_str_match = re.search(r"\{.*\}", llm_response_content, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group()
            result = json.loads(json_str)
        else:
            # Fallback if LLM doesn't return valid JSON
            result = {
                "segment": None,
                "month1_name": last_month_name,
                "month2_name": month_prior_to_last_name,
                "month_of_interest_name": last_month_name,
                "compare_to_month_name": month_prior_to_last_name,
            }

        # Convert month names to actual date objects using _get_month_dates
        mo_interest_dates = _get_month_dates(result.get("month_of_interest_name"))
        compare_mo_dates = _get_month_dates(result.get("compare_to_month_name"))

        result["month_of_interest_start"] = mo_interest_dates[0] if mo_interest_dates else None
        result["month_of_interest_end"] = mo_interest_dates[1] if mo_interest_dates else None
        result["compare_to_month_start"] = compare_mo_dates[0] if compare_mo_dates else None
        result["compare_to_month_end"] = compare_mo_dates[1] if compare_mo_dates else None

        # Add a flag for comparison validity, mainly to catch if date parsing fails
        result["comparison_valid"] = (
                result["month_of_interest_start"] is not None and
                result["compare_to_month_start"] is not None
        )

        # Ensure sensible defaults if LLM somehow fails to provide names or dates
        if not result["month_of_interest_start"]:
            result["month_of_interest_start"] = last_month_start
            result["month_of_interest_end"] = last_month_end
            result["month_of_interest_name"] = last_month_name
            result["comparison_valid"] = False  # Mark as invalid if a core date is missing

        if not result["compare_to_month_start"]:
            result["compare_to_month_start"] = month_prior_to_last_start
            result["compare_to_month_end"] = month_prior_to_last_end
            result["compare_to_month_name"] = month_prior_to_last_name
            result["comparison_valid"] = False  # Mark as invalid if a core date is missing

        # The 'compare_to_previous_month' flag becomes less relevant with explicit month parsing
        result["compare_to_previous_month"] = True  # Keep for now if downstream expects it

        return result

    except Exception as e:
        print(f"Error in LLM call for cost drop parsing: {e}")
        return {
            "segment": None,
            "month_of_interest_name": last_month_name,
            "compare_to_month_name": month_prior_to_last_name,
            "month_of_interest_start": last_month_start,
            "month_of_interest_end": last_month_end,
            "compare_to_month_start": month_prior_to_last_start,
            "compare_to_month_end": month_prior_to_last_end,
            "compare_to_previous_month": True,
            "comparison_valid": False  # Mark as invalid on general error
        }

def _get_quarter_dates(quarter_name: str) -> Optional[Tuple[datetime.date, datetime.date]]:
    today = datetime.now(pytz.timezone("Asia/Kolkata")).date()
    current_calendar_year = today.year

    def get_fy_start_calendar_year(fy_suffix: int) -> int:
        # FY25 means calendar year 2024 (starts April 2024)
        # FY26 means calendar year 2025 (starts April 2025)
        # So, FY25 (suffix 25) means 2000 + 25 - 1 = 2024
        # FY26 (suffix 26) means 2000 + 26 - 1 = 2025
        return 2000 + fy_suffix - 1

    current_q_start_month = None
    current_q_end_month = None
    current_q_num = 0
    current_q_start_year_calc = current_calendar_year
    current_q_end_year_calc = current_calendar_year

    # Determine current FISCAL quarter
    if 4 <= today.month <= 6: # Apr-Jun
        current_q_start_month = 4
        current_q_end_month = 6
        current_q_num = 1 # FY Q1
    elif 7 <= today.month <= 9: # Jul-Sep
        current_q_start_month = 7
        current_q_end_month = 9
        current_q_num = 2 # FY Q2
    elif 10 <= today.month <= 12: # Oct-Dec
        current_q_start_month = 10
        current_q_end_month = 12
        current_q_num = 3 # FY Q3
    else: # Jan-Mar
        current_q_start_month = 1
        current_q_end_month = 3
        current_q_num = 4 # FY Q4
        current_q_start_year_calc = current_calendar_year # FY Q4 is Jan-Mar of the *next* calendar year relative to FY start

    current_q_start_date = datetime(current_q_start_year_calc, current_q_start_month, 1).date()
    current_q_end_date = datetime(current_q_end_year_calc, current_q_end_month, calendar.monthrange(current_q_end_year_calc, current_q_end_month)[1]).date()

    if "this quarter" in quarter_name.lower() or "current quarter" in quarter_name.lower():
        return current_q_start_date, current_q_end_date
    elif "last quarter" in quarter_name.lower():
        # Calculate dates for the last FISCAL quarter
        if current_q_num == 1: # Current is FYQ1 (Apr-Jun). Last was FYQ4 of prev FY.
            prev_fy_calendar_start_year = get_fy_start_calendar_year(today.year % 100) # This FY's calendar start year
            last_q_start = datetime(prev_fy_calendar_start_year, 1, 1).date() # Prev FY Q4 is Jan-Mar of current calendar year
            last_q_end = datetime(prev_fy_calendar_start_year, 3, 31).date()
        elif current_q_num == 2: # Current is FYQ2 (Jul-Sep). Last was FYQ1.
            last_q_start = datetime(current_q_start_year_calc, 4, 1).date()
            last_q_end = datetime(current_q_start_year_calc, 6, 30).date()
        elif current_q_num == 3: # Current is FYQ3 (Oct-Dec). Last was FYQ2.
            last_q_start = datetime(current_q_start_year_calc, 7, 1).date()
            last_q_end = datetime(current_q_start_year_calc, 9, 30).date()
        elif current_q_num == 4: # Current is FYQ4 (Jan-Mar). Last was FYQ3.
            last_q_start = datetime(current_q_start_year_calc, 10, 1).date()
            last_q_end = datetime(current_q_start_year_calc, 12, 31).date()
        return last_q_start, last_q_end
    elif "quarter prior to last" in quarter_name.lower() or "quarter before last" in quarter_name.lower():
        # This logic needs to be robust for "quarter prior to last" (Fiscal based)
        if current_q_num == 1: # FYQ1, so "quarter prior to last" is FYQ3 of previous FY
            prev_fy_calendar_start_year = get_fy_start_calendar_year(today.year % 100) - 1
            return datetime(prev_fy_calendar_start_year, 10, 1).date(), datetime(prev_fy_calendar_start_year, 12, 31).date()
        elif current_q_num == 2: # FYQ2, so "quarter prior to last" is FYQ4 of previous FY
            prev_fy_calendar_start_year = get_fy_start_calendar_year(today.year % 100)
            return datetime(prev_fy_calendar_start_year, 1, 1).date(), datetime(prev_fy_calendar_start_year, 3, 31).date()
        elif current_q_num == 3: # FYQ3, so "quarter prior to last" is FYQ1
            return datetime(current_q_start_year_calc, 4, 1).date(), datetime(current_q_start_year_calc, 6, 30).date()
        elif current_q_num == 4: # FYQ4, so "quarter prior to last" is FYQ2
            return datetime(current_q_start_year_calc, 7, 1).date(), datetime(current_q_start_year_calc, 9, 30).date()

    # --- Updated Regex for FY/Q parsing ---
    # Made 'Q' or 'q' optional with space, allowing for "Q1" or "q 1"
    # This function is specifically for _get_quarter_dates, which is called by parse_cb_comparison_details.
    # The primary quarter parsing logic for question_1 now resides within the LLM prompt of parse_date_range_from_query_llm.
    # This function is used to convert the LLM's 'period_name' (like "FY26 Q1") into actual dates.

    # 1. Try to parse as Fiscal Year Quarter
    fy_match = re.match(r"(?:FY|financial year)\s*(\d{2,4})\s*(?:Q|q(?:tr)?)\s*(\d)", quarter_name, re.IGNORECASE)
    if fy_match:
        year_part = fy_match.group(1)
        q_num = int(fy_match.group(2))

        if len(year_part) == 2:
            fy_suffix = int(year_part)
        elif len(year_part) == 4:
            fy_suffix = int(year_part) % 100
        else:
            return None

        target_calendar_year_start_of_fy = get_fy_start_calendar_year(fy_suffix)

        if q_num == 1: # FY Q1: Apr-Jun
            return datetime(target_calendar_year_start_of_fy, 4, 1).date(), datetime(target_calendar_year_start_of_fy, 6, 30).date()
        elif q_num == 2: # FY Q2: Jul-Sep
            return datetime(target_calendar_year_start_of_fy, 7, 1).date(), datetime(target_calendar_year_start_of_fy, 9, 30).date()
        elif q_num == 3: # FY Q3: Oct-Dec
            return datetime(target_calendar_year_start_of_fy, 10, 1).date(), datetime(target_calendar_year_start_of_fy, 12, 31).date()
        elif q_num == 4: # FY Q4: Jan-Mar (of next calendar year)
            return datetime(target_calendar_year_start_of_fy + 1, 1, 1).date(), datetime(target_calendar_year_start_of_fy + 1, 3, 31).date()

    # 2. Try to parse as Calendar Year Quarter (Q1, Q2, Q3, Q4)
    # This regex is specifically for when the LLM from parse_cb_comparison_details might return something like "2025 Q2"
    cal_q_match = re.match(r"(\d{4})\s*(?:Q|q(?:tr)?)\s*(\d)", quarter_name, re.IGNORECASE)
    if cal_q_match:
        year = int(cal_q_match.group(1))
        q_num = int(cal_q_match.group(2))
        if q_num == 1:
            return datetime(year, 1, 1).date(), datetime(year, 3, 31).date()
        elif q_num == 2:
            return datetime(year, 4, 1).date(), datetime(year, 6, 30).date()
        elif q_num == 3:
            return datetime(year, 7, 1).date(), datetime(year, 9, 30).date()
        elif q_num == 4:
            return datetime(year, 10, 1).date(), datetime(year, 12, 31).date()

    return None

def _get_month_dates(month_name_year: str) -> Optional[Tuple[datetime.date, datetime.date]]:
    today = datetime.now(pytz.timezone("Asia/Kolkata")).date()

    if "this month" in month_name_year.lower() or "current month" in month_name_year.lower():
        start_date = today.replace(day=1)
        last_day = calendar.monthrange(start_date.year, start_date.month)[1]
        end_date = start_date.replace(day=last_day)
        return start_date, end_date
    elif "last month" in month_name_year.lower() or "previous month" in month_name_year.lower():
        end_date = today.replace(day=1) - timedelta(days=1)
        start_date = end_date.replace(day=1)
        return start_date, end_date
    elif "next month" in month_name_year.lower():
        current_month_start = today.replace(day=1)
        start_date = (current_month_start + timedelta(days=32)).replace(day=1)
        last_day = calendar.monthrange(start_date.year, start_date.month)[1]
        end_date = start_date.replace(day=last_day)
        return start_date, end_date

    try:
        # Fuzzy parsing handles "Apr 25", "April 2025" etc.
        parsed_date = parser.parse(month_name_year, fuzzy=True, dayfirst=False)
        start_date = parsed_date.replace(day=1).date()
        last_day = calendar.monthrange(start_date.year, start_date.month)[1]
        end_date = start_date.replace(day=last_day)
        return start_date, end_date
    except ValueError:
        return None, None

def parse_cb_comparison_details(query: str) -> dict:
    today = datetime.now(pytz.timezone("Asia/Kolkata")).date()
    current_year = today.year

    system_prompt = f"""Today's date is {today.strftime('%Y-%m-%d')}.
The financial calendar is:
FYXX Q1: April - June
FYXX Q2: July - September
FYXX Q3: October - December
FYXX Q4: January - March (of the next calendar year)

For example:
- FY25 Q4 is Jan 1, 2025 to Mar 31, 2025
- FY26 Q1 is Apr 1, 2025 to Jun 30, 2025 (This is the current fiscal quarter if today is in Apr-Jun 2025)
- FY26 Q2 is Jul 1, 2025 to Sep 30, 2025
- FY26 Q3 is Oct 1, 2025 to Dec 31, 2025
- FY26 Q4 is Jan 1, 2026 to Mar 31, 2026

You are an expert at identifying two specific time periods (either months or quarters) from a user's query for C&B cost comparison.
Determine if the comparison is 'month' based or 'quarter' based.
Extract the *first* mentioned period as 'period1_name' and the *second* mentioned period as 'period2_name'.
These names can be explicit (e.g., "April 2025", "FY26 Q1") or relative (e.g., "last month", "this quarter").
Prioritize explicit names if they are present.
Assume quarters mentioned without 'FY' (e.g., '2025 Q1', 'Q2 2024') refer to standard calendar year quarters (Jan-Mar for Q1, Apr-Jun for Q2, etc.).

Return a JSON object with:
- 'comparison_type': 'month' or 'quarter'. If unclear or only one period, default to 'quarter'.
- 'period1_name': The name of the first period mentioned in the query.
- 'period2_name': The name of the second period mentioned in the query.

Example outputs:
- Query: "How much C&B varied from April 2025 to May 2025"
  Output: {{"comparison_type": "month", "period1_name": "April 2025", "period2_name": "May 2025"}}
- Query: "How much C&B varied from FY26 Q1 to FY25 Q4"
  Output: {{"comparison_type": "quarter", "period1_name": "FY26 Q1", "period2_name": "FY25 Q4"}}
- Query: "Compare C&B for this quarter and last quarter"
  Output: {{"comparison_type": "quarter", "period1_name": "this quarter", "period2_name": "last quarter"}}
- Query: "C&B variation between June 2025 and July 2024"
  Output: {{"comparison_type": "month", "period1_name": "June 2025", "period2_name": "July 2024"}}
- Query: "Show C&B for August 2025" (only one month mentioned, still classify as 'month' type)
  Output: {{"comparison_type": "month", "period1_name": "August 2025", "period2_name": null}}
- Query: "Show C&B for FY26 Q1" (only one quarter mentioned, still classify as 'quarter' type)
  Output: {{"comparison_type": "quarter", "period1_name": "FY26 Q1", "period2_name": null}}
- Query: "C&B variation" (no specific periods)
  Output: {{"comparison_type": "quarter", "period1_name": "last quarter", "period2_name": "this quarter"}} # Default to quarter comparison if vague
- Query: "Compare C&B for 2025 Q1 and 2025 Q2"
  Output: {{"comparison_type": "quarter", "period1_name": "2025 Q1", "period2_name": "2025 Q2"}} # Calendar quarters

If periods are unclear or only one is mentioned, set the missing one(s) to null but still try to infer comparison_type.
Ensure the JSON output is perfectly valid, with no trailing commas.
"""
    try:
        if not utils_openai_client:
            return {"comparison_type": "quarter", "period1_name": None, "period2_name": None, "comparison_valid": False}

        response = utils_openai_client.chat.completions.create(
            model="gpt-35-turbo", # Use the deployment name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0
        )
        llm_response_content = response.choices[0].message.content.strip()

        json_str_match = re.search(r"\{.*\}", llm_response_content, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group()
            parsed_llm_response = json.loads(json_str)
        else:
            return {"comparison_type": "quarter", "period1_name": None, "period2_name": None, "comparison_valid": False}

        comparison_type = parsed_llm_response.get("comparison_type", "quarter")
        p1_name = parsed_llm_response.get("period1_name")
        p2_name = parsed_llm_response.get("period2_name")

        p1_start, p1_end = None, None
        p2_start, p2_end = None, None

        if comparison_type == 'quarter':
            p1_dates = _get_quarter_dates(p1_name) if p1_name else (None, None)
            p2_dates = _get_quarter_dates(p2_name) if p2_name else (None, None)
            p1_start, p1_end = p1_dates
            p2_start, p2_end = p2_dates
        elif comparison_type == 'month':
            p1_dates = _get_month_dates(p1_name) if p1_name else (None, None)
            p2_dates = _get_month_dates(p2_name) if p2_name else (None, None)
            p1_start, p1_end = p1_dates
            p2_start, p2_end = p2_dates

        # If only P1 is provided and valid, infer P2 as the next month/quarter
        if p1_start and not p2_start:
            if comparison_type == 'month':
                # Calculate next month
                if p1_end: # Ensure p1_end is not None before calculating next month
                    next_month_start = (p1_end + timedelta(days=1)).replace(day=1)
                    last_day_next_month = calendar.monthrange(next_month_start.year, next_month_start.month)[1]
                    next_month_end = next_month_start.replace(day=last_day_next_month)
                    p2_start, p2_end = next_month_start, next_month_end
                    p2_name = next_month_start.strftime("%B %Y") # Give a name to the inferred period
            elif comparison_type == 'quarter':
                # This part is more complex as it depends on whether P1 was fiscal or calendar
                # For simplicity and to avoid over-complicating LLM behavior, let's keep it manual or for simple cases.
                # If a more robust "next quarter" inference is needed, it would require a more sophisticated _get_next_quarter_dates function.
                pass


        result = {
            "comparison_type": comparison_type,
            "period1_name": p1_name,
            "period1_start": p1_start,
            "period1_end": p1_end,
            "period2_name": p2_name, # Updated name for the inferred period
            "period2_start": p2_start,
            "period2_end": p2_end,
            "comparison_valid": p1_start is not None and (p2_start is not None or (p2_name is None and p2_start is None))
            # comparison_valid true if p1 is valid AND (p2 is valid OR p2 wasn't asked for/inferred)
        }
        return result

    except Exception as e:
        print(f"Error in LLM call for C&B comparison parsing: {e}")
        return {"comparison_type": "quarter", "period1_name": None, "period2_name": None, "comparison_valid": False}

# --- NEW FUNCTION FOR QUESTION 4 FILTER PARSING ---
def parse_cb_revenue_trend_details(query: str) -> dict:
    """
    Parses a query for C&B cost % of total revenue trend analysis.
    Primarily extracts date range if specified (e.g., 'last 6 months', '2024').
    """
    # Reuse the general date range parsing LLM call
    date_info = parse_date_range_from_query_llm(query)

    # For trend analysis, we mostly care about the overall date range.
    # No other specific filters like segments or CM thresholds are expected here.
    return {
        "date_filter": date_info["date_filter"],
        "start_date": date_info["start_date"],
        "end_date": date_info["end_date"],
        "description": date_info["description"]
    }
# --- END NEW FUNCTION ---


def parse_query_filters(user_query: str, question_id: str) -> dict:
    if question_id == "question_1":
        return get_cm_query_details(user_query)
    elif question_id == "question_2":
        return get_cost_drop_query_details(user_query)
    elif question_id == "question_3":
        return parse_cb_comparison_details(user_query)
    elif question_id == "question_4": # --- ADDED FOR QUESTION 4 ---
        return parse_cb_revenue_trend_details(user_query)
    elif question_id == "question_5":
        # This is a conceptual example, adapt based on how your OpenAI client
        # is set up in logic/utils.py and how you want to extract dates/periods.
        extraction_prompt = f"""
            Extract the following information from the user query for revenue trend analysis:
            - 'period_type': 'Year', 'Quarter', or 'Month' (based on YoY, QoQ, MoM). Default to 'Month' if not specified.
            - 'start_date': (YYYY-MM-DD or the start date of the period mentioned, e.g., '2024-01-01' for Q1 2024 or Jan 2024. If 'last quarter', 'this quarter', 'last year', 'this year', calculate relative dates from current date {datetime.now().strftime('%Y-%m-%d')}). If no specific period is mentioned, infer a reasonable default (e.g., last 12 months, or start of data).
            - 'end_date': (YYYY-MM-DD or the end date of the period). If not specified, infer end of the inferred period.
            - 'grouping_dimension': 'DU', 'BU', 'Account', or 'All' (if not specified).
            - 'specific_group': The exact name of the DU, BU, or Account if mentioned (e.g., 'CustA').

            Return a JSON object with these keys. If a value cannot be extracted, set it to null.
            User query: "{user_query}"
            """

        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,  # Assuming AZURE_OPENAI_DEPLOYMENT is available
                messages=[
                    {"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": user_query}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            extracted_info = json.loads(response.choices[0].message.content)

            # Convert string dates to datetime objects and handle defaults/logic
            start_date_str = extracted_info.get("start_date")
            end_date_str = extracted_info.get("end_date")

            parsed_start_date = None
            if start_date_str:
                try:
                    parsed_start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                except ValueError:
                    # Handle cases like "last quarter", "this year" by calculating dates
                    # This part would need more sophisticated date parsing logic.
                    pass

            parsed_end_date = None
            if end_date_str:
                try:
                    parsed_end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
                except ValueError:
                    pass

            # If start_date/end_date are still None, set defaults (e.g., last 12 months)
            if not parsed_start_date or not parsed_end_date:
                # Example: Default to last 12 months from current date
                current_date = datetime.now().date()
                parsed_end_date = current_date.replace(day=1) - timedelta(days=1)  # End of last month
                parsed_start_date = (parsed_end_date - timedelta(days=365)).replace(day=1)

            return {
                "date_filter": True,  # Indicate that a date filter attempt was made
                "start_date": parsed_start_date,
                "end_date": parsed_end_date,
                "period_type": extracted_info.get("period_type"),  # 'Year', 'Quarter', 'Month'
                "grouping_dimension": extracted_info.get("grouping_dimension", "All"),
                "specific_group": extracted_info.get("specific_group")
            }

        except Exception as e:
            print(f"Error in parse_query_filters for question_5: {e}")
            return {
                "date_filter": False,
                "start_date": None,
                "end_date": None,
                "period_type": None,
                "grouping_dimension": "All",  # Default to All if parsing fails
                "specific_group": None,
                "Message": f"Could not parse filters for Q5: {e}"
            }
    else:
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
            "compare_to_previous_month": False,
            "comparison_type": "quarter",
            "period1_name": None,
            "period1_start": None,
            "period1_end": None,
            "period2_name": None,
            "period2_start": None,
            "period2_end": None,
            "comparison_valid": False,
            "description": "all available data" # Added for question 4 default
        }

REVENUE_GROUPS = ["ONSITE", "OFFSHORE", "INDIRECT REVENUE"]
COST_GROUPS = [
    "Direct Expense", "OWN OVERHEADS", "Indirect Expense",
    "Project Level Depreciation", "Direct Expense - DU Block Seats Allocation",
    "Direct Expense - DU Pool Allocation", "Establishment Expenses"
]
# Define C&B groups here as well, if they are not exclusively defined in question_3.py
# (It's better to define them globally here if used by multiple logic files)
CB_GROUPS = ["C&B Cost Offshore", "C&B Cost Onsite"]

# --- COMPLETE calculate_cm function ---
def calculate_cm(df: pd.DataFrame, cm_filters: dict) -> pd.DataFrame:
    grouped = df.groupby("FinalCustomerName", as_index=False).apply(lambda x: pd.Series({
        "Revenue": x[x["Group1"].isin(REVENUE_GROUPS)]["Amount in USD"].sum(),
        "Cost": x[x["Group1"].isin(COST_GROUPS)]["Amount in USD"].sum()
    }), include_groups=False)

    grouped["Revenue"] = grouped["Revenue"].round(2)
    original_rows = len(grouped)
    grouped = grouped[grouped["Revenue"] != 0].copy()
    if len(grouped) < original_rows:
        pass # print(f"Removed {original_rows - len(grouped)} customers due to zero revenue.")

    revenue_abs = grouped["Revenue"].abs()
    # Handle cases where revenue_abs might be zero to avoid division by zero or inf/-inf
    grouped["CM_Ratio"] = (grouped["Revenue"] - grouped["Cost"]) / revenue_abs.replace(0, np.nan)
    grouped["CM_Ratio"] = grouped["CM_Ratio"].replace([float('inf'), -float('inf')], float('nan'))

    grouped = grouped.dropna(subset=["CM_Ratio"])

    filtered = grouped.copy()
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
        else:
            # If "between" but bounds are missing/invalid, do not apply CM filter
            pass
    elif cm_filters["type"] == "equals" and cm_filters["lower"] is not None:
        tolerance = 0.0001 # Small tolerance for float equality
        filtered = filtered[
            (filtered["CM_Ratio"] >= cm_filters["lower"] - tolerance) &
            (filtered["CM_Ratio"] <= cm_filters["lower"] + tolerance)
        ]

    if cm_filters["type"] != "none":
        # Sort based on filter type for more intuitive results
        filtered = filtered.sort_values(
            by="CM_Ratio",
            ascending=(cm_filters["type"] == "less_than" or cm_filters["type"] == "equals") # Ascending for less_than/equals, descending for greater_than/between
        ).reset_index(drop=True)
    else:
        filtered = filtered.sort_values(by="CM_Ratio", ascending=False).reset_index(drop=True)

    filtered["CM_Value"] = filtered["CM_Ratio"] * 100 # Store raw CM value for plotting
    filtered["CM (%)"] = filtered["CM_Value"].apply(
        lambda x: "N/A" if pd.isna(x) else f"{x:,.2f}%"
    )

    filtered["Revenue"] = filtered["Revenue"].apply(lambda x: f"${x:,.2f}")
    filtered["Cost"] = filtered["Cost"].apply(lambda x: f"${x:,.2f}")

    # Add S.No
    filtered.reset_index(drop=True, inplace=True)
    filtered.index = filtered.index + 1
    filtered = filtered.rename_axis("S.No").reset_index()

    return filtered[["S.No", "FinalCustomerName", "Revenue", "Cost", "CM (%)", "CM_Value"]] # Include CM_Value for plotting in app.py
# --- END COMPLETE calculate_cm function ---