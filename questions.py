import os
from openai import AzureOpenAI

# ---------- CONFIGURATION ----------
# IMPORTANT: Read API key and endpoint from environment variables for deployment.
# Set these as secrets in your Streamlit Community Cloud deployment settings.
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
# Get raw endpoint, then strip if it's not None
raw_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_ENDPOINT = raw_endpoint.strip() if raw_endpoint else None

AZURE_OPENAI_DEPLOYMENT = "gpt-35-turbo" # Ensure this deployment name is correct
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"

# --- DEBUGGING PRINTS ---
print(f"DEBUG: questions.py - AZURE_OPENAI_KEY (first 5 chars): {AZURE_OPENAI_KEY[:5] if AZURE_OPENAI_KEY else 'None'}")
print(f"DEBUG: questions.py - AZURE_OPENAI_ENDPOINT (RAW from env): '{AZURE_OPENAI_ENDPOINT}'")
# --- END DEBUGGING PRINTS ---

# Check if environment variables are set
if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
    print("WARNING: AZURE_OPENAI_KEY or AZURE_OPENAI_ENDPOINT environment variables are not set.")
    print("Please set them as Streamlit secrets for deployment, or as local environment variables in your .env file for local testing.")
    raise ValueError("API keys for Azure OpenAI are not configured as environment variables.")

# Add a try-except block around client initialization for better error reporting
try:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION
    )
    # Attempt a small, quick API call to verify connectivity and credentials
    test_response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )
    print("Azure OpenAI client initialized and tested successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize Azure OpenAI client or connect to API.")
    print(f"Please check your AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, and network connectivity.")
    print(f"Detailed error: {e}")
    client = None
    raise # Re-raise the exception to stop the import and show the error

def get_question_id(user_input: str) -> str:
    """
    Routes user natural language questions to specific logic file identifiers
    using an Azure OpenAI model.
    """
    if client is None:
        print("Azure OpenAI client is not initialized. Cannot route question.")
        return "unknown"

    system_prompt = """
    You are a routing engine that maps natural language questions to logic file identifiers.
    Each identifier corresponds to a question type.
    Return ONLY the identifier, like 'question_1', 'question_2', 'question_3', 'question_4', 'question_5', or 'unknown'.

    Available identifiers and examples:
    - question_1: Questions about customers with CM% thresholds and optional date filters, including listing customers, average CM, total revenue, and CM distribution.
        Examples for question_1:
        - "List customers with CM > 90% last quarter"
        - "Show customers with CM less than 30%"
        - "What is the CM for customers between 10% and 15% in January 2023?"
        - "Show me all customers with their CM"
        - "What is the average CM for all customers?"
        - "List customers with CM greater than 20%"
        - "Customers with CM less than 50% in Q1 2024"
        - "Give me the contribution margin details for all clients"
    - question_2: Questions about specific cost increases/drops for a segment over time (month-over-month comparison).
        Examples for question_2:
        - "Which cost triggered the Margin drop last month as compared to its previous month in Transportation"
        - "Show me cost increases in Healthcare for July 2024 compared to June"
        - "What expenses went up in Retail last quarter?"
    - question_3: Questions asking for C&B cost variation between any two specified periods (months or quarters).
        Examples for question_3:
        - "How much C&B varied from last quarter to this quarter"
        - "C&B cost variation between FY26 Q1 and FY25 Q4"
        - "Compare C&B for FY25 Q4 and FY26 Q2"
        - "What's the difference in C&B between this quarter and last quarter?"
        - "How much C&B varied from April 2025 to May 2025"
        - "Compare C&B for last month and this month"
        - "C&B variation between June 2025 and July 2024"
    - question_4: Questions about the Month-over-Month (M-o-M) trend of C&B cost as a percentage of total revenue.
        Examples for question_4:
        - "What is M-o-M trend of C&B cost % w.r.t total revenue"
        - "Show me the monthly C&B cost as a percentage of revenue"
        - "C&B cost to revenue trend over time"
        - "Analyze C&B cost percentage of revenue month over month"
        - "C&B as % of revenue trend for last 6 months"
    - question_5: Questions about Year-over-Year (YoY), Quarter-over-Quarter (QoQ), or Month-over-Month (MoM) revenue trends, often segmented by DU, BU, or account.
        Examples for question_5:
        - "What is the YoY, QoQ, MoM revenue for DU/BU/account"
        - "Show me the quarterly revenue trend by DU"
        - "Give me the month over month revenue for specific accounts"
        - "Analyze yearly revenue changes for all BUs"
        - "What's the QoQ revenue for account 'XYZ'?"

    If the input doesn’t match any known type, return 'unknown'.
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in get_question_id API call: {e}")
        return "unknown"