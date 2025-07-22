from openai import AzureOpenAI
import os

# Initialize the client (example: Azure)
# IMPORTANT: Replace with your actual Azure OpenAI key and endpoint if different.
AZURE_OPENAI_KEY = "4V95tcXbVO3y2uyhKiAISmxC9ALiUaKvofpcEgNCoYWlBPvAatWcJQQJ99BGACfhMk5XJ3w3AAAAACOGPrhY" # Ensure this is your actual, full API key
AZURE_OPENAI_ENDPOINT = "https://azure-md46msq5-swedencentral.openai.azure.com/" # Your actual, working endpoint
AZURE_OPENAI_DEPLOYMENT = "gpt-35-turbo" # Ensure this deployment name is correct
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"

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
Return ONLY the identifier, like 'question_1', 'question_2', or 'unknown'.

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
- question_3: Questions asking for average CM% across project types.
    Examples for question_3:
    - "What is the average CM for 'Type A' projects?"
    - "Calculate average CM for projects of type 'Consulting'"

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

