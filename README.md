# FedEx Topic Modeling 

## Project Overview
This tool automates the classification of customer emails into hierarchical categories (Category > Subcategory). It utilizes a **Local Large Language Model (Llama 3.1)** to ensure data privacy and **Semantic Search** for deduplication.

## Security & Architecture
*   **Zero Data Exfiltration:** All inference runs locally on the machine using Ollama. No email data is sent to external APIs (e.g., OpenAI, Azure).
*   **PII Cleaning:** Automated Regex removes emails, phone numbers, and addresses before processing.
*   **Deduplication:** Uses `sentence-transformers` (all-MiniLM-L6-v2) to remove semantic duplicates locally.

## Prerequisites
1.  **Python 3.13.5** installed.
2.  **Ollama** (Required for the LLM):
    *   Download and install from [ollama.com](https://ollama.com).
    *   Ensure Ollama is running in the background.
    *   Pull the specific model by running this command in your terminal:
        ```bash
        ollama pull llama3.1:8b
        ```

## Installation
1.  Clone this repository or download the source code.
2.  Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run
1.  **Prepare Data:** 
    *   Place your raw Excel file in this folder.
    *   **Rename the file** to: `input_data.xlsx`
2.  **Execute Script:**
    ```bash
    python main.py
    ```

## Output
The script generates an `output/` folder containing:
*   `Topic_Modelling_Results.xlsx` (Final classified data)
*   `Topic_Modelling_Results.json` (JSON format)

*   `Topic_Modelling_Results.html` (Visual report)
