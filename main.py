import pandas as pd
import re 
import nltk 
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import subprocess

# ==============================================================================
# CONFIGURATION - PLEASE UPDATE INPUT FILE NAME HERE

# Place your Excel file in the same folder as this script and rename it to 'input_data.xlsx'
# or change the name below to match your file.
input_file_path = "input_data.xlsx" 
categories_config_path = "categories.json"

# Output files will be saved in the current folder
output_excel_path = "Topic_Modelling_Output.xlsx"
output_json_path = "Topic_Modelling_Output.json"
output_html_path = "Topic_Modelling_Report.html"
# ==============================================================================

tqdm.pandas()

print(f"Reading data from: {input_file_path}")
# Using try/except to give a helpful error if file is missing
try:
    df = pd.read_excel(input_file_path, engine='openpyxl')
except FileNotFoundError:
    print(f"ERROR: Could not find {input_file_path}. Please make sure the file exists.")
    exit()

model = SentenceTransformer("all-MiniLM-L6-v2")

# Check if 'cleaned_text' exists, if not, use the first column or 'textbody'
if 'cleaned_text' in df.columns:
    df = df[['cleaned_text']]
else:
    # Fallback to create cleaned_text from the first available text column
    col_to_use = 'textbody' if 'textbody' in df.columns else df.columns[0]
    df['cleaned_text'] = df[col_to_use].astype(str)
    df = df[['cleaned_text']]

#lowercase
df['cleaned_text'] = df['cleaned_text'].str.lower()

#duplicates removal
print("Removing Duplicates...")
df = df.drop_duplicates(subset=['cleaned_text'], keep='first')

#Cosine similarity removal
texts = df['cleaned_text'].fillna('').astype(str).tolist()
embeddings = model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)

similarity_matrix = cosine_similarity(embeddings)

to_remove = set()

threshold = 1
for i in tqdm(range(len(texts))):
    for j in range(i):
        if similarity_matrix[i][j] >= threshold:
            to_remove.add(i)
            break  # skip remaining checks for this i

valid_indices = list(set(df.index).intersection(to_remove))
df_dedup = df.drop(index=valid_indices).reset_index(drop=True)

#Remove standard disclaimers
disclaimers = [
    "this e-mail is intended solely for use by the individual or entity to which it is addressed and may contain information that is proprietary, privileged, company confidential and/or restricted from disclosure under applicable law. if the reader is not the intended recipient or authorized representative, you are hereby notified that any use, retention, disclosure, copying, printing, forwarding or dissemination of this communication is strictly prohibited. if you have received this communication in error, please erase all copies of the message and its attachments and notify the sender immediately.",
    "caution! this email originated outside of fedex. please do not open attachments or click links from an unknown or suspicious origin.",
    "you may receive an invitation to participate in a fedex dedicated customer care survey. i kindly request you to take a few moments to share your candid feedback and experiences of the service that i have provided.",
    "our new customer center guides you through all the necessary steps when shipping with fedex! for and payment inquiries, please go to customer support or register to fedex billing online our call center will be closed during major national holidays. operations will resume on the next business day."
]
pattern = "|".join(re.escape(text) for text in disclaimers)
def remove_disclaimers(text):
    if isinstance(text, str):
        return re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
    return text
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].apply(remove_disclaimers)

#Remove confidentiality clause
conf_pattern = re.compile(
    r"\*statement of confidentiality:\*.*?if you have received this email in error, please immediately notify the sender and destroy all hard copies and any copies that may be on your computer\.",
    re.IGNORECASE | re.DOTALL
)
def clean_confidentiality(text):
    if pd.isna(text):
        return text
    return re.sub(conf_pattern, '', str(text)).strip()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(clean_confidentiality)
print("All columns cleaned!")

#Remove HTML using safer check (FIXED HERE)
def remove_html(text):
    if pd.isna(text):
        return text
    text = str(text)
    if not re.search(r'<[^>]+>', text):
        return text
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ")
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(remove_html)

#Remove URLs
def remove_urls(text):
    if pd.isna(text):
        return text
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', str(text)).strip()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(remove_urls)

# General text cleaning
def clean_text(text):
    if pd.isna(text):
        return text
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', '', text)
    text = re.sub(r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b', '', text)
    text = re.sub(r'\b\d{1,2}:\d{2}(\s?[APap][Mm])?\b', '', text)
    return text.strip()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(clean_text)
print("Emails, dates, times, and URLs removed.")

# Remove punctuation/special characters
def clean_text_punct(text):
    if isinstance(text, str):
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text
df = df.map(clean_text_punct)

# Remove common form field phrases
words_to_remove = ["form details", "submitted from", "field name value", "categorytopic", "queryintent",
    "detailsstatement", "contact form details", "first name", "last name", "email", "phone number",
    "your message", "account and shipping details", "account number", "tracking number",
    "fedex invoice number", "company name", "address", "address line 2", "postal code", "city",
    "request specific fields", "current recipient company name", "current delivery address",
    "current delivery address 2", "current delivery postal code", "current delivery city",
    "new recipient company name", "new delivery address", "new delivery address 2",
    "new delivery postal code", "new delivery city", "confirmed recipient", "confirmed delivery address",
    "confirmed delivery address 2", "confirmed delivery postal code", "confirmed delivery city",
    "new date of delivery", "collection company name", "collection address", "collection address 2",
    "collection postal code", "collection city", "pickup reference number", "current pickup date",
    "new pickup date", "reason of cancelation", "how many packages were delivered in total",
    "how many packages are damaged", "was there any visible damage to the outer box if so please describe the damage",
    "what type of inner packaging material was used to protect contents was the inner packaging material damaged",
    "can you describe the contents and specify the damage that was done to the content",
    "how many itemsunits are damaged", "please provide the value declared value and declared customs value for the damaged items",
    "if there was damage to the box does the damage correspond with the damage to the contents",
    "can the contents be repaired if so what is the repair estimate cost", "payment method for the transportation charges",
    "refund method", "how many packagesitems were sent", "how many packagesitems are missing",
    "please describe the external and internal packaging of the missing shipment",
    "are there any identifying markingslabelscolors on the outer and inner packaging",
    "what is the description of the missing items including size color fabric etc",
    "are there any partserial numbers on the items or packaging",
    "if it is a missing phoneelectricalcomputer item please provide the serialmodel or imei number",
    "are there any order numbers associated with the item lost",
    "what is the manufacturersbrand name", "application name", "fedexcom login name", "origin city",
    "origin countryterritory", "origin postal code", "destination city", "destination countryterritory",
    "destination postal code", "ship date", "dimensions per package lxwxh inscms",
    "numbers of packages", "weight per package", "submission details", "detailsstatement scenarionumber",
    "language code", "country code", "campaign id", "page path", "submitted time", "thank you"]
pattern = re.compile(r"\b(" + "|".join(map(re.escape, words_to_remove)) + r")\b", flags=re.IGNORECASE)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna("").astype(str).apply(lambda x: pattern.sub("", x))

# Remove templated greetings/phrases
regex_patterns = [
    r"\bhello\b", r"\bgood\s*day\b", r"\byou\s+may\s+receive.*?thank\s+you\s+for\s+your\s+time",
    r"please\s+feel\s+free\s+to\s+drop\s+us\s+a\s+note.*?stay\s+safe",
    r"\bdear\s+customer\b", r"download\s+the\s+fedex\s+mobile\s+app\s+now",
    r"thank\s+you\s+for\s+choosing\s+fedex", r"\bdear\s+team\b",
    r"\bautomated\s+message\b", r"please\s+do\s+not\s+respond",
    r"your\s+enquiry\s+is\s+important\s+to\s+us.*?shipping\s+needs"
]
combined_pattern = re.compile("|".join(regex_patterns), flags=re.IGNORECASE | re.DOTALL)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna("").astype(str).apply(lambda x: combined_pattern.sub("", x))
print("Flexible regex boilerplates removed.")

# Remove numbers
def remove_numbers(text):
    if pd.isna(text):
        return text
    return re.sub(r'\d+', '', str(text))
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(remove_numbers)


with open(categories_config_path, encoding="utf-8") as f:
    categories_hierarchy = json.load(f)

# Llama calling
def run_llama_local(prompt: str, timeout: int = 90) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.1:8b"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
        output = result.stdout.strip()
        return output
    except:
        return ""

# Prompt
def build_few_shot_prompt(text):
    prompt = "Classify the following customer message into category and subcategory using examples:\n\n"
    for cat, subcats in categories_hierarchy.items():
        prompt += f"Category: {cat}\n"
        for subcat, examples in subcats.items():
            prompt += f"  Subcategory: {subcat}\n"
            for ex in examples[:3]:
                prompt += f"    - {ex}\n"
    prompt += f"\nCustomer message:\n{text}\n\n"
    prompt += "Respond in clean JSON only: {\"category\":\"...\",\"subcategory\":\"...\",\"category_conf\":0-1,\"subcategory_conf\":0-1}"
    return prompt

# categories.txt json file
def safe_json_parse(output):
    try:
        start = output.find("{")
        end = output.rfind("}")
        if start == -1 or end == -1:
            return None
        return json.loads(output[start:end + 1])
    except:
        return None


def classify_llama_few_shot(text):
    if not isinstance(text, str) or not text.strip():
        return {"category": "unknown", "subcategory": "unknown", "category_conf": 0.0, "subcategory_conf": 0.0}

    prompt = build_few_shot_prompt(text)
    raw = run_llama_local(prompt)

    parsed = safe_json_parse(raw)
    if not parsed:
        return {"category": "unknown", "subcategory": "unknown", "category_conf": 0.0, "subcategory_conf": 0.0}

    return {
        "category": parsed.get("category", "unknown"),
        "subcategory": parsed.get("subcategory", "unknown"),
        "category_conf": float(parsed.get("category_conf", 0.0)),
        "subcategory_conf": float(parsed.get("subcategory_conf", 0.0))
    }


N = 5000 
print("Classifying Emails...")
text_column = 'cleaned_text' if 'cleaned_text' in df.columns else 'textbody'
df_subset = df.iloc[:N].copy()

# Applying classification
results = df_subset[text_column].progress_apply(classify_llama_few_shot)
results_df = pd.DataFrame(list(results))

df_subset = pd.concat([df_subset.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

html = df_subset.to_html(
    index=False,
    justify="center",
    border=0,
    classes="table table-striped",
    escape=False
)

html_full = f"""
<html>
<head>
<style>
body {{
    font-family: Arial, sans-serif;
    padding: 30px;
    background: #f5f5f5;
}}
h1 {{
    color: #333;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    background: white;
    font-size: 14px;
}}
th, td {{
    border: 1px solid #ddd;
    padding: 8px;
}}
td {{
    white-space: normal;
    word-wrap: break-word;
    max-width: 1000px;
}}
th {{
    background: #4a90e2;
    color: white;
}}
tr:nth-child(even) {{ background: #f2f2f2; }}
</style>
</head>
<body>
<h1>Email Classification Report</h1>
{html}
</body>
</html>
"""

with open(output_html_path, "w", encoding="utf-8") as f:
    f.write(html_full)

print("HTML saved to:", output_html_path)

df_subset.to_excel(output_excel_path, index=False)

with open(output_json_path, "w", encoding="utf-8") as jf:
    json.dump(df_subset.to_dict(orient="records"), jf, indent=4, ensure_ascii=False)