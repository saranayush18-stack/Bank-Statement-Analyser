"""
Offline AI-Enabled Bank Statement Audit & Tax Analyzer
=======================================================
Privacy-first · Zero cloud APIs · Runs 100% locally
"""

import re
import io
import json
import datetime
import warnings
import requests
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from thefuzz import fuzz, process

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AuditLens · Bank Statement Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_BG      = "#0D1117"
PANEL_BG     = "#161B22"
BORDER_CLR   = "#30363D"
ACCENT_BLUE  = "#388BFD"
ACCENT_GREEN = "#3FB950"
ACCENT_RED   = "#F85149"

def hex_to_rgba(hex_color: str, alpha: float = 0.09) -> str:
    """Convert a '#RRGGBB' hex color to a Plotly-compatible 'rgba(r,g,b,a)' string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
ACCENT_AMBER = "#D29922"
TEXT_PRIMARY = "#E6EDF3"
TEXT_MUTED   = "#8B949E"
FONT_MONO    = "'JetBrains Mono', 'Fira Code', monospace"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
  }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background-color: {PANEL_BG};
    border-right: 1px solid {BORDER_CLR};
  }}
  [data-testid="stSidebar"] * {{ color: {TEXT_PRIMARY} !important; }}

  /* Main area */
  .main .block-container {{
    padding: 1.5rem 2rem;
    max-width: 1400px;
  }}

  /* Tab bar */
  [data-baseweb="tab-list"] {{
    background-color: {PANEL_BG};
    border-bottom: 1px solid {BORDER_CLR};
    border-radius: 8px 8px 0 0;
    gap: 0;
    padding: 0 1rem;
  }}
  [data-baseweb="tab"] {{
    color: {TEXT_MUTED} !important;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    padding: 0.75rem 1.25rem !important;
    border-bottom: 2px solid transparent;
    transition: all 0.15s ease;
  }}
  [data-baseweb="tab"]:hover {{ color: {TEXT_PRIMARY} !important; }}
  [aria-selected="true"] {{
    color: {ACCENT_BLUE} !important;
    border-bottom: 2px solid {ACCENT_BLUE} !important;
    background: transparent !important;
  }}
  [data-baseweb="tab-panel"] {{
    background: {DARK_BG};
    padding: 1.5rem 0;
  }}

  /* KPI Cards */
  .kpi-card {{
    background: {PANEL_BG};
    border: 1px solid {BORDER_CLR};
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    transition: border-color 0.2s;
  }}
  .kpi-card:hover {{ border-color: {ACCENT_BLUE}44; }}
  .kpi-label {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT_MUTED};
    margin-bottom: 0.4rem;
  }}
  .kpi-value {{
    font-size: 1.75rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    font-family: {FONT_MONO};
    line-height: 1;
  }}
  .kpi-delta {{
    font-size: 0.78rem;
    margin-top: 0.3rem;
    font-weight: 500;
  }}
  .kpi-positive {{ color: {ACCENT_GREEN}; }}
  .kpi-negative {{ color: {ACCENT_RED}; }}
  .kpi-neutral  {{ color: {TEXT_MUTED}; }}

  /* Section headers */
  .section-header {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {TEXT_MUTED};
    border-bottom: 1px solid {BORDER_CLR};
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
  }}

  /* Status badge */
  .badge {{
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.55rem;
    border-radius: 20px;
    letter-spacing: 0.04em;
  }}
  .badge-green  {{ background: {ACCENT_GREEN}22; color: {ACCENT_GREEN}; border: 1px solid {ACCENT_GREEN}44; }}
  .badge-red    {{ background: {ACCENT_RED}22;   color: {ACCENT_RED};   border: 1px solid {ACCENT_RED}44;   }}
  .badge-amber  {{ background: {ACCENT_AMBER}22; color: {ACCENT_AMBER}; border: 1px solid {ACCENT_AMBER}44; }}
  .badge-blue   {{ background: {ACCENT_BLUE}22;  color: {ACCENT_BLUE};  border: 1px solid {ACCENT_BLUE}44;  }}

  /* Flag row highlight */
  .flag-high {{ background-color: {ACCENT_RED}18 !important; }}
  .flag-med  {{ background-color: {ACCENT_AMBER}18 !important; }}

  /* Dataframe overrides */
  [data-testid="stDataFrame"] {{ border: 1px solid {BORDER_CLR}; border-radius: 8px; overflow: hidden; }}
  .dataframe thead tr th {{
    background: {PANEL_BG} !important;
    color: {TEXT_MUTED} !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-bottom: 1px solid {BORDER_CLR} !important;
  }}
  .dataframe tbody tr td {{
    font-size: 0.82rem !important;
    font-family: {FONT_MONO};
    border-bottom: 1px solid {BORDER_CLR}22 !important;
  }}
  .dataframe tbody tr:hover td {{ background: {ACCENT_BLUE}0A !important; }}

  /* Plotly chart bg */
  .js-plotly-plot .plotly {{ background: transparent !important; }}

  /* Chat bubble */
  .chat-msg-user {{
    background: {ACCENT_BLUE}22;
    border: 1px solid {ACCENT_BLUE}44;
    border-radius: 8px 8px 2px 8px;
    padding: 0.65rem 0.9rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
    max-width: 85%;
    margin-left: auto;
  }}
  .chat-msg-ai {{
    background: {PANEL_BG};
    border: 1px solid {BORDER_CLR};
    border-radius: 8px 8px 8px 2px;
    padding: 0.65rem 0.9rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
    max-width: 85%;
    line-height: 1.6;
  }}

  /* Spinner override */
  .stSpinner > div {{ border-top-color: {ACCENT_BLUE} !important; }}

  /* Selectbox */
  [data-testid="stSelectbox"] > div > div {{
    background: {PANEL_BG};
    border: 1px solid {BORDER_CLR};
    border-radius: 6px;
    color: {TEXT_PRIMARY};
  }}

  /* Divider */
  hr {{ border-color: {BORDER_CLR} !important; margin: 1.5rem 0; }}

  /* Upload area */
  [data-testid="stFileUploader"] {{
    border: 2px dashed {BORDER_CLR};
    border-radius: 10px;
    padding: 1rem;
    background: {PANEL_BG};
    transition: border-color 0.2s;
  }}
  [data-testid="stFileUploader"]:hover {{ border-color: {ACCENT_BLUE}; }}

  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: {DARK_BG}; }}
  ::-webkit-scrollbar-thumb {{ background: {BORDER_CLR}; border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: {TEXT_MUTED}; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"   # Override in sidebar

TAX_CATEGORIES = [
    "Salary / Employment Income",
    "Business Income",
    "Rental Income",
    "Capital Gains",
    "Other Income",
    "Deductible Expense",
    "Non-Deductible Expense",
]

OPEX_CATEGORIES = [
    "Groceries & Supermarket",
    "Utilities & Bills",
    "Travel & Transport",
    "Dining & Restaurants",
    "Software & Subscriptions",
    "Healthcare & Pharmacy",
    "Entertainment & Leisure",
    "Education & Training",
    "Fuel & Auto",
    "Shopping & Retail",
    "Insurance",
    "Government & Taxes",
    "Transfers & Payments",
    "Bank Fees & Charges",
    "Uncategorized",
]

# Regex keyword → category mapping (offline fallback)
TAX_KEYWORD_MAP = {
    r"\b(salary|payroll|payslip|wages|stipend|employer|hris|hrms)\b":     "Salary / Employment Income",
    r"\b(invoice|client|freelance|consultancy|retainer|project)\b":        "Business Income",
    r"\b(rent|lease|tenant|property|landlord|accommodation)\b":            "Rental Income",
    r"\b(dividend|equity|mutual.?fund|brokerage|zerodha|groww|shares?)\b": "Capital Gains",
    r"\b(insurance|lic|mediclaim|premium|policy)\b":                       "Deductible Expense",
}

OPEX_KEYWORD_MAP = {
    r"\b(grocery|grocer|supermarket|dmart|bigbasket|blinkit|jiomart|zepto|swiggy.?instamart)\b": "Groceries & Supermarket",
    r"\b(electricity|bescom|water.?board|gas|broadband|wifi|airtel|jio|bsnl|vodafone|vi\b|act.?fibernet)\b": "Utilities & Bills",
    r"\b(ola|uber|rapido|irctc|railways|metro|bus|cab|flight|indigo|spicejet|airways|airline|mmt|makemytrip)\b": "Travel & Transport",
    r"\b(zomato|swiggy|restaurant|cafe|dine|bistro|kitchen|food.?court|hotel.?restaurant)\b": "Dining & Restaurants",
    r"\b(netflix|amazon.?prime|hotstar|spotify|adobe|github|slack|notion|figma|subscription|saas|app.?store|google.?play)\b": "Software & Subscriptions",
    r"\b(hospital|clinic|doctor|pharmacy|medplus|apollo|chemist|labs?|pathology|diagnostic)\b": "Healthcare & Pharmacy",
    r"\b(cinema|pvr|inox|multiplex|gaming|entertainment|bookmyshow|concert)\b": "Entertainment & Leisure",
    r"\b(school|college|university|course|udemy|coursera|tuition|education|book)\b": "Education & Training",
    r"\b(petrol|fuel|indian.?oil|hp.?petrol|bharat.?petrol|cng|ev.?charging)\b": "Fuel & Auto",
    r"\b(amazon|flipkart|myntra|ajio|meesho|nykaa|shopping|mall|retail)\b": "Shopping & Retail",
    r"\b(insurance|lic|star.?health|hdfc.?ergo|bajaj.?allianz|premium)\b": "Insurance",
    r"\b(income.?tax|gst|tds|government|mca|epfo|esi|nsdl|challan)\b": "Government & Taxes",
    r"\b(neft|rtgs|imps|upi|transfer|payment|settlement|remittance)\b":    "Transfers & Payments",
    r"\b(bank.?charge|annual.?fee|processing.?fee|late.?fee|penal|interest.?charged|emi)\b": "Bank Fees & Charges",
}


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA CONNECTIVITY
# ─────────────────────────────────────────────────────────────────────────────
def check_ollama() -> bool:
    """Ping Ollama health endpoint."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def ollama_generate(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 60) -> str:
    """Stream a completion from Ollama; return concatenated text."""
    try:
        payload = {"model": model, "prompt": prompt, "stream": True}
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout, stream=True)
        if response.status_code != 200:
            return ""
        full = []
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))
                full.append(chunk.get("response", ""))
                if chunk.get("done"):
                    break
        return "".join(full).strip()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# DATA EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_ALIASES = {
    "date":        ["date", "txn date", "transaction date", "value date", "posting date",
                    "trans date", "dt", "tdate", "book date"],
    "description": ["description", "narration", "particulars", "details", "remarks",
                    "transaction details", "trans desc", "trans narration", "note",
                    "narrative", "memo", "reference"],
    "debit":       ["debit", "dr", "withdrawal", "paid out", "debit amount", "amount debited",
                    "dr amount", "withdrawals", "debits", "expense", "dr."],
    "credit":      ["credit", "cr", "deposit", "paid in", "credit amount", "amount credited",
                    "cr amount", "deposits", "credits", "income", "cr."],
    "balance":     ["balance", "closing balance", "running balance", "avail balance",
                    "balance amount", "bal", "ledger balance", "available balance"],
}


def _match_column(col_name: str, target_group: str) -> bool:
    """Case-insensitive fuzzy match between a df column and a target group."""
    col_clean = col_name.lower().strip()
    for alias in COLUMN_ALIASES.get(target_group, []):
        if alias in col_clean or col_clean in alias:
            return True
        if fuzz.ratio(col_clean, alias) > 78:
            return True
    return False


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map arbitrary column names → [Date, Description, Debit, Credit, Balance]."""
    rename_map = {}
    for col in df.columns:
        for group in ["date", "description", "debit", "credit", "balance"]:
            if group not in [v.lower() for v in rename_map.values()] and _match_column(str(col), group):
                rename_map[col] = group.capitalize() if group != "description" else "Description"
                break

    df = df.rename(columns=rename_map)
    # Ensure mandatory columns exist
    for required in ["Date", "Description", "Debit", "Credit"]:
        if required not in df.columns:
            df[required] = np.nan if required in ["Debit", "Credit"] else ""
    if "Balance" not in df.columns:
        df["Balance"] = np.nan
    return df


def parse_amount(val) -> float:
    """Safely parse amount strings (handles commas, parentheses for negatives)."""
    if pd.isna(val) or val == "":
        return 0.0
    s = str(val).replace(",", "").replace(" ", "").strip()
    if s.startswith("(") and s.endswith(")"):   # (1000.00) → -1000
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_date(val) -> pd.Timestamp | None:
    """Parse diverse date strings into a uniform Timestamp."""
    if pd.isna(val) or val == "":
        return None
    s = str(val).strip()
    formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
        "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%d.%m.%Y",
        "%Y%m%d", "%d-%m-%y", "%d/%m/%y",
    ]
    for fmt in formats:
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    try:
        return pd.to_datetime(s, infer_datetime_format=True, dayfirst=True)
    except Exception:
        return None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize amounts, dates, drop empty rows."""
    df = df.copy()
    df["Debit"]  = df["Debit"].apply(parse_amount)
    df["Credit"] = df["Credit"].apply(parse_amount)
    df["Balance"] = df["Balance"].apply(parse_amount)
    df["Date"]   = df["Date"].apply(parse_date)
    df = df.dropna(subset=["Date"])
    df = df[df["Date"].notna()]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()
    # Drop rows where both debit and credit are 0 and description is empty
    df = df[~((df["Debit"] == 0) & (df["Credit"] == 0) & (df["Description"] == ""))]
    return df


def extract_from_pdf(file_obj) -> pd.DataFrame:
    """Extract tabular data from text-based PDFs using pdfplumber."""
    all_rows = []
    header = None
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                # Try to detect header row
                if header is None:
                    first_row = [str(c).strip() if c else "" for c in table[0]]
                    # Check if first row looks like a header
                    has_date = any(_match_column(c, "date") for c in first_row)
                    has_amount = any(_match_column(c, "debit") or _match_column(c, "credit") for c in first_row)
                    if has_date and has_amount:
                        header = first_row
                        data_rows = table[1:]
                    else:
                        data_rows = table
                else:
                    data_rows = table
                    # Skip rows that look like repeated headers
                    if data_rows and all(str(c).strip().lower() in [h.lower() for h in header]
                                         for c in data_rows[0] if c):
                        data_rows = data_rows[1:]

                for row in data_rows:
                    cleaned = [str(c).strip() if c else "" for c in row]
                    if any(cleaned):
                        all_rows.append(cleaned)

    if not all_rows:
        return pd.DataFrame()

    # Determine a target column width and normalise every row to it.
    # Real bank-statement PDFs frequently have inconsistent column counts
    # across rows/pages (merged cells, wrapped text, stray blank columns),
    # so checking only the first row's length is not reliable.
    if header:
        target_width = len(header)
    else:
        # Use the most common row length across all extracted rows
        from collections import Counter
        target_width = Counter(len(r) for r in all_rows).most_common(1)[0][0]

    def _normalise(row: list, width: int) -> list:
        if len(row) < width:
            return row + [""] * (width - len(row))
        if len(row) > width:
            return row[:width]
        return row

    all_rows = [_normalise(r, target_width) for r in all_rows]

    if header and len(header) == target_width:
        df = pd.DataFrame(all_rows, columns=header)
    else:
        df = pd.DataFrame(all_rows)
        df.columns = [f"col_{i}" for i in range(len(df.columns))]

    return df


def extract_from_csv(file_obj) -> pd.DataFrame:
    """Load CSV with smart encoding and delimiter detection."""
    try:
        df = pd.read_csv(file_obj, encoding="utf-8", thousands=",")
    except UnicodeDecodeError:
        file_obj.seek(0)
        df = pd.read_csv(file_obj, encoding="latin-1", thousands=",")
    return df


def extract_from_excel(file_obj) -> pd.DataFrame:
    """Load XLSX/XLS; try all sheets and pick the largest."""
    xl = pd.ExcelFile(file_obj)
    best = pd.DataFrame()
    for sheet in xl.sheet_names:
        candidate = xl.parse(sheet)
        if len(candidate) > len(best):
            best = candidate
    return best


def load_and_parse(uploaded_file) -> pd.DataFrame:
    """Entry point: dispatch to correct extractor, normalize, clean."""
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()
    file_obj  = io.BytesIO(raw_bytes)

    if name.endswith(".pdf"):
        raw_df = extract_from_pdf(file_obj)
        if raw_df.empty:
            st.warning("⚠️ pdfplumber found no tables — attempting OCR fallback (requires Tesseract).")
            raw_df = _ocr_fallback(raw_bytes)
    elif name.endswith(".csv"):
        file_obj.seek(0)
        raw_df = extract_from_csv(file_obj)
    elif name.endswith((".xlsx", ".xls")):
        raw_df = extract_from_excel(file_obj)
    else:
        st.error("Unsupported file format.")
        return pd.DataFrame()

    if raw_df.empty:
        st.error("Could not extract any tabular data from this file.")
        return pd.DataFrame()

    normalized = normalize_columns(raw_df)
    cleaned    = clean_dataframe(normalized)
    return cleaned


def _ocr_fallback(raw_bytes: bytes) -> pd.DataFrame:
    """Last-resort OCR via pytesseract for scanned PDFs."""
    try:
        import pytesseract
        from PIL import Image
        import pdf2image
        images = pdf2image.convert_from_bytes(raw_bytes, dpi=250)
        texts  = [pytesseract.image_to_string(img) for img in images]
        # Very simple line parser for OCR output
        rows = []
        for text in texts:
            for line in text.split("\n"):
                parts = re.split(r"\s{2,}", line.strip())
                if len(parts) >= 3:
                    rows.append(parts)
        if rows:
            max_cols = max(len(r) for r in rows)
            df = pd.DataFrame([r + [""] * (max_cols - len(r)) for r in rows])
            return df
    except Exception as e:
        st.warning(f"OCR fallback failed: {e}")
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# COUNTERPARTY EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
# Patterns to STRIP from narrations before entity extraction
NOISE_PATTERNS = [
    r"\bUPI\b[-/]?", r"\bNEFT\b[-/]?", r"\bRTGS\b[-/]?", r"\bIMPS\b[-/]?",
    r"\bPOS\b[-/]?", r"\bATM\b[-/]?", r"\bCHQ\b[-/]?", r"\bACH\b[-/]?",
    r"\bECS\b[-/]?", r"\bEMI\b[-/]?", r"\bSI\b[-/]?", r"\bDC\b[-/]?",
    r"\bCR\b[-/]?", r"\bDR\b[-/]?",
    r"[A-Z]{2,4}\d{6,}",          # Bank ref codes like HDFC123456
    r"\d{10,}",                    # Long phone / account numbers
    r"/\d{4,}/",                   # /1234/ style refs
    r"-\d{4,}-",                   # -1234- style refs
    r"\b(HDFC|ICICI|SBI|AXIS|KOTAK|YES|PNB|BOB|CANARA|UNION|UCO|IOB|BANDHAN|IDBI)\b",
    r"\b(BANK|LTD|PVT|INC|CORP|CO\.?)\b",
    r"\b(REF|TXN|TXR|TXNID|TRAN|TRANSF|MR|MRS|MS|DR)\b",
    r"\b\d{1,4}[-/]\d{1,2}[-/]\d{2,4}\b",  # dates embedded in narration
    r"[^A-Za-z0-9 @&\-_'.]+",    # special chars (keep readable chars)
]

NOISE_RE = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)

def extract_counterparty(narration: str) -> str:
    """
    Strip banking clutter from a narration string and return
    the likely counterparty name.
    """
    if not narration or not isinstance(narration, str):
        return "UNKNOWN"

    s = narration.upper()

    # Remove noise patterns
    s = NOISE_RE.sub(" ", s)

    # Collapse extra whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Tokenise: keep tokens that look like name segments (>1 char, mostly alpha)
    tokens = [t for t in s.split() if len(t) > 1 and re.search(r"[A-Z]", t)]

    if not tokens:
        return "UNKNOWN"

    # Heuristic: take first 3 meaningful tokens as the name
    name = " ".join(tokens[:3]).title().strip()
    return name if name else "UNKNOWN"


def deduplicate_counterparties(df: pd.DataFrame, threshold: int = 82) -> pd.DataFrame:
    """
    Use fuzzy matching to consolidate near-identical counterparty names
    (e.g. 'Revathi K' and 'Revathi Kumar' → same cluster).
    """
    if "Counterparty" not in df.columns:
        return df

    unique_parties = df["Counterparty"].unique().tolist()
    canonical_map  = {}

    for party in unique_parties:
        if party in canonical_map:
            continue
        # Find all close matches
        matches = process.extractBests(party, unique_parties, scorer=fuzz.token_sort_ratio,
                                        score_cutoff=threshold, limit=20)
        for match, score in matches:
            if match not in canonical_map:
                canonical_map[match] = party  # map variant → canonical

    df["Counterparty"] = df["Counterparty"].map(canonical_map).fillna(df["Counterparty"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORIZATION ENGINE (AI + Offline Fallback)
# ─────────────────────────────────────────────────────────────────────────────
def categorize_offline(description: str) -> tuple[str, str]:
    """
    Rule-based categorisation using keyword regex maps.
    Returns (tax_category, opex_category).
    """
    desc_lower = description.lower()

    tax_cat = "Other Income"
    for pattern, cat in TAX_KEYWORD_MAP.items():
        if re.search(pattern, desc_lower, re.IGNORECASE):
            tax_cat = cat
            break

    opex_cat = "Uncategorized"
    for pattern, cat in OPEX_KEYWORD_MAP.items():
        if re.search(pattern, desc_lower, re.IGNORECASE):
            opex_cat = cat
            break

    return tax_cat, opex_cat


def categorize_batch_ai(df: pd.DataFrame, model: str, ollama_ok: bool) -> pd.DataFrame:
    """
    Categorise up to N unique descriptions via Ollama in batch;
    fall back to offline rules for any failures.
    """
    df = df.copy()

    # Initialise with offline rules first (instant & always available)
    results = df["Description"].apply(lambda d: pd.Series(categorize_offline(d),
                                                           index=["TaxCategory", "OpexCategory"]))
    df["TaxCategory"]  = results["TaxCategory"]
    df["OpexCategory"] = results["OpexCategory"]

    if not ollama_ok:
        return df

    # Collect unique un-categorized or generic descriptions for AI refinement
    unique_descs = df["Description"].unique()[:80]  # cap at 80 to keep latency sane
    if len(unique_descs) == 0:
        return df

    prompt = f"""You are a financial categorisation engine. Classify each bank transaction description below.
Return ONLY a JSON array where each element has the keys:
  "desc" (original description, unchanged),
  "tax" (one of: {', '.join(TAX_CATEGORIES)}),
  "opex" (one of: {', '.join(OPEX_CATEGORIES)}).

Descriptions:
{json.dumps(list(unique_descs))}

Return ONLY the JSON array. No markdown, no explanation."""

    try:
        raw = ollama_generate(prompt, model=model, timeout=90)
        # Strip any markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        parsed = json.loads(raw)
        mapping = {item["desc"]: (item.get("tax", ""), item.get("opex", "")) for item in parsed}

        def apply_ai(row):
            ai = mapping.get(row["Description"])
            if ai:
                tc, oc = ai
                if tc in TAX_CATEGORIES:
                    row["TaxCategory"] = tc
                if oc in OPEX_CATEGORIES:
                    row["OpexCategory"] = oc
            return row

        df = df.apply(apply_ai, axis=1)
    except Exception:
        pass  # Silently fall back to offline rules already applied

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY / RED FLAG DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect multiple categories of suspicious activity and tag each row.
    Returns df with added columns: Flags (list), FlagCount (int), Severity.
    """
    df = df.copy()
    df["Flags"] = [[] for _ in range(len(df))]

    # 1. Round-number spike (whole numbers ≥ 1000 with zero cents)
    for idx, row in df.iterrows():
        amount = max(row["Debit"], row["Credit"])
        if amount >= 1000 and amount % 1 == 0 and int(amount) % 100 == 0:
            df.at[idx, "Flags"] = df.at[idx, "Flags"] + ["🔴 Round-Number Spike"]

    # 2. Potential duplicate entries (same date + same amount + same party, within tolerance)
    df["_amt"]   = df[["Debit", "Credit"]].max(axis=1)
    df["_date_str"] = df["Date"].dt.strftime("%Y-%m-%d")
    dup_mask = df.duplicated(subset=["_date_str", "_amt", "Counterparty"], keep=False) & (df["_amt"] > 0)
    for idx in df[dup_mask].index:
        df.at[idx, "Flags"] = df.at[idx, "Flags"] + ["🟡 Possible Duplicate"]

    # 3. Split transactions: 3+ transfers to SAME counterparty within 24 h
    df["_ts"] = pd.to_datetime(df["Date"])
    for party in df["Counterparty"].unique():
        if party == "UNKNOWN":
            continue
        sub = df[df["Counterparty"] == party].sort_values("_ts")
        for i, (idx, row) in enumerate(sub.iterrows()):
            window = sub[
                (sub["_ts"] >= row["_ts"]) &
                (sub["_ts"] <= row["_ts"] + pd.Timedelta(hours=24))
            ]
            if len(window) >= 3:
                for widx in window.index:
                    if "🟠 Split Transaction" not in df.at[widx, "Flags"]:
                        df.at[widx, "Flags"] = df.at[widx, "Flags"] + ["🟠 Split Transaction"]

    # 4. High-value weekend activity (Debit or Credit > 75th percentile on Sat/Sun)
    threshold = df["_amt"].quantile(0.75)
    weekend_mask = (df["Date"].dt.dayofweek >= 5) & (df["_amt"] > threshold) & (df["_amt"] > 0)
    for idx in df[weekend_mask].index:
        df.at[idx, "Flags"] = df.at[idx, "Flags"] + ["🟡 Weekend High-Value"]

    # 5. Midnight / off-hours (if time info exists — most bank statements don't have it, so skip safely)
    # Left as a hook for enriched data sets

    # Compute flag count + severity
    df["FlagCount"] = df["Flags"].apply(len)
    def severity(flags):
        if any("🔴" in f for f in flags):
            return "HIGH"
        if any("🟠" in f for f in flags):
            return "MEDIUM"
        if any("🟡" in f for f in flags):
            return "LOW"
        return ""
    df["Severity"] = df["Flags"].apply(severity)
    df["FlagsStr"] = df["Flags"].apply(lambda f: " · ".join(f) if f else "")

    # Cleanup temp columns
    df.drop(columns=["_amt", "_date_str", "_ts"], inplace=True, errors="ignore")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PARTY LEDGER ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def build_party_ledger(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two summary tables:
      Table A – Gross Party Volume  [Party | TxnCount | TotalDebit | TotalCredit]
      Table B – Net Position Pivot  [Party | NetFlow | Position]
    """
    grp = df.groupby("Counterparty", as_index=False).agg(
        TxnCount   = ("Description", "count"),
        TotalDebit = ("Debit",  "sum"),
        TotalCredit= ("Credit", "sum"),
    )
    grp = grp.sort_values("TxnCount", ascending=False)
    grp["TotalDebit"]  = grp["TotalDebit"].round(2)
    grp["TotalCredit"] = grp["TotalCredit"].round(2)

    # Table B — Net netting
    net = grp[["Counterparty", "TotalDebit", "TotalCredit"]].copy()
    net["NetFlow"] = (net["TotalCredit"] - net["TotalDebit"]).round(2)
    net["Position"] = net["NetFlow"].apply(
        lambda v: "✅ Net Receiver" if v > 0 else ("🔴 Net Payer" if v < 0 else "⚖️ Neutral")
    )
    net = net[["Counterparty", "NetFlow", "Position"]].sort_values("NetFlow", ascending=False)

    # Rename for display
    grp.rename(columns={
        "Counterparty": "Party Name",
        "TxnCount":     "Txn Count",
        "TotalDebit":   "Total Debit",
        "TotalCredit":  "Total Credit",
    }, inplace=True)
    net.rename(columns={
        "Counterparty": "Party Name",
        "NetFlow":      "Net Flow",
    }, inplace=True)

    return grp, net


def build_excel_report(df: pd.DataFrame, table_a: pd.DataFrame, table_b: pd.DataFrame) -> bytes:
    """
    Build a single multi-sheet Excel workbook mirroring the dashboard's tabs:
      - All Transactions           (full detail, master reference)
      - Tax Segregation            (TaxCategory summary)
      - OPEX Segregation           (OpexCategory summary)
      - Party Ledger               (gross volume per counterparty)
      - Party Netting              (net position per counterparty)
      - Risk & Anomaly Register    (flagged transactions only)
    Returns raw .xlsx bytes suitable for st.download_button.
    """
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    # ---- Sheet 1: All Transactions -----------------------------------------
    all_txns = df[["Date", "Description", "Counterparty", "Debit", "Credit", "Balance",
                   "TaxCategory", "OpexCategory", "FlagsStr", "Severity"]].copy()
    all_txns["Date"] = pd.to_datetime(all_txns["Date"]).dt.date
    all_txns.rename(columns={"FlagsStr": "Flags Triggered", "Severity": "Risk Level"}, inplace=True)

    # ---- Sheet 2: Tax Segregation -------------------------------------------
    tax_seg = (
        df.groupby("TaxCategory")
        .agg(TxnCount=("Description", "count"), TotalCredit=("Credit", "sum"), TotalDebit=("Debit", "sum"))
        .reset_index()
        .sort_values("TotalCredit", ascending=False)
    )
    tax_seg["NetFlow"] = tax_seg["TotalCredit"] - tax_seg["TotalDebit"]
    tax_seg.columns = ["Tax Category", "Txn Count", "Total Credit", "Total Debit", "Net Flow"]

    # ---- Sheet 3: OPEX Segregation -------------------------------------------
    opex_seg = (
        df.groupby("OpexCategory")
        .agg(TxnCount=("Description", "count"), TotalDebit=("Debit", "sum"))
        .reset_index()
        .sort_values("TotalDebit", ascending=False)
    )
    opex_total = opex_seg["TotalDebit"].sum()
    opex_seg["% of Spend"] = (opex_seg["TotalDebit"] / opex_total * 100).round(2) if opex_total else 0.0
    opex_seg.columns = ["OPEX Category", "Txn Count", "Total Spend", "% of Spend"]

    # ---- Sheet 6: Risk & Anomaly Register -----------------------------------
    flagged = df[df["FlagCount"] > 0][["Date", "Description", "Counterparty", "Debit", "Credit",
                                        "FlagsStr", "Severity"]].copy()
    flagged["Date"] = pd.to_datetime(flagged["Date"]).dt.date
    sev_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "": 3}
    flagged = flagged.sort_values(by="Severity", key=lambda s: s.map(sev_rank).fillna(3))
    flagged.rename(columns={"FlagsStr": "Flags Triggered", "Severity": "Risk Level"}, inplace=True)

    sheets = {
        "All Transactions":        all_txns,
        "Tax Segregation":         tax_seg,
        "OPEX Segregation":        opex_seg,
        "Party Ledger":            table_a,
        "Party Netting":           table_b,
        "Risk & Anomaly Register": flagged,
    }

    currency_headers = {"Debit", "Credit", "Balance", "Total Credit", "Total Debit",
                         "Net Flow", "Total Spend"}

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=name[:31], index=False)

        header_fill = PatternFill(start_color="1F2937", end_color="1F2937", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)

        for name, sheet_df in sheets.items():
            ws = writer.sheets[name[:31]]
            if ws.max_row < 1 or ws.max_column < 1:
                continue
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            max_row = ws.max_row

            for col_idx, col_name in enumerate(sheet_df.columns, start=1):
                col_letter = get_column_letter(col_idx)
                cell = ws.cell(row=1, column=col_idx)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center")

                max_len = max(
                    [len(str(col_name))] + [len(str(v)) for v in sheet_df[col_name].astype(str).values]
                    if len(sheet_df) else [len(str(col_name))]
                )
                ws.column_dimensions[col_letter].width = min(max_len + 3, 45)

                if col_name in currency_headers:
                    for row in range(2, max_row + 1):
                        ws.cell(row=row, column=col_idx).number_format = "#,##0.00"

    return buffer.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=11),
    xaxis=dict(gridcolor=BORDER_CLR, linecolor=BORDER_CLR, tickfont=dict(size=10)),
    yaxis=dict(gridcolor=BORDER_CLR, linecolor=BORDER_CLR, tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER_CLR),
    margin=dict(l=10, r=10, t=30, b=10),
)


def cashflow_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Monthly aggregated Debit / Credit bar + Net Balance line."""
    monthly = df.set_index("Date").resample("ME").agg(
        TotalDebit  = ("Debit",  "sum"),
        TotalCredit = ("Credit", "sum"),
    ).reset_index()
    monthly["Net"] = monthly["TotalCredit"] - monthly["TotalDebit"]
    monthly["Month"] = monthly["Date"].dt.strftime("%b %Y")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["Month"], y=monthly["TotalCredit"],
        name="Inflow (Credit)",
        marker_color=ACCENT_GREEN, marker_opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        x=monthly["Month"], y=monthly["TotalDebit"],
        name="Outflow (Debit)",
        marker_color=ACCENT_RED, marker_opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=monthly["Month"], y=monthly["Net"],
        name="Net Flow",
        mode="lines+markers",
        line=dict(color=ACCENT_BLUE, width=2.5),
        marker=dict(size=6, color=ACCENT_BLUE),
        yaxis="y2",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        title=dict(text="Monthly Cash Flow Overview", font=dict(size=13), x=0.01),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, tickfont=dict(size=10),
                    linecolor=BORDER_CLR),
        height=360,
    )
    return fig


def opex_donut_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart of OPEX category spend."""
    cat_spend = df.groupby("OpexCategory")["Debit"].sum().reset_index()
    cat_spend = cat_spend[cat_spend["Debit"] > 0].sort_values("Debit", ascending=False)
    colors = px.colors.qualitative.Bold[:len(cat_spend)]

    fig = go.Figure(go.Pie(
        labels=cat_spend["OpexCategory"],
        values=cat_spend["Debit"],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=DARK_BG, width=2)),
        textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>%{value:,.2f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        **{**PLOTLY_LAYOUT, "legend": {**PLOTLY_LAYOUT["legend"], "orientation": "v", "font": dict(size=10)}},
        title=dict(text="Expense Distribution by Category", font=dict(size=13), x=0.01),
        height=340,
        showlegend=True,
    )
    return fig


def tax_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: Tax category vs Credit vs Debit."""
    grp = df.groupby("TaxCategory").agg(
        Credit=("Credit", "sum"), Debit=("Debit", "sum")
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=grp["TaxCategory"], y=grp["Credit"],
                         name="Income", marker_color=ACCENT_GREEN, marker_opacity=0.85))
    fig.add_trace(go.Bar(x=grp["TaxCategory"], y=grp["Debit"],
                         name="Expense", marker_color=ACCENT_RED, marker_opacity=0.85))
    fig.update_layout(
        **{**PLOTLY_LAYOUT, "xaxis": {**PLOTLY_LAYOUT["xaxis"], "tickangle": -25, "tickfont": dict(size=9)}},
        barmode="group",
        title=dict(text="Tax Category Breakdown", font=dict(size=13), x=0.01),
        height=340,
    )
    return fig


def balance_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Running balance line chart."""
    df_bal = df[df["Balance"] != 0][["Date", "Balance"]].dropna()
    if df_bal.empty:
        # Reconstruct balance from debits/credits
        df["_RunningBal"] = (df["Credit"] - df["Debit"]).cumsum()
        df_bal = df[["Date", "_RunningBal"]].rename(columns={"_RunningBal": "Balance"})

    fig = go.Figure(go.Scatter(
        x=df_bal["Date"], y=df_bal["Balance"],
        fill="tozeroy", fillcolor=hex_to_rgba(ACCENT_BLUE),
        line=dict(color=ACCENT_BLUE, width=2),
        mode="lines",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Running Balance", font=dict(size=13), x=0.01),
        height=250,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, delta: str = "", delta_type: str = "neutral") -> str:
    delta_class = f"kpi-{delta_type}"
    delta_html  = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""
    return f"""
<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  {delta_html}
</div>"""


def fmt_inr(val: float) -> str:
    """Format a float as currency with thousands separator."""
    return f"₹{val:,.2f}" if val >= 0 else f"-₹{abs(val):,.2f}"


def section_header(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def render_styled_df(df: pd.DataFrame, height: int = 400):
    st.dataframe(df, use_container_width=True, height=height)


# ─────────────────────────────────────────────────────────────────────────────
# CHAT WITH YOUR STATEMENT (local LLM Q&A)
# ─────────────────────────────────────────────────────────────────────────────
def build_statement_context(df: pd.DataFrame) -> str:
    """Serialize a concise summary of the statement for the LLM context window."""
    total_credit = df["Credit"].sum()
    total_debit  = df["Debit"].sum()
    net          = total_credit - total_debit
    top_debits   = df.nlargest(5, "Debit")[["Date", "Description", "Debit"]].to_string(index=False)
    top_credits  = df.nlargest(5, "Credit")[["Date", "Description", "Credit"]].to_string(index=False)
    categories   = df.groupby("OpexCategory")["Debit"].sum().sort_values(ascending=False).head(8).to_string()
    flags        = df[df["FlagCount"] > 0][["Date", "Description", "FlagsStr"]].head(10).to_string(index=False)

    return f"""BANK STATEMENT SUMMARY
======================
Total Transactions : {len(df)}
Date Range         : {df['Date'].min().strftime('%Y-%m-%d')} → {df['Date'].max().strftime('%Y-%m-%d')}
Total Credits      : ₹{total_credit:,.2f}
Total Debits       : ₹{total_debit:,.2f}
Net Position       : ₹{net:,.2f}

TOP 5 DEBITS:
{top_debits}

TOP 5 CREDITS:
{top_credits}

EXPENSE BY CATEGORY:
{categories}

FLAGGED TRANSACTIONS:
{flags}
"""


def chat_with_statement(df: pd.DataFrame, model: str, ollama_ok: bool):
    """Render the chat interface for querying the statement."""
    section_header("💬 Chat with your Bank Statement")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    context = build_statement_context(df)

    # Display chat history
    for msg in st.session_state.chat_history:
        role_class = "chat-msg-user" if msg["role"] == "user" else "chat-msg-ai"
        role_label = "You" if msg["role"] == "user" else "🤖 AuditLens AI"
        st.markdown(
            f'<div style="font-size:0.7rem;color:{TEXT_MUTED};margin-bottom:2px">{role_label}</div>'
            f'<div class="{role_class}">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )

    if not ollama_ok:
        st.warning("⚠️ Ollama is offline. Chat requires Ollama running locally on port 11434.")
        return

    user_q = st.chat_input("Ask anything about your statement...")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})

        system = f"""You are a expert financial auditor and tax consultant.
You have access to the following bank statement summary:

{context}

Answer the user's question concisely and accurately based on this data.
If the question cannot be answered from the data provided, say so clearly.
Do NOT make up numbers."""

        full_prompt = f"{system}\n\nUser Question: {user_q}\n\nAnswer:"

        with st.spinner("Thinking…"):
            answer = ollama_generate(full_prompt, model=model, timeout=60)

        if not answer:
            answer = "⚠️ Could not get a response from Ollama. Please check the service is running."

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:0.5rem 0 1.5rem">
          <div style="font-size:1.35rem;font-weight:700;letter-spacing:-0.02em">
            🔍 AuditLens
          </div>
          <div style="font-size:0.75rem;color:{TEXT_MUTED};margin-top:0.2rem">
            Offline AI · Bank Statement Analyzer
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Ollama status
        ollama_ok = check_ollama()
        status_html = (
            f'<span class="badge badge-green">● Ollama Online</span>'
            if ollama_ok else
            f'<span class="badge badge-red">● Ollama Offline</span>'
        )
        st.markdown(f'<div style="margin-bottom:1rem">{status_html}</div>', unsafe_allow_html=True)

        selected_model = st.text_input("Ollama Model", value="llama3.2",
                                       help="Run `ollama pull llama3.2` to download.")

        st.divider()
        st.markdown(f'<div class="section-header">Upload Statement</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop a PDF, CSV, or Excel file",
            type=["pdf", "csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown(f'<div class="section-header">Analysis Options</div>', unsafe_allow_html=True)

        ai_categorise = st.toggle("AI Categorisation", value=ollama_ok,
                                   help="Uses Ollama if online; falls back to rule-based engine.")
        fuzzy_threshold = st.slider("Counterparty Fuzzy Match Threshold", 60, 95, 82,
                                     help="Higher = stricter matching. Lower = more aggressive deduplication.")
        round_number_threshold = st.number_input("Round-number Spike Min (₹)", value=1000, step=100)

        st.divider()
        st.caption(f"Privacy-first · Zero cloud APIs\nAll data stays on your machine.")

    # ── Main content ─────────────────────────────────────────────────────────
    if not uploaded:
        # Landing / hero state
        st.markdown(f"""
        <div style="
          display:flex; flex-direction:column; align-items:center; justify-content:center;
          min-height:65vh; text-align:center; padding:2rem;
        ">
          <div style="font-size:3.5rem;margin-bottom:1rem">🔍</div>
          <div style="font-size:2rem;font-weight:700;letter-spacing:-0.03em;margin-bottom:0.5rem">
            AuditLens
          </div>
          <div style="font-size:1rem;color:{TEXT_MUTED};max-width:480px;line-height:1.65;margin-bottom:2rem">
            Drop a bank statement (PDF, CSV, Excel) in the sidebar.<br>
            AuditLens extracts, categorises, and audits your transactions —<br>
            <strong>entirely offline</strong>, with zero cloud data exposure.
          </div>
          <div style="display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-green">✓ Local AI via Ollama</span>
            <span class="badge badge-blue">✓ PDF + CSV + Excel</span>
            <span class="badge badge-amber">✓ Party Ledger Netting</span>
            <span class="badge badge-red">✓ Anomaly Detection</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Parse uploaded file ───────────────────────────────────────────────────
    with st.spinner("Extracting and normalising statement data…"):
        df_raw = load_and_parse(uploaded)

    if df_raw.empty:
        st.error("❌ No usable data extracted from the file. Please check the format.")
        return

    # Counterparty extraction + deduplication
    with st.spinner("Extracting counterparties…"):
        df_raw["Counterparty"] = df_raw["Description"].apply(extract_counterparty)
        df_raw = deduplicate_counterparties(df_raw, threshold=fuzzy_threshold)

    # AI or rule-based categorisation
    with st.spinner("Categorising transactions…"):
        df = categorize_batch_ai(df_raw, model=selected_model, ollama_ok=(ollama_ok and ai_categorise))

    # Anomaly detection
    with st.spinner("Running anomaly detection…"):
        df = detect_anomalies(df)

    # Computed summary stats
    total_credit = df["Credit"].sum()
    total_debit  = df["Debit"].sum()
    net_flow     = total_credit - total_debit
    txn_count    = len(df)
    flag_count   = (df["FlagCount"] > 0).sum()
    date_range   = f"{df['Date'].min().strftime('%d %b %Y')} → {df['Date'].max().strftime('%d %b %Y')}"

    # Party ledger tables (computed once, reused in Tab 3 and the Excel export)
    table_a, table_b = build_party_ledger(df)

    # ── EXPORT ────────────────────────────────────────────────────────────────
    col_title, col_export = st.columns([5, 1.3])
    with col_title:
        st.markdown(f'<div style="padding-top:0.4rem;font-size:0.82rem;color:{TEXT_MUTED}">'
                    f'{txn_count} transactions · {date_range}</div>', unsafe_allow_html=True)
    with col_export:
        excel_bytes = build_excel_report(df, table_a, table_b)
        st.download_button(
            "⬇️ Export to Excel",
            data=excel_bytes,
            file_name=f"AuditLens_Report_{datetime.date.today().isoformat()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # ── TAB LAYOUT ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Executive Dashboard",
        "🧾 Tax & Expense Segregation",
        "🤝 Party Ledger & Netting",
        "🚨 Risk & Anomaly Register",
    ])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1: EXECUTIVE DASHBOARD
    # ═════════════════════════════════════════════════════════════════════════
    with tab1:
        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        net_type = "positive" if net_flow >= 0 else "negative"
        c1.markdown(kpi_card("Total Inflow",  fmt_inr(total_credit), f"Across {txn_count} txns", "neutral"), unsafe_allow_html=True)
        c2.markdown(kpi_card("Total Outflow", fmt_inr(total_debit),  f"{len(df[df['Debit']>0])} debits", "neutral"), unsafe_allow_html=True)
        c3.markdown(kpi_card("Net Position",  fmt_inr(net_flow),     "Credit − Debit", net_type), unsafe_allow_html=True)
        c4.markdown(kpi_card("Transactions",  str(txn_count),        date_range, "neutral"), unsafe_allow_html=True)
        c5.markdown(kpi_card("Flagged",       str(flag_count),       "Needs review" if flag_count else "All clear", "negative" if flag_count else "positive"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Cashflow chart
        col_chart, col_meta = st.columns([3, 1])
        with col_chart:
            st.plotly_chart(cashflow_trend_chart(df), use_container_width=True)
        with col_meta:
            section_header("Quick Stats")
            avg_debit  = df[df["Debit"]>0]["Debit"].mean()
            avg_credit = df[df["Credit"]>0]["Credit"].mean()
            max_debit  = df["Debit"].max()
            max_credit = df["Credit"].max()
            meta_rows  = {
                "Avg Debit":    fmt_inr(avg_debit)  if not np.isnan(avg_debit)  else "—",
                "Avg Credit":   fmt_inr(avg_credit) if not np.isnan(avg_credit) else "—",
                "Largest Debit":  fmt_inr(max_debit),
                "Largest Credit": fmt_inr(max_credit),
                "Unique Parties": str(df["Counterparty"].nunique()),
                "Date Span":     f"{(df['Date'].max()-df['Date'].min()).days}d",
            }
            for k, v in meta_rows.items():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:0.4rem 0;border-bottom:1px solid {BORDER_CLR};font-size:0.82rem">'
                    f'<span style="color:{TEXT_MUTED}">{k}</span>'
                    f'<span style="font-family:{FONT_MONO};font-weight:500">{v}</span></div>',
                    unsafe_allow_html=True,
                )

        # Balance trend
        section_header("Running Balance")
        st.plotly_chart(balance_trend_chart(df), use_container_width=True)

        # Raw statement preview
        section_header("Statement Preview (Last 30 Rows)")
        preview = df[["Date", "Description", "Counterparty", "Debit", "Credit", "Balance", "OpexCategory"]].tail(30).copy()
        preview["Date"] = preview["Date"].dt.strftime("%Y-%m-%d")
        render_styled_df(preview, height=350)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2: TAX & EXPENSE SEGREGATION
    # ═════════════════════════════════════════════════════════════════════════
    with tab2:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            section_header("Tax Category Summary")
            tax_summary = (
                df.groupby("TaxCategory")
                .agg(TxnCount=("Description","count"), TotalCredit=("Credit","sum"), TotalDebit=("Debit","sum"))
                .reset_index()
                .sort_values("TotalCredit", ascending=False)
            )
            tax_summary["TotalCredit"] = tax_summary["TotalCredit"].map(lambda x: f"₹{x:,.2f}")
            tax_summary["TotalDebit"]  = tax_summary["TotalDebit"].map(lambda x:  f"₹{x:,.2f}")
            tax_summary.columns = ["Tax Category", "# Txns", "Total Credit", "Total Debit"]
            render_styled_df(tax_summary, height=320)
            st.plotly_chart(tax_bar_chart(df), use_container_width=True)

        with col_right:
            section_header("OPEX / Expense Breakdown")
            opex_summary = (
                df.groupby("OpexCategory")
                .agg(TxnCount=("Description","count"), TotalDebit=("Debit","sum"))
                .reset_index()
                .sort_values("TotalDebit", ascending=False)
            )
            opex_total = opex_summary["TotalDebit"].sum()
            opex_summary["% of Spend"] = (opex_summary["TotalDebit"] / opex_total * 100).round(1).astype(str) + "%"
            opex_summary["TotalDebit"] = opex_summary["TotalDebit"].map(lambda x: f"₹{x:,.2f}")
            opex_summary.columns = ["OPEX Category", "# Txns", "Total Spend", "% of Spend"]
            render_styled_df(opex_summary, height=320)
            st.plotly_chart(opex_donut_chart(df), use_container_width=True)

        # Drill-down: click a category
        st.markdown("---")
        section_header("Category Drill-Down")
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            sel_tax = st.selectbox("Filter by Tax Category", ["(All)"] + sorted(df["TaxCategory"].unique().tolist()), key="tax_drill")
        with col_sel2:
            sel_opex = st.selectbox("Filter by OPEX Category", ["(All)"] + sorted(df["OpexCategory"].unique().tolist()), key="opex_drill")

        drill = df.copy()
        if sel_tax  != "(All)": drill = drill[drill["TaxCategory"]  == sel_tax]
        if sel_opex != "(All)": drill = drill[drill["OpexCategory"] == sel_opex]

        drill_display = drill[["Date","Description","Counterparty","Debit","Credit","TaxCategory","OpexCategory"]].copy()
        drill_display["Date"] = drill_display["Date"].dt.strftime("%Y-%m-%d")
        st.markdown(f'<div style="font-size:0.8rem;color:{TEXT_MUTED};margin-bottom:0.4rem">'
                    f'{len(drill_display)} transactions matching filters</div>', unsafe_allow_html=True)
        render_styled_df(drill_display, height=380)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3: PARTY LEDGER & NETTING
    # ═════════════════════════════════════════════════════════════════════════
    with tab3:
        section_header("Counterparty Analysis")

        # Auditor drill-down selector at the top
        all_parties = sorted(df[df["Counterparty"] != "UNKNOWN"]["Counterparty"].unique().tolist())
        selected_party = st.selectbox(
            "🔎 Select counterparty to drill into",
            ["— Select a party —"] + all_parties,
            key="party_drill",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"""
            <div style="background:{PANEL_BG};border:1px solid {BORDER_CLR};border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.75rem">
              <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:{TEXT_MUTED}">
                Table A — Gross Party Volume
              </div>
              <div style="font-size:0.78rem;color:{TEXT_MUTED};margin-top:0.2rem">
                Aggregate debit & credit per counterparty
              </div>
            </div>
            """, unsafe_allow_html=True)
            # Format currency
            ta_display = table_a.copy()
            ta_display["Total Debit"]  = ta_display["Total Debit"].map(lambda x: f"₹{x:,.2f}")
            ta_display["Total Credit"] = ta_display["Total Credit"].map(lambda x: f"₹{x:,.2f}")
            render_styled_df(ta_display, height=420)

        with col_b:
            st.markdown(f"""
            <div style="background:{PANEL_BG};border:1px solid {BORDER_CLR};border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.75rem">
              <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:{TEXT_MUTED}">
                Table B — Net Position Pivot
              </div>
              <div style="font-size:0.78rem;color:{TEXT_MUTED};margin-top:0.2rem">
                Net receiver vs net payer analysis
              </div>
            </div>
            """, unsafe_allow_html=True)
            tb_display = table_b.copy()
            tb_display["Net Flow"] = tb_display["Net Flow"].map(lambda x: f"₹{x:,.2f}")
            render_styled_df(tb_display, height=420)

        # Auditor drill-down mini table
        if selected_party and selected_party != "— Select a party —":
            st.markdown("---")
            party_txns = df[df["Counterparty"] == selected_party].copy()
            party_txns["Date"] = party_txns["Date"].dt.strftime("%Y-%m-%d")

            party_debit  = party_txns["Debit"].sum()
            party_credit = party_txns["Credit"].sum()
            party_net    = party_credit - party_debit

            section_header(f"Drill-Down: {selected_party}")
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(kpi_card("Transactions", str(len(party_txns)), "", "neutral"), unsafe_allow_html=True)
            m2.markdown(kpi_card("Total Debit",  fmt_inr(party_debit),  "", "negative" if party_debit>0 else "neutral"), unsafe_allow_html=True)
            m3.markdown(kpi_card("Total Credit", fmt_inr(party_credit), "", "positive" if party_credit>0 else "neutral"), unsafe_allow_html=True)
            m4.markdown(kpi_card("Net Flow",     fmt_inr(party_net),    "Receiver" if party_net>0 else "Payer", "positive" if party_net>=0 else "negative"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            drill_cols = ["Date","Description","Debit","Credit","Balance","TaxCategory","OpexCategory","FlagsStr"]
            render_styled_df(party_txns[drill_cols], height=320)

            # Mini balance chart for this party
            if len(party_txns) > 2:
                party_txns_sorted = party_txns.copy()
                party_txns_sorted["Date"] = pd.to_datetime(party_txns_sorted["Date"])
                party_txns_sorted = party_txns_sorted.sort_values("Date")
                party_txns_sorted["CumNet"] = (party_txns_sorted["Credit"] - party_txns_sorted["Debit"]).cumsum()

                fig_p = go.Figure(go.Scatter(
                    x=party_txns_sorted["Date"], y=party_txns_sorted["CumNet"],
                    fill="tozeroy",
                    fillcolor=hex_to_rgba(ACCENT_GREEN if party_net >= 0 else ACCENT_RED),
                    line=dict(color=ACCENT_GREEN if party_net >= 0 else ACCENT_RED, width=2),
                    mode="lines+markers",
                ))
                fig_p.update_layout(**PLOTLY_LAYOUT, height=200,
                                    title=dict(text=f"Cumulative Net Flow — {selected_party}", font=dict(size=12), x=0.01))
                st.plotly_chart(fig_p, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 4: RISK & ANOMALY REGISTER
    # ═════════════════════════════════════════════════════════════════════════
    with tab4:
        flagged = df[df["FlagCount"] > 0].copy()
        flagged_sorted = flagged.sort_values(["FlagCount", "Debit"], ascending=[False, False])

        # Summary badges
        high   = (flagged["Severity"] == "HIGH").sum()
        medium = (flagged["Severity"] == "MEDIUM").sum()
        low    = (flagged["Severity"] == "LOW").sum()

        col_h, col_m, col_l, col_tot = st.columns(4)
        col_h.markdown(kpi_card("🔴 High Risk",   str(high),        "Immediate review",    "negative"), unsafe_allow_html=True)
        col_m.markdown(kpi_card("🟠 Medium Risk", str(medium),      "Investigate",         "negative" if medium else "neutral"), unsafe_allow_html=True)
        col_l.markdown(kpi_card("🟡 Low Risk",    str(low),         "Monitor",             "neutral"), unsafe_allow_html=True)
        col_tot.markdown(kpi_card("Total Flagged", str(len(flagged)), f"of {txn_count} txns", "neutral"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Anomaly Register")

        if flagged_sorted.empty:
            st.success("✅ No anomalies detected in this statement.")
        else:
            # Severity filter
            sev_filter = st.multiselect("Filter by Severity", ["HIGH", "MEDIUM", "LOW"],
                                         default=["HIGH", "MEDIUM", "LOW"], key="sev_filter")
            filtered_flags = flagged_sorted[flagged_sorted["Severity"].isin(sev_filter)]

            display_cols = ["Date","Description","Counterparty","Debit","Credit","FlagsStr","Severity"]
            disp = filtered_flags[display_cols].copy()
            disp["Date"] = disp["Date"].dt.strftime("%Y-%m-%d")
            disp["Debit"]  = disp["Debit"].map(lambda x: f"₹{x:,.2f}" if x > 0 else "—")
            disp["Credit"] = disp["Credit"].map(lambda x: f"₹{x:,.2f}" if x > 0 else "—")
            disp.rename(columns={"FlagsStr": "Flags", "Severity": "⚠ Risk"}, inplace=True)
            render_styled_df(disp, height=380)

            # Flag type distribution mini chart
            flag_type_counts: dict[str, int] = {}
            for flags_list in flagged["Flags"]:
                for f in flags_list:
                    flag_type_counts[f] = flag_type_counts.get(f, 0) + 1

            if flag_type_counts:
                fig_flags = go.Figure(go.Bar(
                    x=list(flag_type_counts.values()),
                    y=list(flag_type_counts.keys()),
                    orientation="h",
                    marker_color=[ACCENT_RED if "🔴" in k else ACCENT_AMBER if "🟠" in k else "#D29922"
                                  for k in flag_type_counts.keys()],
                    marker_opacity=0.85,
                ))
                fig_flags.update_layout(
                    **{**PLOTLY_LAYOUT, "yaxis": {**PLOTLY_LAYOUT["yaxis"], "tickfont": dict(size=10)}},
                    height=220,
                    title=dict(text="Flag Type Distribution", font=dict(size=13), x=0.01),
                    xaxis_title="Count",
                )
                st.plotly_chart(fig_flags, use_container_width=True)

        st.markdown("---")
        # Chat interface
        chat_with_statement(df, model=selected_model, ollama_ok=ollama_ok)


if __name__ == "__main__":
    main()
