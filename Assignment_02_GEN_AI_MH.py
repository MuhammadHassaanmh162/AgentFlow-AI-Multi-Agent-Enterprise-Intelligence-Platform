
# ─────────────────────────────────────────────────────────────────
# AI System for Pharmaceutical Drug Information Extraction
# WHO Drug Report Analyzer — Assignment 02
# ─────────────────────────────────────────────────────────────────

import json
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# ── Load environment variables (.env must contain GROQ_API_KEY) ──
load_dotenv()

# ── Page config (must be FIRST Streamlit call) ───────────────────
st.set_page_config(
    page_title="WHO Drug Report Analyzer",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# Custom CSS — clean medical / dark-teal theme
# ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0d1b2a;
        color: #e2e8f0;
    }
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }

    /* ── Hero header ── */
    .hero-header {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #0a3d47 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 32px rgba(0,200,180,0.2);
        border: 1px solid rgba(0,200,180,0.25);
    }
    .hero-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00c8b4, #7efff5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-header p { color: #80cbc4; margin: 0.35rem 0 0; font-size: 0.95rem; }

    /* ── Info card ── */
    .info-card {
        background: linear-gradient(135deg, #132f3e, #0d2233);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        border-left: 4px solid #00c8b4;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,200,180,0.2);
    }
    .card-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #00c8b4;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .card-body { color: #cbd5e1; font-size: 0.93rem; line-height: 1.7; }

    /* ── Summary box ── */
    .summary-box {
        background: linear-gradient(135deg, #1a3a4a, #0d2233);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(0,200,180,0.3);
        box-shadow: 0 4px 20px rgba(0,200,180,0.15);
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.75;
        margin-bottom: 1.5rem;
    }
    .summary-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        color: #7efff5;
        margin-bottom: 0.6rem;
    }

    /* ── Upload area ── */
    [data-testid="stFileUploader"] {
        background: #132f3e !important;
        border: 2px dashed rgba(0,200,180,0.35) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s !important;
        border: none !important;
    }
    .stButton > button:first-child {
        background: linear-gradient(135deg, #00b4a0, #007a6e) !important;
        color: white !important;
    }
    .stButton > button:first-child:hover {
        background: linear-gradient(135deg, #00c8b4, #00b4a0) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(0,200,180,0.4) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0d1b2a !important;
        border-right: 1px solid #1a3a4a;
    }
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 { color: #00c8b4; }

    /* ── Divider ── */
    hr { border-color: #1a3a4a; }

    /* ── Status badge ── */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        font-size: 0.73rem;
        font-weight: 600;
        background: rgba(0,200,180,0.12);
        color: #00c8b4;
        border: 1px solid rgba(0,200,180,0.3);
        margin-bottom: 0.5rem;
    }

    /* ── Text areas ── */
    .stTextArea textarea {
        background-color: #132f3e !important;
        color: #e2e8f0 !important;
        border: 1px solid #1a4a5a !important;
        border-radius: 10px !important;
    }

    /* ── Spinners / info ── */
    .stSpinner > div { border-color: #00c8b4 !important; }

    /* ── Section divider ── */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #7efff5;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(0,200,180,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────
# LLM setup
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)


# ─────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert pharmaceutical analyst working for the World Health Organization (WHO).
Your task is to carefully read the drug report below and extract the key information.

Return your response as a VALID JSON object with EXACTLY these keys:
{{
  "drug_name": "...",
  "active_ingredients": "...",
  "indications": "...",
  "dosage_information": "...",
  "side_effects": "...",
  "contraindications": "..."
}}

Rules:
- Use "Not mentioned" if any field is not found in the report.
- Be concise but complete.
- Do NOT add any text outside the JSON object.

Drug Report:
{report_text}
"""
)

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """You are a medical documentation specialist.
Read the following pharmaceutical drug report and write a clear, concise summary (3-5 sentences)
that highlights the most important information for a healthcare professional.
Focus on what the drug is, what it treats, and key safety considerations.

Drug Report:
{report_text}

Summary:"""
)


# ─────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────
def extract_drug_info(report_text: str, llm) -> dict:
    """Extract structured drug information from report text."""
    chain = EXTRACTION_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"report_text": report_text}).strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "drug_name": "Parse error",
            "active_ingredients": raw,
            "indications": "—",
            "dosage_information": "—",
            "side_effects": "—",
            "contraindications": "—",
        }


def generate_summary(report_text: str, llm) -> str:
    """Generate a short narrative summary of the drug report."""
    chain = SUMMARY_PROMPT | llm | StrOutputParser()
    return chain.invoke({"report_text": report_text}).strip()


# ─────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "report_name" not in st.session_state:
    st.session_state.report_name = None


# ─────────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-header">
        <h1>💊 WHO Drug Report Analyzer</h1>
        <p>AI-powered pharmaceutical information extraction system &nbsp;·&nbsp; Powered by Groq LLaMA&nbsp;3</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 How to Use")
    st.markdown("---")
    st.markdown(
        """
1. **Upload** a drug report (`.txt` or `.pdf* `)
2. Click **Analyze Report**
3. View the extracted information and AI summary

---
**Extracted Fields**
- 🏷️ Drug Name
- 🧪 Active Ingredients
- 🩺 Indications
- 💉 Dosage Information
- ⚠️ Side Effects
- 🚫 Contraindications
- 📝 AI Summary

---
> *PDF text extraction requires the report to be a text-based PDF.
        """
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        """
This tool helps healthcare professionals quickly extract and understand key information
from long pharmaceutical drug reports, improving accessibility of WHO drug documentation.
        """
    )

    if st.session_state.results:
        st.markdown("---")
        st.markdown(
            '<span class="status-badge">✅ Analysis Complete</span>',
            unsafe_allow_html=True,
        )
        if st.button("🔄 Analyze New Report", use_container_width=True):
            st.session_state.results = None
            st.session_state.summary = None
            st.session_state.report_name = None
            st.rerun()


# ─────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────
if not st.session_state.results:
    # ── Upload section ────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">📂 Upload Drug Report</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a pharmaceutical drug report (TXT or PDF)",
        type=["txt", "pdf"],
        help="Upload a WHO or similar drug report document.",
    )

    # Optional: paste text directly
    with st.expander("✏️ Or paste report text directly"):
        pasted_text = st.text_area(
            "Paste drug report text here:",
            height=220,
            placeholder="Paste the full drug report text here...",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        analyze_btn = st.button("🔬 Analyze Report", use_container_width=True)

    if analyze_btn:
        report_text = ""
        file_name = "Pasted Text"

        # Initialize LLM here (lazy) — so app loads cleanly without API key
        try:
            llm = get_llm()
        except Exception as e:
            st.error(
                "⚠️ Could not connect to Groq. "
                "Please make sure your `.env` file contains a valid `GROQ_API_KEY`.\n\n"
                f"Error: {e}"
            )
            st.stop()

        # -- Read uploaded file --
        if uploaded_file is not None:
            file_name = uploaded_file.name
            if uploaded_file.type == "text/plain":
                report_text = uploaded_file.read().decode("utf-8", errors="ignore")
            else:
                # PDF: try pdfplumber, fall back gracefully
                try:
                    import pdfplumber
                    with pdfplumber.open(uploaded_file) as pdf:
                        report_text = "\n".join(
                            page.extract_text() or "" for page in pdf.pages
                        )
                except ImportError:
                    st.error(
                        "📦 `pdfplumber` is not installed. "
                        "Run `pip install pdfplumber` to enable PDF support, "
                        "or paste the text directly."
                    )
                    st.stop()

        elif pasted_text.strip():
            report_text = pasted_text.strip()

        else:
            st.warning("⚠️ Please upload a file or paste report text first.")
            st.stop()

        if not report_text.strip():
            st.error("Could not extract text from the uploaded file. Try pasting the text directly.")
            st.stop()

        # -- Run AI analysis --
        with st.spinner("🔬 Analyzing drug report… this may take a few seconds"):
            drug_info = extract_drug_info(report_text, llm)

        with st.spinner("📝 Generating summary…"):
            summary = generate_summary(report_text, llm)

        st.session_state.results = drug_info
        st.session_state.summary = summary
        st.session_state.report_name = file_name
        st.rerun()

else:
    # ── Results section ───────────────────────────────────────────
    info = st.session_state.results
    report_name = st.session_state.report_name

    st.markdown(
        f'<span class="status-badge">📄 Report: {report_name}</span>',
        unsafe_allow_html=True,
    )

    # ── AI Summary ───────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">📝 AI-Generated Summary</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="summary-box">
            <div class="summary-title">🤖 Report Summary</div>
            {st.session_state.summary}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Extracted Information ─────────────────────────────────────
    st.markdown(
        '<div class="section-header">🔍 Extracted Drug Information</div>',
        unsafe_allow_html=True,
    )

    FIELDS = [
        ("drug_name",          "🏷️",  "Drug Name"),
        ("active_ingredients", "🧪",  "Active Ingredients"),
        ("indications",        "🩺",  "Indications"),
        ("dosage_information", "💉",  "Dosage Information"),
        ("side_effects",       "⚠️",  "Side Effects"),
        ("contraindications",  "🚫",  "Contraindications"),
    ]

    # Display in 2-column grid
    col_a, col_b = st.columns(2)
    for i, (key, icon, label) in enumerate(FIELDS):
        value = info.get(key, "Not mentioned")
        target_col = col_a if i % 2 == 0 else col_b
        with target_col:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="card-title">{icon} {label}</div>
                    <div class="card-body">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Action buttons ────────────────────────────────────────────
    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        if st.button("🔄 Analyze Another Report", use_container_width=True):
            st.session_state.results = None
            st.session_state.summary = None
            st.session_state.report_name = None
            st.rerun()
    with col2:
        # Download results as JSON
        result_json = json.dumps(
            {
                "report": report_name,
                "extracted_info": info,
                "summary": st.session_state.summary,
            },
            indent=2,
        )
        st.download_button(
            label="📥 Download Results (JSON)",
            data=result_json,
            file_name="drug_analysis_results.json",
            mime="application/json",
            use_container_width=True,
        )
