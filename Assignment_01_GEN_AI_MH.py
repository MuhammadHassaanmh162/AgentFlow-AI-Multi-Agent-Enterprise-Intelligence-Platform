"""
Assignment 01 - Executive Strategic Article Generator
Tool: Streamlit + LangChain
Author: GEN_AI_MH
"""

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Executive Article Generator",
    page_icon="📋",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { font-family: 'Georgia', serif; }
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    .article-box {
        background: #f9f9f9;
        border-left: 4px solid #1a1a2e;
        padding: 1.5rem 2rem;
        border-radius: 6px;
        font-size: 1rem;
        line-height: 1.8;
        color: #222;
    }
    .stButton > button {
        background-color: #1a1a2e;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #16213e;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📋 Executive Article Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered strategic content for corporate teams</div>', unsafe_allow_html=True)
st.divider()

# ── Sidebar – API Key ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google Gemini API Key", type="password", placeholder="AIza...")
    st.caption("Your key is never stored.")
    st.divider()
    st.markdown("**How to use**")
    st.markdown("1. Enter your Gemini API key\n2. Type a topic\n3. Choose a tone\n4. Click **Generate**")

# ── Inputs ────────────────────────────────────────────────────────────────────
topic = st.text_input(
    "📌 Article Topic",
    placeholder="e.g. AI in Healthcare, Global Supply Chain Disruption",
)

tone = st.selectbox(
    "🎯 Select Tone",
    options=["Formal", "Concise", "Strategic"],
    index=2,
)

tone_descriptions = {
    "Formal":    "professional, structured, and authoritative",
    "Concise":   "brief, clear, and to the point",
    "Strategic": "forward-looking, insight-driven, and executive-focused",
}

# ── Generate Button ───────────────────────────────────────────────────────────
generate = st.button("🚀 Generate Article")

if generate:
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar.")
    elif not topic.strip():
        st.warning("⚠️ Please enter a topic before generating.")
    else:
        # ── Build LangChain chain ──────────────────────────────────────────
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            streaming=True,
            temperature=0.7,
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are a senior corporate strategy writer producing executive-level articles "
                    "for C-suite audiences. Write in a {tone} style: {tone_desc}. "
                    "Structure every article with:\n"
                    "1. Executive Summary (2-3 sentences)\n"
                    "2. Background & Context\n"
                    "3. Key Strategic Insights (3-5 bullet points)\n"
                    "4. Implications for Business Leaders\n"
                    "5. Recommendations & Next Steps\n"
                    "6. Conclusion\n\n"
                    "Use clear section headings. Be substantive and data-aware."
                ),
            ),
            ("human", "Write an executive article on the topic: {topic}"),
        ])

        chain = prompt | llm | StrOutputParser()

        # ── Stream Output ──────────────────────────────────────────────────
        st.divider()
        st.markdown(f"### 📄 Article: *{topic}*")
        st.caption(f"Tone: **{tone}** · Powered by Gemini 2.0 Flash")

        article_placeholder = st.empty()
        full_text = ""

        with st.spinner("Generating your article…"):
            for chunk in chain.stream({
                "tone": tone,
                "tone_desc": tone_descriptions[tone],
                "topic": topic,
            }):
                full_text += chunk
                article_placeholder.markdown(
                    f'<div class="article-box">{full_text}▌</div>',
                    unsafe_allow_html=True,
                )

        # Final render without cursor
        article_placeholder.markdown(
            f'<div class="article-box">{full_text}</div>',
            unsafe_allow_html=True,
        )

        st.success("✅ Article generated successfully!")

        # ── Download Button ────────────────────────────────────────────────
        st.download_button(
            label="⬇️ Download as .txt",
            data=full_text,
            file_name=f"{topic.replace(' ', '_')}_article.txt",
            mime="text/plain",
        )
