
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# 0.  Load environment variables
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# 1.  Page config  (MUST be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI History Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 2.  Custom CSS  — premium dark-indigo theme
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

    /* ── Root overrides ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0f172a;
        color: #e2e8f0;
    }

    /* ── Main container ── */
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }

    /* ── App header ── */
    .hero-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e3a5f 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(99,102,241,0.25);
        border: 1px solid rgba(99,102,241,0.3);
    }
    .hero-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a5b4fc, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-header p {
        color: #a5b4fc;
        margin: 0.4rem 0 0;
        font-size: 1rem;
    }

    /* ── Chat bubbles ── */
    .msg-user {
        background: linear-gradient(135deg, #3730a3, #4338ca);
        border-radius: 18px 18px 4px 18px;
        padding: 0.85rem 1.1rem;
        margin: 0.5rem 0 0.5rem 3rem;
        box-shadow: 0 4px 12px rgba(67,56,202,0.35);
        border-left: 3px solid #818cf8;
        animation: slideIn 0.3s ease;
    }
    .msg-assistant {
        background: linear-gradient(135deg, #1e293b, #1e2a3b);
        border-radius: 18px 18px 18px 4px;
        padding: 0.85rem 1.1rem;
        margin: 0.5rem 3rem 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border-left: 3px solid #6366f1;
        animation: slideIn 0.3s ease;
    }
    .msg-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .label-user  { color: #a5b4fc; }
    .label-ai    { color: #6366f1; }
    .msg-text    { color: #e2e8f0; line-height: 1.65; font-size: 0.95rem; }

    @keyframes slideIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0);   }
    }

    /* ── Input area ── */
    .stTextArea textarea {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
        transition: border-color 0.2s;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 1.25rem !important;
        transition: all 0.2s !important;
        border: none !important;
    }
    .stButton > button:first-child {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: white !important;
    }
    .stButton > button:first-child:hover {
        background: linear-gradient(135deg, #818cf8, #6366f1) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #a5b4fc;
    }

    /* ── Status badge ── */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(99,102,241,0.15);
        color: #818cf8;
        border: 1px solid rgba(99,102,241,0.3);
        margin-bottom: 0.5rem;
    }

    /* ── History summary box ── */
    .summary-box {
        background: #1e293b;
        border-left: 4px solid #6366f1;
        border-radius: 0 10px 10px 0;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-top: 0.5rem;
        font-style: italic;
    }

    /* ── Divider ── */
    hr { border-color: #1e293b; }

    /* ── Scrollable chat area ── */
    .chat-scroll {
        max-height: 520px;
        overflow-y: auto;
        padding-right: 4px;
    }
    .chat-scroll::-webkit-scrollbar { width: 5px; }
    .chat-scroll::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# 3.  LangChain setup
# ─────────────────────────────────────────────
MAX_TURNS      = 10   # messages kept before summarisation kicks in
SUMMARY_KEEP   = 4    # newest messages to keep after summarisation


@st.cache_resource(show_spinner=False)
def build_chain():
    """Build and cache the RAG chain (expensive initialisation)."""
    llm       = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    retriever = WikipediaRetriever()

    rag_prompt = ChatPromptTemplate.from_template(
        """You are an expert AI History Assistant for the general public.
Use ONLY the provided Wikipedia passages to answer — do not hallucinate.
If the passages do not contain the answer, say so politely.

Conversation so far (for context):
{chat_history}

User question: {user_question}

Wikipedia passages:
{retrieved_passages}

Give a well-structured, informative answer.
"""
    )

    rag_chain = (
        {
            "retrieved_passages": (
                lambda x: x["user_question"]
            )
            | retriever
            | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "user_question":  lambda x: x["user_question"],
            "chat_history":   lambda x: x["chat_history"],
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return llm, rag_chain


# ─────────────────────────────────────────────
# 4.  Session state initialisation
# ─────────────────────────────────────────────
if "messages"   not in st.session_state:
    st.session_state.messages   = []   # list of {"role": ..., "content": ...}
if "summary"    not in st.session_state:
    st.session_state.summary    = ""   # running summary of old messages
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0


# ─────────────────────────────────────────────
# 5.  Helper: format history for the LLM
# ─────────────────────────────────────────────
def build_chat_history_text():
    """Return conversation history as a readable string for the prompt."""
    parts = []
    if st.session_state.summary:
        parts.append(f"[Earlier conversation summary]\n{st.session_state.summary}")
    for m in st.session_state.messages:
        role  = "User"      if m["role"] == "user"      else "Assistant"
        parts.append(f"{role}: {m['content']}")
    return "\n".join(parts) if parts else "No prior conversation."


# ─────────────────────────────────────────────
# 6.  History Management — Summarisation
# ─────────────────────────────────────────────
def summarise_history(llm, force: bool = False):
    """
    Auto-compress old messages via LLM summarisation.
    Keeps the SUMMARY_KEEP newest messages intact.
    Triggered automatically when len(messages) > MAX_TURNS, or on demand.
    """
    msgs = st.session_state.messages
    if not force and len(msgs) <= MAX_TURNS:
        return False   # nothing to do

    # Messages to compress  (everything except the newest SUMMARY_KEEP)
    to_compress = msgs[:-SUMMARY_KEEP] if len(msgs) > SUMMARY_KEEP else msgs
    if not to_compress:
        return False

    # Build a text block for the LLM
    dialogue = "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in to_compress
    )
    old_summary = st.session_state.summary
    prefix = f"Previous summary:\n{old_summary}\n\n" if old_summary else ""

    summary_prompt = (
        f"{prefix}"
        f"Conversation to summarise:\n{dialogue}\n\n"
        "Write a concise bullet-point summary of the key topics and facts discussed. "
        "Be brief but capture all important history assistant facts."
    )

    new_summary = llm.invoke(summary_prompt).content.strip()
    st.session_state.summary  = new_summary
    # Keep only the newest messages
    st.session_state.messages = msgs[-SUMMARY_KEEP:] if len(msgs) > SUMMARY_KEEP else []
    return True


# ─────────────────────────────────────────────
# 7.  Streaming ask function
# ─────────────────────────────────────────────
def stream_answer(rag_chain, user_question: str):
    """Stream tokens from the RAG chain; yields str chunks."""
    chat_history = build_chat_history_text()
    payload = {"user_question": user_question, "chat_history": chat_history}
    for chunk in rag_chain.stream(payload):
        yield chunk


# ─────────────────────────────────────────────
# 8.  UI — Hero header
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-header">
        <h1>🏛️ AI History Assistant</h1>
        <p>Ask me anything about history — powered by Wikipedia RAG &amp; Groq LLaMA&nbsp;3</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# 9.  Sidebar — History management panel
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗂️ History Management")
    st.markdown("---")

    llm, rag_chain = build_chain()

    # Stats
    n_msgs = len(st.session_state.messages)
    st.markdown(
        f'<span class="status-badge">💬 {n_msgs} messages · '
        f'Turn #{st.session_state.turn_count}</span>',
        unsafe_allow_html=True,
    )

    # Manual summarise button
    if st.button("🗜️ Summarise History Now", use_container_width=True):
        with st.spinner("Summarising …"):
            did_it = summarise_history(llm, force=True)
        if did_it:
            st.success("History compressed into summary ✅")
        else:
            st.info("Nothing to summarise yet.")
        st.rerun()

    # Clear everything
    if st.button("🗑️ Clear All History", use_container_width=True):
        st.session_state.messages   = []
        st.session_state.summary    = ""
        st.session_state.turn_count = 0
        st.rerun()

    st.markdown("---")
    st.markdown("### 📜 Running Summary")

    if st.session_state.summary:
        st.markdown(
            f'<div class="summary-box">{st.session_state.summary}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("_No summary yet. Chat a bit and the assistant will auto-compress old messages._")

    st.markdown("---")
    st.markdown("### 💡 How It Works")
    st.markdown(
        """
- **RAG**: retrieves live Wikipedia passages for every question
- **Streaming**: tokens appear word-by-word
- **History**: full conversation context is passed to the LLM
- **Summarisation**: old turns are auto-compressed when history exceeds **10 messages**
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# 10.  Main area — Chat window
# ─────────────────────────────────────────────
chat_col, _ = st.columns([1, 0.001])

with chat_col:
    # Render all existing messages
    chat_html = '<div class="chat-scroll">'
    for m in st.session_state.messages:
        if m["role"] == "user":
            chat_html += (
                f'<div class="msg-user">'
                f'  <div class="msg-label label-user">🧑 You</div>'
                f'  <div class="msg-text">{m["content"]}</div>'
                f'</div>'
            )
        else:
            chat_html += (
                f'<div class="msg-assistant">'
                f'  <div class="msg-label label-ai">🏛️ History Assistant</div>'
                f'  <div class="msg-text">{m["content"]}</div>'
                f'</div>'
            )
    chat_html += "</div>"

    if st.session_state.messages:
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div style="text-align:center; padding:3rem; color:#475569;">
                <div style="font-size:3rem;">🏛️</div>
                <p style="font-size:1.1rem; margin-top:1rem; color:#94a3b8;">
                    Ask me anything about history!<br>
                    <span style="font-size:0.85rem; color:#64748b;">
                    e.g. "What were the key events of the Battle of Thermopylae?"
                    </span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# 11.  Input area
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area(
        label="Your question",
        placeholder="Ask a history question… (e.g. Who was Napoleon Bonaparte?)",
        height=90,
        label_visibility="collapsed",
    )
    col_btn1, col_btn2, col_spacer = st.columns([1, 1, 4])
    with col_btn1:
        submitted = st.form_submit_button("🚀 Ask", use_container_width=True)
    with col_btn2:
        st.form_submit_button(
            "🔄 Refresh", use_container_width=True
        )   # triggers rerun to refresh stats

# ─────────────────────────────────────────────
# 12.  Handle submission + streaming
# ─────────────────────────────────────────────
if submitted and user_input.strip():
    question = user_input.strip()

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.turn_count += 1

    # Display thinking indicator
    with st.status("🔍 Searching Wikipedia & generating answer …", expanded=False):
        st.write("Retrieving relevant passages …")

    # Stream the answer
    answer_placeholder = st.empty()
    full_answer = ""

    answer_placeholder.markdown(
        '<div class="msg-assistant">'
        '  <div class="msg-label label-ai">🏛️ History Assistant</div>'
        '  <div class="msg-text">▌</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    for chunk in stream_answer(rag_chain, question):
        full_answer += chunk
        answer_placeholder.markdown(
            f'<div class="msg-assistant">'
            f'  <div class="msg-label label-ai">🏛️ History Assistant</div>'
            f'  <div class="msg-text">{full_answer}▌</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Final render (remove cursor)
    answer_placeholder.markdown(
        f'<div class="msg-assistant">'
        f'  <div class="msg-label label-ai">🏛️ History Assistant</div>'
        f'  <div class="msg-text">{full_answer}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_answer})

    # ── Auto-summarise if history is getting long ──
    if len(st.session_state.messages) > MAX_TURNS:
        with st.spinner("🗜️ Auto-compressing conversation history …"):
            summarise_history(llm, force=False)

    # Rerun to refresh chat window and sidebar stats
    st.rerun()
