import html
import json
import os

import streamlit as st
import streamlit.components.v1 as components

from util.bedrock_kb import (
    knowledge_base_search,
    revise_text,
)

st.set_page_config(
    page_title="Knowledge Base Chat",
    layout="wide",
)

st.title("Knowledge Base Chat")

# Full-width, readable, character-exact rendering for source text.
# Theme-safe colours (translucent grey works on both light and dark themes).
st.markdown(
    """
    <style>
    .verbatim-block {
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        line-height: 1.6;
        padding: 0.9rem 1.1rem;
        margin: 0.3rem 0 0.2rem 0;
        border-left: 3px solid #4c8bf5;
        border-radius: 6px;
        background: rgba(128, 128, 128, 0.10);
        font-size: 0.98rem;
    }
    .result-head {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-top: 0.6rem;
    }
    .result-head .rnum { font-weight: 600; font-size: 1.05rem; }
    .result-head .rscore { font-size: 0.8rem; opacity: 0.65; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Copy button lives in its own iframe because Streamlit strips <script> from
# st.markdown. __PAYLOAD__ is replaced with a JSON-encoded string literal.
_COPY_BUTTON = """
<style>
  body { margin: 0; padding: 0; }
  .bar { display: flex; justify-content: flex-end; }
  #cp {
    font: 13px -apple-system, "Segoe UI", Roboto, sans-serif;
    padding: 3px 12px; border-radius: 6px; cursor: pointer;
    border: none; background: #4c8bf5; color: #fff;
  }
  #cp:hover { background: #3b78e0; }
</style>
<div class="bar"><button id="cp">📋 Copy</button></div>
<script>
  const t = __PAYLOAD__;
  const b = document.getElementById('cp');
  b.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(t);
      const o = b.textContent; b.textContent = '✓ Copied';
      setTimeout(() => { b.textContent = o; }, 1500);
    } catch (e) { b.textContent = 'Copy failed'; }
  });
</script>
"""


def render_verbatim(text: str) -> None:
    """Full, character-exact text (no markdown reformatting) plus a copy button."""
    st.markdown(
        f'<div class="verbatim-block">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )
    # json.dumps -> valid JS string literal; guard against </script> breakout.
    payload = json.dumps(text).replace("</", "<\\/")
    components.html(_COPY_BUTTON.replace("__PAYLOAD__", payload), height=40)


def render_markdown(text: str) -> None:
    """Render Markdown (paragraphs/bullets/tables) + copy the raw text.
    Used for Planning Letter output and for structure-parsed source chunks."""
    st.markdown(text)
    payload = json.dumps(text).replace("</", "<\\/")
    components.html(_COPY_BUTTON.replace("__PAYLOAD__", payload), height=40)


def extract_source_uri(location: dict) -> str:
    """Handle every data-source type in the KB, not just S3."""
    source_type = location.get("type", "")
    return {
        "S3": location.get("s3Location", {}).get("uri"),
        "WEB": location.get("webLocation", {}).get("url"),
        "SHAREPOINT": location.get("sharePointLocation", {}).get("url"),
        "CONFLUENCE": location.get("confluenceLocation", {}).get("url"),
        "CUSTOM": location.get("customDocumentLocation", {}).get("id"),
    }.get(source_type) or "Unknown Source"


# Sources ingested with a structure-preserving parser (Route 1) — their chunks
# contain Markdown and are rendered formatted. Every other source stays escaped
# and verbatim. Set this to your authoritative-doc prefix AFTER you re-ingest it
# with the foundation-model parser, e.g.:
#   MARKDOWN_SOURCE_PREFIXES = ("s3://review-authoritative-docs-abc/",)
# Sources ingested with a structure-preserving parser (Route 1) — their chunks
# contain Markdown and are rendered formatted. Every other source stays escaped
# and verbatim. This is the authoritative-doc bucket, FM-parsed to Markdown.
MARKDOWN_SOURCE_PREFIXES: tuple = ("s3://ir-access-and-revocation/",)


def is_markdown_source(uri: str) -> bool:
    return bool(MARKDOWN_SOURCE_PREFIXES) and uri.startswith(MARKDOWN_SOURCE_PREFIXES)


def render_access_result(idx: int, r: dict) -> None:
    """Render one Access Letter result: header, verbatim/markdown body, source."""
    uri = r["source_uri"]
    name = os.path.basename(uri) if uri != "Unknown Source" else uri
    st.markdown(
        f'<div class="result-head">'
        f'<span class="rnum">Result {idx}</span>'
        f'<span class="rscore">relevance {r["score"]:.3f}</span>'
        f'<span class="rscore">· {html.escape(name)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if is_markdown_source(uri):
        render_markdown(r["text"])
    else:
        render_verbatim(r["text"])
    st.caption(f"Source: {uri}")
    st.divider()


# --- Conversation history -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_message(msg: dict) -> None:
    """Re-render one stored turn. Keyed on the turn's own kind, not the current
    Letter Type, so switching modes doesn't re-render past turns wrongly."""
    kind = msg["kind"]

    if kind == "text":
        st.write(msg["content"])

    elif kind == "access":
        results = msg["results"]
        if not results:
            st.warning("No results found.")
        else:
            # Top hit prominent; remaining results tucked behind an expander.
            render_access_result(1, results[0])
            rest = results[1:]
            if rest:
                with st.expander(f"More results ({len(rest)})"):
                    for i, r in enumerate(rest, start=2):
                        render_access_result(i, r)

    elif kind == "planning":
        revised = msg["revised"]
        if not revised:
            st.warning("No revised text was returned.")
        else:
            st.markdown("**Revised letter**")
            render_markdown(revised)

    elif kind == "error":
        st.error(msg["message"])


# --- Sidebar --------------------------------------------------------------
letter_type = st.sidebar.radio(
    "Letter Type",
    [
        "Access Letter",
        "Planning Letter",
    ],
)

if letter_type == "Access Letter":
    st.sidebar.caption(
        "Returns exact, verbatim text from the source documents."
    )
else:
    st.sidebar.caption(
        "Paste a draft below; it is revised against the Planning Letter "
        "guidance. No source lookup."
    )

st.sidebar.divider()
if st.sidebar.button("🗑️  New chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()


# --- Replay history -------------------------------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        render_message(m)

if not st.session_state.messages:
    st.caption("Ask a question or paste a draft to get started.")


# --- Handle new input -----------------------------------------------------
placeholder = (
    "Ask a question — returns verbatim source text…"
    if letter_type == "Access Letter"
    else "Paste your draft Planning Letter text to revise…"
)

if question := st.chat_input(placeholder):
    user_msg = {"role": "user", "kind": "text", "content": question}
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        render_message(user_msg)

    with st.chat_message("assistant"):
        if letter_type == "Access Letter":
            try:
                results = knowledge_base_search(question, top_k=3)
                access = [
                    {
                        "score": r.get("score", 0),
                        "text": r.get("content", {}).get("text", ""),
                        "source_uri": extract_source_uri(r.get("location", {})),
                    }
                    for r in results
                ]
                # Hierarchical chunking can surface the same parent passage via
                # several matching children — dedupe by text, keep first (best).
                seen, deduped = set(), []
                for a in access:
                    if a["text"] in seen:
                        continue
                    seen.add(a["text"])
                    deduped.append(a)
                assistant_msg = {
                    "role": "assistant",
                    "kind": "access",
                    "results": deduped,
                }
            except RuntimeError as e:
                assistant_msg = {
                    "role": "assistant",
                    "kind": "error",
                    "message": str(e),
                }
        else:
            with st.spinner("Revising…"):
                try:
                    revised = revise_text(question)
                    assistant_msg = {
                        "role": "assistant",
                        "kind": "planning",
                        "revised": revised,
                    }
                except RuntimeError as e:
                    assistant_msg = {
                        "role": "assistant",
                        "kind": "error",
                        "message": str(e),
                    }

        st.session_state.messages.append(assistant_msg)
        render_message(assistant_msg)
