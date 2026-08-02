"""
AWS Bedrock Knowledge Base helper module.

Two access paths:
  - Verbatim  : retrieve_only()          -> raw Retrieve API, no generation, deterministic
  - Reasoning : retrieve_and_generate()  -> Nova Pro grounded answer
                *_with_citations()       -> answer + citations
                stream_*()               -> token stream (reasoning path only)

Config is read from the environment so nothing sensitive is hardcoded:
  AWS_REGION                 (default: xxxx)
  BEDROCK_KNOWLEDGE_BASE_ID  (default: xxxx)
  BEDROCK_INFERENCE_PROFILE  (default: apac.amazon.nova-pro-v1:0)
  AWS_ACCOUNT_ID             (optional; resolved via STS if absent)

Credentials come from the attached IAM role / default provider chain.
"""

import os
import logging

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

REGION = os.environ.get("AWS_REGION", "xxxx")
KNOWLEDGE_BASE_ID = os.environ.get("BEDROCK_KNOWLEDGE_BASE_ID", "xxxx")

# Nova Pro is NOT invokable on-demand via the bare foundation-model ARN outside
# us-east-1. In ap-southeast-2 it must go through the APAC cross-region
# inference profile, otherwise Retrieve & Generate raises a ValidationException.
INFERENCE_PROFILE_ID = os.environ.get(
    "BEDROCK_INFERENCE_PROFILE", "apac.amazon.nova-pro-v1:0"
)

DEFAULT_TOP_K = 5

# --- Reasoning-path generation prompt (Nova Pro needs explicit instructions) ---
# The template MUST contain $search_results$. Keep $output_format_instructions$
# or the model won't emit the citation markers that populate response["citations"].
# The reviewer's question is NOT a placeholder; it arrives via input.text.
GENERATION_PROMPT_TEMPLATE = """You are a knowledge assistant for an internal review team. Answer the reviewer's question using only the information in the search results below.

Instructions:
- Use only facts found in the search results. Do not add outside knowledge.
- If the search results do not contain the answer, say you could not find it in the source material. Do not guess.
- Reproduce names, dates, figures, defined terms, and section numbers exactly as they appear in the sources. Do not paraphrase these.
- Be concise and objective. Do not speculate beyond the sources.

Search results:
$search_results$

$output_format_instructions$"""

GENERATION_TEMPERATURE = 0.0  # deterministic / faithful
GENERATION_TOP_P = 1.0
GENERATION_MAX_TOKENS = 2048

# --- Planning Letter revision (Converse API, NO knowledge base retrieval) ---
# The reviewer pastes a draft; Nova Pro revises it against this system prompt.
# Edit this to encode your Planning Letter house style / revision rules.
PLANNING_LETTER_SYSTEM_PROMPT = """You are an editor for an internal review team that writes Planning Letters. Revise the reviewer's draft below.

Rules:
- Preserve every fact, figure, name, date, amount, and decision exactly. Do not invent, remove, or change factual content.
- Do not add new claims, recommendations, or information that is not in the draft.
- Improve clarity, structure, grammar, and tone. Use plain, professional, objective language suitable for the recipient.
- Keep the meaning of every sentence intact.

Formatting:
- Use clear, short paragraphs separated by a blank line. Never return one unbroken block of text.
- Use "-" bullet points for any list of items, conditions, eligibility criteria, or steps.
- Preserve the section order of the draft.
- Use plain paragraphs and simple bullets only. Do not use bold, headings, or other markup.

Return only the revised letter text, with no preamble, notes, or commentary."""

REVISION_TEMPERATURE = 0.2  # small fluency margin; set 0.0 for maximum determinism
REVISION_TOP_P = 0.9
REVISION_MAX_TOKENS = 4096


def _resolve_account_id() -> str:
    account_id = os.environ.get("AWS_ACCOUNT_ID")
    if account_id:
        return account_id
    return boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]


class BedrockKnowledgeBase:
    def __init__(
        self,
        knowledge_base_id: str = KNOWLEDGE_BASE_ID,
        region_name: str = REGION,
    ):
        self.knowledge_base_id = knowledge_base_id
        self.region_name = region_name
        self.client = boto3.client(
            "bedrock-agent-runtime",
            region_name=region_name,
        )
        # Separate client for direct model calls (Planning Letter revision).
        self.runtime_client = boto3.client(
            "bedrock-runtime",
            region_name=region_name,
        )
        self.model_arn = (
            f"arn:aws:bedrock:{region_name}:{_resolve_account_id()}"
            f":inference-profile/{INFERENCE_PROFILE_ID}"
        )

    # ---- Verbatim path -----------------------------------------------------

    def retrieve_only(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list:
        """Deterministic retrieval. No model, no generation."""
        try:
            response = self.client.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={"text": question},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {
                        "numberOfResults": top_k,
                        "overrideSearchType": "HYBRID",
                    }
                },
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            logger.exception("retrieve() failed")
            raise RuntimeError(f"Retrieval failed ({code})") from e

        return response.get("retrievalResults", [])

    # ---- Reasoning path ----------------------------------------------------

    def retrieve_and_generate(
        self,
        question: str,
        prompt_template: str = GENERATION_PROMPT_TEMPLATE,
    ) -> dict:
        try:
            return self.client.retrieve_and_generate(
                input={"text": question},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": self.knowledge_base_id,
                        "modelArn": self.model_arn,
                        "retrievalConfiguration": {
                            "vectorSearchConfiguration": {
                                "numberOfResults": DEFAULT_TOP_K,
                                "overrideSearchType": "HYBRID",
                            }
                        },
                        "generationConfiguration": {
                            "promptTemplate": {
                                "textPromptTemplate": prompt_template,
                            },
                            "inferenceConfig": {
                                "textInferenceConfig": {
                                    "temperature": GENERATION_TEMPERATURE,
                                    "topP": GENERATION_TOP_P,
                                    "maxTokens": GENERATION_MAX_TOKENS,
                                }
                            },
                        },
                    },
                },
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            logger.exception("retrieve_and_generate() failed")
            raise RuntimeError(f"Retrieve & generate failed ({code})") from e

    def retrieve_and_generate_with_citations(
        self,
        question: str,
        prompt_template: str = GENERATION_PROMPT_TEMPLATE,
    ) -> dict:
        response = self.retrieve_and_generate(question, prompt_template)
        return {
            "answer": response.get("output", {}).get("text", ""),
            "citations": response.get("citations", []),
        }

    def stream_retrieve_and_generate(
        self,
        question: str,
        prompt_template: str = GENERATION_PROMPT_TEMPLATE,
    ):
        """Reasoning path only. Never use this for the verbatim path."""
        try:
            response = self.client.retrieve_and_generate_stream(
                input={"text": question},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": self.knowledge_base_id,
                        "modelArn": self.model_arn,
                        "retrievalConfiguration": {
                            "vectorSearchConfiguration": {
                                "numberOfResults": DEFAULT_TOP_K,
                                "overrideSearchType": "HYBRID",
                            }
                        },
                        "generationConfiguration": {
                            "promptTemplate": {
                                "textPromptTemplate": prompt_template,
                            },
                            "inferenceConfig": {
                                "textInferenceConfig": {
                                    "temperature": GENERATION_TEMPERATURE,
                                    "topP": GENERATION_TOP_P,
                                    "maxTokens": GENERATION_MAX_TOKENS,
                                }
                            },
                        },
                    },
                },
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            logger.exception("retrieve_and_generate_stream() failed")
            raise RuntimeError(f"Streaming failed ({code})") from e

        for event in response["stream"]:
            if "output" in event:
                text = event["output"].get("text", "")
                if text:
                    yield text

    # ---- Planning Letter revision (direct model call, no retrieval) --------

    def revise_text(
        self,
        draft: str,
        system_prompt: str = PLANNING_LETTER_SYSTEM_PROMPT,
    ) -> str:
        """Revise pasted draft text with Nova Pro. Does NOT touch the KB."""
        try:
            response = self.runtime_client.converse(
                modelId=self.model_arn,
                system=[{"text": system_prompt}],
                messages=[
                    {"role": "user", "content": [{"text": draft}]},
                ],
                inferenceConfig={
                    "temperature": REVISION_TEMPERATURE,
                    "topP": REVISION_TOP_P,
                    "maxTokens": REVISION_MAX_TOKENS,
                },
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            logger.exception("converse() failed")
            raise RuntimeError(f"Revision failed ({code})") from e

        content = (
            response.get("output", {})
            .get("message", {})
            .get("content", [])
        )
        return content[0].get("text", "") if content else ""


# Singleton instance
_kb = BedrockKnowledgeBase()


def knowledge_base_search(question: str, top_k: int = DEFAULT_TOP_K) -> list:
    return _kb.retrieve_only(question, top_k=top_k)


def knowledge_base_answer(question: str) -> dict:
    return _kb.retrieve_and_generate(question)


def knowledge_base_answer_with_citations(question: str) -> dict:
    return _kb.retrieve_and_generate_with_citations(question)


def revise_text(
    draft: str,
    system_prompt: str = PLANNING_LETTER_SYSTEM_PROMPT,
) -> str:
    return _kb.revise_text(draft, system_prompt=system_prompt)


#----------------------------------------------------------

import html
import json
import os

import streamlit as st
import streamlit.components.v1 as components

from util.bedrock_kb import (
    knowledge_base_search,
    revise_text,
)
from util.markdown_normalize import normalize_markdown

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
    """Render one Access Letter result: header, Markdown body, source.
    Every source in this KB is FM-parsed to Markdown, so render formatted.

    normalize_markdown() is applied first because FM-parsed chunk text
    sometimes arrives with headings (###, ##) glued mid-line instead of on
    their own line — st.markdown() only recognizes an ATX heading when the
    '#' starts a line, so without this the marker shows up as literal text
    and nothing gets a paragraph break either.
    """
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
    render_markdown(normalize_markdown(r["text"]))
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



#--------------------------------------------------------

"""
markdown_normalize.py

Fixes FM-parsed chunk text where ATX headings ('### Section') and paragraph
breaks arrive without real newlines around them — the retrieved text still
contains the '#' markers, but because they're not at the start of a line,
Streamlit's markdown renderer (react-markdown + remark-gfm) treats them as
plain inline text instead of headings, and nothing separates into paragraphs.

Import normalize_markdown() and apply it to chunk text before render_markdown()
in chat_kb.py's render_access_result().
"""

import re

# Matches a heading marker (1-6 '#'s) followed by whitespace and the first
# character of the heading text, wherever it appears in the string.
_HEADING_RE = re.compile(r"(#{1,6}\s+\S)")


def normalize_markdown(text: str) -> str:
    if not text:
        return text

    # Defensive: literal backslash-n (two characters) -> real newline, in case
    # the text round-tripped through JSON encoding/decoding somewhere upstream
    # and the escape sequence never got converted back.
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n")

    def _break_before_heading(match: re.Match) -> str:
        preceding = match.string[: match.start()]
        if not preceding or preceding.endswith("\n"):
            return match.group(1)
        return "\n\n" + match.group(1)

    text = _HEADING_RE.sub(_break_before_heading, text)

    # Collapse 3+ consecutive newlines down to a single blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text
