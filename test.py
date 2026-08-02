

import os
import logging

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
KNOWLEDGE_BASE_ID = os.environ.get("BEDROCK_KNOWLEDGE_BASE_ID", "")


INFERENCE_PROFILE_ID = os.environ.get(
    "BEDROCK_INFERENCE_PROFILE", "apac.amazon.nova-pro-v1:0"
)

DEFAULT_TOP_K = 5

# --- Access Letter generation prompt (Nova Pro needs explicit instructions) ---
# This is what powers chat_kb.py's "Access Letter" mode: retrieve chunks from
# the KB, then have Nova Pro turn them into a grounded answer.
# The template MUST contain $search_results$. Keep $output_format_instructions$
# or the model won't emit the citation markers that populate response["citations"].
# The reviewer's question is NOT a placeholder; it arrives via input.text.
GENERATION_PROMPT_TEMPLATE = """

Search results:
$search_results$

$output_format_instructions$"""

GENERATION_TEMPERATURE = 0.0  # deterministic / faithful
GENERATION_TOP_P = 1.0
GENERATION_MAX_TOKENS = 2048

# --- Planning Letter revision (Converse API, NO knowledge base retrieval) ---
# The reviewer pastes a draft; Nova Pro revises it against this system prompt.
# Edit this to encode your Planning Letter house style / revision rules.
PLANNING_LETTER_SYSTEM_PROMPT = """."""

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

    # ---- Verbatim path (not wired into the chat UI) ------------------------

    def retrieve_only(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list:
        """Deterministic retrieval. No model, no generation.

        Kept for scripting/debugging. chat_kb.py's Access Letter mode uses
        retrieve_and_generate_with_citations() instead.
        """
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

    # ---- Access Letter path (Nova Pro, retrieve + generate) -----------------

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
        """Used by chat_kb.py's Access Letter mode: {"answer": str, "citations": [...]}."""
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
        """Access Letter path only. Never use this for the verbatim path."""
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
    """Verbatim retrieval only. Not used by the Access Letter UI anymore."""
    return _kb.retrieve_only(question, top_k=top_k)


def knowledge_base_answer(question: str) -> dict:
    return _kb.retrieve_and_generate(question)


def knowledge_base_answer_with_citations(question: str) -> dict:
    """Access Letter entry point: Nova Pro grounded answer + citations."""
    return _kb.retrieve_and_generate_with_citations(question)


def revise_text(
    draft: str,
    system_prompt: str = PLANNING_LETTER_SYSTEM_PROMPT,
) -> str:
    return _kb.revise_text(draft, system_prompt=system_prompt)






import html
import json
import os

import streamlit as st
import streamlit.components.v1 as components

from util.bedrock_kb import (
    knowledge_base_answer_with_citations,
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
    Used for Planning Letter output and for the Access Letter generated answer."""
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


def dedupe_sources(citations: list) -> list:
    """Flatten retrieve_and_generate() citations into a deduped, ordered list
    of source URIs (first-seen order)."""
    seen, sources = set(), []
    for citation in citations:
        for ref in citation.get("retrievedReferences", []):
            uri = extract_source_uri(ref.get("location", {}))
            if uri not in seen:
                seen.add(uri)
                sources.append(uri)
    return sources


def render_access_answer(answer: str, sources: list) -> None:
    """Render one Access Letter turn: Nova Pro's generated answer, then the
    deduped list of KB sources it was grounded on."""
    render_markdown(answer)
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for uri in sources:
                name = os.path.basename(uri) if uri != "Unknown Source" else uri
                st.caption(f"{name}  \n{uri}")


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
        answer = msg["answer"]
        if not answer:
            st.warning("No answer was returned.")
        else:
            render_access_answer(answer, msg.get("sources", []))

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
        "Answers your question with Nova Pro, grounded in the source "
        "documents. Sources are listed below each answer."
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
    "Ask a question — answered by Nova Pro from the source documents…"
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
            with st.spinner("Generating answer…"):
                try:
                    result = knowledge_base_answer_with_citations(question)
                    assistant_msg = {
                        "role": "assistant",
                        "kind": "access",
                        "answer": result.get("answer", ""),
                        "sources": dedupe_sources(result.get("citations", [])),
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
