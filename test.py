st.markdown(
    """
    <style>
    .verbatim-block {
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        line-height: 1.6;
        padding: 0.9rem 1.1rem;
        margin: 0.3rem 0 0.6rem 0;
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
 
 
def render_verbatim(text: str) -> None:
    """Show source text in full and character-exact — no markdown reformatting."""
    st.markdown(
        f'<div class="verbatim-block">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )
 
 
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
 
 
placeholder = (
    "Ask a question — returns verbatim source text…"
    if letter_type == "Access Letter"
    else "Paste your draft Planning Letter text to revise…"
)
question = st.chat_input(placeholder)
 
if question:
    st.chat_message("user").write(question)
 
    with st.chat_message("assistant"):
 
        # ---- Access Letter: verbatim source text only ----------------------
        if letter_type == "Access Letter":
            try:
                results = knowledge_base_search(question, top_k=5)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
 
            if not results:
                st.warning("No results found.")
 
            for idx, result in enumerate(results, start=1):
                score = result.get("score", 0)
                text = result.get("content", {}).get("text", "")
                source_uri = extract_source_uri(result.get("location", {}))
                source_name = (
                    os.path.basename(source_uri)
                    if source_uri != "Unknown Source"
                    else source_uri
                )
 
                st.markdown(
                    f'<div class="result-head">'
                    f'<span class="rnum">Result {idx}</span>'
                    f'<span class="rscore">relevance {score:.3f}</span>'
                    f'<span class="rscore">· {html.escape(source_name)}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
 
                # Full text, on the page, verbatim.
                render_verbatim(text)
                st.caption(f"Source: {source_uri}")
 
                # Copy path kept available for reviewers, but tucked away so it
                # doesn't clutter the read.
                with st.expander("📋 Copy exact text"):
                    st.code(text, language=None)
 
                st.divider()
 
        # ---- Planning Letter: revise the pasted draft ----------------------
        else:
            draft = question
            with st.spinner("Revising…"):
                try:
                    revised = revise_text(draft)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()
 
            if not revised:
                st.warning("No revised text was returned.")
            else:
                st.markdown("**Revised letter**")
                # Generated prose — markdown rendering is fine here.
                st.write(revised)
 
                with st.expander("📋 Copy revised text"):
                    st.code(revised, language=None)




Instructions:
- Use only facts found in the search results. Do not add outside knowledge.
- If the search results do not contain the answer, say you could not find it in the source material. Do not guess.
- Reproduce names, dates, figures, defined terms, and section numbers exactly as they appear in the sources. Do not paraphrase these.
- Be concise and objective. Do not speculate beyond the sources.
 
Search results:
$search_results$
 
$output_format_instructions$"""


PLANNING_LETTER_SYSTEM_PROMPT =

REVISION_TEMPERATURE = 0.2  # small fluency margin; set 0.0 for maximum determinism
REVISION_TOP_P = 0.9
REVISION_MAX_TOKENS = 4096

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
 
 
def revise_text(
    draft: str,
    system_prompt: str = PLANNING_LETTER_SYSTEM_PROMPT,
) -> str:
    return _kb.revise_text(draft, system_prompt=system_prompt)







