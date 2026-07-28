 for citation in citations:
                    for ref in citation.get("retrievedReferences", []):
                        location = ref.get("location", {})
                        source_uri = extract_source_uri(location)
                        snippet = ref.get("content", {}).get("text", "")
 
                        if source_uri in displayed_sources:
                            continue
                        displayed_sources.add(source_uri)
 
                        filename = os.path.basename(source_uri)
 
                        with st.expander(f"📄 {filename}"):
                            st.caption(source_uri)
                            if snippet:
                                # Retrieved source text — render verbatim.
                                st.code(snippet[:SNIPPET_LIMIT], language=None)
                                if len(snippet) > SNIPPET_LIMIT:
                                    st.caption(
                                        f"… showing first {SNIPPET_LIMIT} of "
                                        f"{len(snippet)} characters"
                                    )
