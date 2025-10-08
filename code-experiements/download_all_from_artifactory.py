# app.py
# Streamlit Knowledge Graph Explorer for survey data
# pip install streamlit pandas networkx pyvis pandas
import re
import pandas as pd
import networkx as nx
import streamlit as st
from pyvis.network import Network
# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="Knowledge Graph Explorer", layout="wide")
st.title("📊🔗 Knowledge Graph Explorer")
st.caption("Upload your CSV → map columns → build a knowledge graph → query & visualize")
# ----------------------------
# Sidebar: Upload & Config
# ----------------------------
st.sidebar.header("1) Upload CSV")
uploaded = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
st.sidebar.header("2) Column Mapping")
# Helpful notes
with st.expander("CSV expectations & tips (click to open)"):
    st.markdown("""
- **Required**: A column for *Participant ID*.  
- **Optional**: *Assessor ID*, *Review ID*.  
- **Questions**: All other columns are treated as question columns **unless** you map them as labels.  
- **Labels**: You can specify **Domain** and **Subdomain** label columns (supports multi-label like `A;B;C`).  
- For multi-label separators, use `;`, `|`, or `,`.
""")
# Defaults (you can adjust after upload)
participant_col = None
assessor_col = None
review_col = None
domain_cols = []
subdomain_cols = []
multilabel_seps = [";", "|", ","]
if uploaded is not None:
    # Load once
    df = pd.read_csv(uploaded)
    # Basic sanitation: make all column names strings
    df.columns = [str(c) for c in df.columns]
    all_cols = list(df.columns)
    st.success(f"Loaded CSV with **{df.shape[0]} rows** & **{df.shape[1]} columns**")
    # Sidebar selectors
    participant_col = st.sidebar.selectbox("Participant ID column", options=all_cols, index=0)
    review_col = st.sidebar.selectbox("Review ID column (optional)", options=["(none)"] + all_cols, index=0)
    assessor_col = st.sidebar.selectbox("Assessor ID column (optional)", options=["(none)"] + all_cols, index=0)
    domain_cols = st.sidebar.multiselect("Domain label column(s) (optional)", options=all_cols, default=[])
    subdomain_cols = st.sidebar.multiselect("Subdomain label column(s) (optional)", options=all_cols, default=[])
    sep_choice = st.sidebar.multiselect("Multi-label separators", multilabel_seps, default=multilabel_seps)
    st.sidebar.header("3) Answer Value Nodes")
    enable_value_nodes = st.sidebar.checkbox(
        "Create value nodes for low-cardinality answers (Yes/No, Likert, etc.)",
        value=True
    )
    value_cardinality_threshold = st.sidebar.slider(
        "Max unique answers to consider 'low cardinality'",
        min_value=3, max_value=30, value=12
    )
    # Determine question columns (everything not mapped)
    non_question = set([participant_col])
    if review_col != "(none)": non_question.add(review_col)
    if assessor_col != "(none)": non_question.add(assessor_col)
    non_question.update(domain_cols)
    non_question.update(subdomain_cols)
    question_cols = [c for c in all_cols if c not in non_question]
    with st.expander("Detected columns"):
        st.write("**Participant:**", participant_col)
        st.write("**Review:**", None if review_col == "(none)" else review_col)
        st.write("**Assessor:**", None if assessor_col == "(none)" else assessor_col)
        st.write("**Domain labels:**", domain_cols if domain_cols else "(none)")
        st.write("**Subdomain labels:**", subdomain_cols if subdomain_cols else "(none)")
        st.write(f"**Question columns ({len(question_cols)}):**", question_cols[:30] + (["..."] if len(question_cols) > 30 else []))
    # ----------------------------
    # Helpers
    # ----------------------------
    def split_multilabel(val):
        if pd.isna(val):
            return []
        s = str(val).strip()
        if not s:
            return []
        separators = [sep for sep in sep_choice if sep]
        if separators:
            pattern = "|".join(map(re.escape, separators))
            parts = re.split(pattern, s)
            return [p.strip() for p in parts if p.strip()]
        return [s]
    @st.cache_data(show_spinner=False)
    def compute_low_cardinality(df_in: pd.DataFrame, qcols, thresh: int):
        low = set()
        for c in qcols:
            uniq = df_in[c].dropna().astype(str).nunique()
            if 0 < uniq <= thresh:
                low.add(c)
        return low
    low_card_qs = compute_low_cardinality(df, question_cols, value_cardinality_threshold) if enable_value_nodes else set()
    # ----------------------------
    # Build Graph
    # ----------------------------
    build_graph = st.sidebar.button("4) Build Knowledge Graph")
    if build_graph:
        G = nx.Graph()
        # Pre-add label nodes cache (so reused)
        def ensure_label_node(label_text: str, label_type: str):
            nid = f"{label_type}:{label_text}"
            if not G.has_node(nid):
                G.add_node(nid, type=label_type, label=label_text)
            return nid
        # Iterate rows
        for _, row in df.iterrows():
            pid = str(row[participant_col]).strip()
            pnode = f"participant:{pid}"
            if not G.has_node(pnode):
                meta = {}
                if assessor_col != "(none)": meta["assessor"] = row[assessor_col]
                if review_col != "(none)": meta["review"] = row[review_col]
                G.add_node(pnode, type="participant", **meta)
            # Domains / Subdomains
            dnodes = []
            for dcol in domain_cols:
                for lab in split_multilabel(row[dcol]):
                    dnode = ensure_label_node(lab, "domain")
                    dnodes.append(dnode)
                    G.add_edge(pnode, dnode, rel="has_domain")
            sdnodes = []
            for sdcol in subdomain_cols:
                for lab in split_multilabel(row[sdcol]):
                    sdnode = ensure_label_node(lab, "subdomain")
                    sdnodes.append(sdnode)
                    G.add_edge(pnode, sdnode, rel="has_subdomain")
            # Optional domain-subdomain hierarchy edges
            for dn in dnodes:
                for sdn in sdnodes:
                    if not G.has_edge(dn, sdn):
                        G.add_edge(dn, sdn, rel="hierarchy")
            # Questions & Answers
            for q in question_cols:
                qnode = f"question:{q}"
                if not G.has_node(qnode):
                    G.add_node(qnode, type="question", label=q)
                ans_raw = row[q]
                ans = None if pd.isna(ans_raw) else str(ans_raw)
                G.add_edge(pnode, qnode, rel="answered", answer=ans)
                if enable_value_nodes and q in low_card_qs and ans not in (None, ""):
                    vnode = f"answer_value:{q}::{ans}"
                    if not G.has_node(vnode):
                        G.add_node(vnode, type="answer_value", question=q, value=ans)
                    G.add_edge(qnode, vnode, rel="has_value")
                    G.add_edge(pnode, vnode, rel="gave_value")
        st.session_state["G"] = G
        st.success(f"Graph built ✅  Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")
# ----------------------------
# Main Tabs (Query & Visualize)
# ----------------------------
if "G" in st.session_state:
    G = st.session_state["G"]
    tab1, tab2, tab3 = st.tabs(["🔍 Queries", "🕸️ Graph View", "ℹ️ Node Inspector"])
    # -------- Queries Tab --------
    with tab1:
        st.subheader("Quick Queries")
        if not question_cols:
            st.info("No question columns detected. Adjust your column mapping to analyze question answers.")
        else:
            colA, colB = st.columns(2)
            # Query 1: How many participants share a specific answer to a question?
            with colA:
                st.markdown("**Q1. Participants with a specific answer**")
                qsel = st.selectbox("Question", options=question_cols)
                # Offer known values (from dataset)
                vals = sorted(df[qsel].dropna().astype(str).unique().tolist())
                vsel = st.selectbox("Answer value", options=vals)
                run1 = st.button("Run Q1")
                if run1:
                    # Find participants where edge answer == vsel
                    result = []
                    qnode = f"question:{qsel}"
                    if G.has_node(qnode):
                        for nbr in G.neighbors(qnode):
                            if G.nodes[nbr].get("type") == "participant":
                                ans = G.get_edge_data(nbr, qnode).get("answer")
                                if ans == vsel:
                                    result.append(nbr.split("participant:", 1)[1])
                    st.metric("Count", len(result))
                    st.dataframe(pd.DataFrame({"Participant_ID": result}))
            # Query 2: How many participants are grouped in a (sub)domain?
            with colB:
                st.markdown("**Q2. Participants by Domain/Subdomain**")
                mode = st.radio("Type", ["Domain", "Subdomain"], horizontal=True)
                # Collect all label values from graph nodes
                label_type = "domain" if mode == "Domain" else "subdomain"
                labels = [n for n, d in G.nodes(data=True) if d.get("type") == label_type]
                labels_sorted = sorted([G.nodes[n]["label"] for n in labels]) if labels else []
                lsel = st.selectbox(f"{mode} label", options=labels_sorted if labels_sorted else ["(none)"])
                run2 = st.button("Run Q2")
                if run2 and labels_sorted:
                    lnode = f"{label_type}:{lsel}"
                    participants = [n for n in G.neighbors(lnode) if G.nodes[n].get("type") == "participant"] if G.has_node(lnode) else []
                    plist = [p.split("participant:", 1)[1] for p in participants]
                    st.metric("Count", len(plist))
                    st.dataframe(pd.DataFrame({"Participant_ID": plist}))
            st.divider()
            # Query 3: Show all answers for one participant (first N)
            st.markdown("**Q3. Answers for a participant**")
            # Offer IDs from dataset
            pids = sorted(df[participant_col].dropna().astype(str).unique().tolist()) if uploaded is not None else []
            psel = st.selectbox("Participant ID", options=pids)
            max_show = st.slider("Max answers to display", 5, 100, 20)
            run3 = st.button("Run Q3")
            if run3:
                pnode = f"participant:{psel}"
                qa = []
                if G.has_node(pnode):
                    for nbr in G.neighbors(pnode):
                        if G.nodes[nbr].get("type") == "question":
                            edge = G.get_edge_data(pnode, nbr)
                            qa.append((G.nodes[nbr]["label"], edge.get("answer")))
                qa = qa[:max_show]
                st.dataframe(pd.DataFrame(qa, columns=["Question", "Answer"]))
    # -------- Graph View Tab --------
    with tab2:
        st.subheader("Interactive Graph (ego view)")
        st.markdown(
            "Render a compact **ego graph** around a participant to keep the visualization responsive."
        )
        pids = sorted(df[participant_col].dropna().astype(str).unique().tolist()) if uploaded is not None else []
        psel = st.selectbox("Participant (ego center)", options=pids, key="ego_pid")
        radius = st.slider("Radius (hops)", 1, 2, 1)
        max_nodes = st.slider("Max nodes to display", 50, 1000, 300, step=50)
        if st.button("Render Ego Graph"):
            center = f"participant:{psel}"
            if not G.has_node(center):
                st.error("Participant not found in graph.")
            else:
                H = nx.ego_graph(G, center, radius=radius, undirected=True)
                # Trim extra-large ego graphs
                if H.number_of_nodes() > max_nodes:
                    # Keep center + highest-degree neighbors until under limit
                    nodes_sorted = sorted(H.degree, key=lambda x: x[1], reverse=True)
                    keep = set([center])
                    for n, _deg in nodes_sorted:
                        if len(keep) >= max_nodes:
                            break
                        keep.add(n)
                    H = H.subgraph(keep).copy()
                # Build PyVis network
                net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#222222", notebook=False, directed=False)
                net.barnes_hut()
                # Map styles
                def style_for(node_type):
                    if node_type == "participant":
                        return {"color": "#8ecae6", "shape": "dot", "size": 18}
                    if node_type == "question":
                        return {"color": "#90be6d", "shape": "ellipse", "size": 12}
                    if node_type == "domain":
                        return {"color": "#f4a261", "shape": "box", "size": 14}
                    if node_type == "subdomain":
                        return {"color": "#e76f51", "shape": "box", "size": 14}
                    if node_type == "answer_value":
                        return {"color": "#adb5bd", "shape": "diamond", "size": 10}
                    return {"color": "#cccccc", "shape": "dot", "size": 10}
                # Add nodes
                for n, d in H.nodes(data=True):
                    t = d.get("type", "other")
                    sty = style_for(t)
                    label = d.get("label", "")
                    if t == "participant":
                        label = n.split("participant:", 1)[1]
                    if t == "answer_value":
                        label = d.get("value", "")
                    net.add_node(n, label=label or n, title=f"type={t}", **sty)
                # Add edges (show answer on participant-question edges)
                for u, v, ed in H.edges(data=True):
                    title = ed.get("rel", "")
                    if ed.get("rel") == "answered":
                        # Find the answer text and show as tooltip
                        if H.nodes[u].get("type") == "participant" and H.nodes[v].get("type") == "question":
                            title = f"answered: {ed.get('answer', '')}"
                        elif H.nodes[v].get("type") == "participant" and H.nodes[u].get("type") == "question":
                            title = f"answered: {ed.get('answer', '')}"
                    net.add_edge(u, v, title=title, physics=True)
                # Render
                html = net.generate_html()
                st.components.v1.html(html, height=730, scrolling=True)
    # -------- Node Inspector Tab --------
    with tab3:
        st.subheader("Node Inspector")
        st.markdown("Lookup a node by exact ID (e.g., `participant:123`, `question:Q15`, `domain:Health`).")
        nid = st.text_input("Node ID")
        if st.button("Inspect"):
            if not G.has_node(nid):
                st.error("Node not found.")
            else:
                st.write("**Attributes:**", G.nodes[nid])
                nbrs = list(G.neighbors(nid))
                st.write(f"**Neighbors ({len(nbrs)}):**", nbrs[:200] + (["..."] if len(nbrs) > 200 else []))
else:
    st.info("Upload a CSV and configure columns in the sidebar to get started.")
