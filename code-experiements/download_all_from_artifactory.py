# app.py
# Streamlit Knowledge Graph (multi-label domains/subdomains)

import pandas as pd
import networkx as nx
import streamlit as st
from pyvis.network import Network
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Multi-Label Knowledge Graph", layout="wide")
st.title("🔗 Knowledge Graph — Participants ↔ Domains/Subdomains")

# ---------- Upload ----------
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
st.sidebar.caption("Tip: Domain/Subdomain columns may contain multi-labels like: 'Health; Education | Safety'.")

def split_multi(val, seps):
    if pd.isna(val): return []
    s = str(val).strip()
    if not s: return []
    # split by the first matching separator present
    for sp in seps:
        if sp in s:
            return [p.strip() for p in s.split(sp) if p.strip()]
    return [s]

if uploaded is not None:
    df = pd.read_csv(uploaded)
    df.columns = [str(c) for c in df.columns]

    st.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # ---------- Mapping ----------
    st.sidebar.header("Column Mapping")
    cols = list(df.columns)

    participant_col = st.sidebar.selectbox("Participant ID", options=cols)
    review_col = st.sidebar.selectbox("Review ID (optional)", options=["(none)"] + cols, index=0)
    assessor_col = st.sidebar.selectbox("Assessor ID (optional)", options=["(none)"] + cols, index=0)

    domain_cols = st.sidebar.multiselect("Domain label column(s)", options=cols)
    subdomain_cols = st.sidebar.multiselect("Subdomain label column(s)", options=cols)

    # remaining columns become “answer” columns (57 answers)
    reserved = set([participant_col])
    if review_col != "(none)": reserved.add(review_col)
    if assessor_col != "(none)": reserved.add(assessor_col)
    reserved.update(domain_cols)
    reserved.update(subdomain_cols)
    answer_cols = [c for c in cols if c not in reserved]

    st.sidebar.header("Multi-label separators")
    seps = st.sidebar.multiselect("Choose all separators used", [";", "|", ","], default=[";", "|", ","])

    with st.expander("Detected columns", expanded=False):
        st.write("**Domains:**", domain_cols or "(none)")
        st.write("**Subdomains:**", subdomain_cols or "(none)")
        st.write(f"**Answer columns ({len(answer_cols)}):**", answer_cols[:25] + (["..."] if len(answer_cols) > 25 else []))

    # ---------- Build Graph ----------
    if st.sidebar.button("Build Knowledge Graph"):
        G = nx.Graph()

        # helper to add (once) label nodes
        def ensure_label_node(label: str, ltype: str):
            nid = f"{ltype}:{label}"
            if not G.has_node(nid):
                G.add_node(nid, type=ltype, label=label)
            return nid

        for _, row in df.iterrows():
            pid = str(row[participant_col]).strip()
            pnode = f"participant:{pid}"
            if not G.has_node(pnode):
                meta = {}
                if review_col != "(none)": meta["review_id"] = row[review_col]
                if assessor_col != "(none)": meta["assessor_id"] = row[assessor_col]
                G.add_node(pnode, type="participant", **meta)

            # domains
            dnodes = []
            for dcol in domain_cols:
                for lab in split_multi(row[dcol], seps):
                    if not lab: continue
                    dnode = ensure_label_node(lab, "domain")
                    dnodes.append(dnode)
                    G.add_edge(pnode, dnode, rel="in_domain")

            # subdomains
            sdnodes = []
            for sdcol in subdomain_cols:
                for lab in split_multi(row[sdcol], seps):
                    if not lab: continue
                    sdnode = ensure_label_node(lab, "subdomain")
                    sdnodes.append(sdnode)
                    G.add_edge(pnode, sdnode, rel="in_subdomain")

            # optional hierarchy (domain—subdomain)
            for dn in dnodes:
                for sdn in sdnodes:
                    if not G.has_edge(dn, sdn):
                        G.add_edge(dn, sdn, rel="hierarchy")

            # (Optional) store answers on participant node for table view
            for q in answer_cols:
                v = None if pd.isna(row[q]) else str(row[q])
                if v not in (None, ""):
                    # store a few as attributes (avoid ballooning)
                    # attr key: a sanitized version
                    key = f"Q::{q}"
                    G.nodes[pnode][key] = v

        st.session_state["G"] = G
        st.session_state["data_snapshot"] = {
            "participant_col": participant_col,
            "domain_cols": domain_cols,
            "subdomain_cols": subdomain_cols,
            "answer_cols": answer_cols,
            "df_head": df.head(3).to_dict(orient="list")
        }
        st.success(f"Graph built ✅ Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,}")

# ---------- UI Tabs ----------
if "G" in st.session_state:
    G = st.session_state["G"]
    participant_col = st.session_state["data_snapshot"]["participant_col"]
    domain_cols = st.session_state["data_snapshot"]["domain_cols"]
    subdomain_cols = st.session_state["data_snapshot"]["subdomain_cols"]
    answer_cols = st.session_state["data_snapshot"]["answer_cols"]

    tabQ, tabG, tabI = st.tabs(["🔍 Queries", "🕸️ Graph (Ego View)", "ℹ️ Inspect"])

    # ===== Queries =====
    with tabQ:
        st.subheader("Quick Queries")
        c1, c2 = st.columns(2)

        # Q1: Count/list by Domain/Subdomain
        with c1:
            mode = st.radio("Label type", ["Domain", "Subdomain"], horizontal=True)
            ltype = "domain" if mode == "Domain" else "subdomain"
            labels = sorted([d["label"] for n, d in G.nodes(data=True) if d.get("type") == ltype])
            choice = st.selectbox(f"Select a {mode}", options=labels if labels else ["(none)"])
            if st.button(f"Show participants in {mode}"):
                target = f"{ltype}:{choice}"
                plist = []
                if G.has_node(target):
                    for nbr in G.neighbors(target):
                        if G.nodes[nbr].get("type") == "participant":
                            plist.append(nbr.split("participant:", 1)[1])
                st.metric("Participants", len(plist))
                st.dataframe(pd.DataFrame({"Participant_ID": plist}))

        # Q2: Show all labels for one participant
        with c2:
            # gather pids from graph
            pids = sorted(n.split("participant:", 1)[1]
                          for n, d in G.nodes(data=True) if d.get("type") == "participant")
            pid = st.selectbox("Participant", options=pids)
            if st.button("Show participant labels"):
                pnode = f"participant:{pid}"
                domains, sds = [], []
                if G.has_node(pnode):
                    for nbr in G.neighbors(pnode):
                        t = G.nodes[nbr].get("type")
                        if t == "domain": domains.append(G.nodes[nbr]["label"])
                        if t == "subdomain": sds.append(G.nodes[nbr]["label"])
                st.write("**Domains**:", sorted(set(domains)))
                st.write("**Subdomains**:", sorted(set(sds)))

        st.divider()
        # Q3 (optional): Cohort by exact match on a few answer columns
        st.markdown("**Cohort by exact matches on selected answer columns** (optional, simple filter)")
        pick = st.multiselect("Pick up to 5 answer columns", options=answer_cols, default=answer_cols[:3])
        vals = {}
        for q in pick[:5]:
            # propose value candidates from attributes found in graph (fallback to text box omitted for brevity)
            vals[q] = st.text_input(f"Value for `{q}` (exact match)")

        if st.button("Find participants with these answers"):
            hits = []
            for n, d in G.nodes(data=True):
                if d.get("type") != "participant": continue
                ok = True
                for q, val in vals.items():
                    if val is None or val == "": continue
                    if d.get(f"Q::{q}") != val:
                        ok = False; break
                if ok: hits.append(n.split("participant:", 1)[1])
            st.metric("Matched participants", len(hits))
            st.dataframe(pd.DataFrame({"Participant_ID": hits}))

    # ===== Graph (Ego View) =====
    with tabG:
        st.subheader("Interactive Ego View")
        st.caption("Pick a node to center the view. Radius 1 shows direct neighbors (participants ↔ labels).")

        # choose center type
        focus_type = st.radio("Center type", ["Participant", "Domain", "Subdomain"], horizontal=True)
        if focus_type == "Participant":
            options = sorted(n.split("participant:", 1)[1] for n, d in G.nodes(data=True) if d.get("type") == "participant")
            center_raw = st.selectbox("Participant ID", options=options)
            center = f"participant:{center_raw}"
        elif focus_type == "Domain":
            options = sorted(d["label"] for n, d in G.nodes(data=True) if d.get("type") == "domain")
            center_raw = st.selectbox("Domain", options=options)
            center = f"domain:{center_raw}"
        else:
            options = sorted(d["label"] for n, d in G.nodes(data=True) if d.get("type") == "subdomain")
            center_raw = st.selectbox("Subdomain", options=options)
            center = f"subdomain:{center_raw}"

        radius = st.slider("Radius (hops)", 1, 2, 1)
        max_nodes = st.slider("Max nodes", 50, 2000, 500, step=50)

        if st.button("Render"):
            if not G.has_node(center):
                st.error("Center node not found.")
            else:
                H = nx.ego_graph(G, center, radius=radius, undirected=True)

                # Trim if too large
                if H.number_of_nodes() > max_nodes:
                    # keep center, then highest-degree neighbors until limit
                    deg_sorted = sorted(H.degree, key=lambda x: x[1], reverse=True)
                    keep = {center}
                    for n, _deg in deg_sorted:
                        if len(keep) >= max_nodes: break
                        keep.add(n)
                    H = H.subgraph(keep).copy()

                net = Network(height="740px", width="100%", bgcolor="#ffffff", font_color="#222")
                net.barnes_hut()

                def style(ntype):
                    return {
                        "participant": {"color": "#8ecae6", "shape": "dot", "size": 18},
                        "domain": {"color": "#f4a261", "shape": "box", "size": 14},
                        "subdomain": {"color": "#e76f51", "shape": "box", "size": 14},
                    }.get(ntype, {"color": "#cccccc", "shape": "dot", "size": 10})

                for n, d in H.nodes(data=True):
                    t = d.get("type", "other")
                    lab = d.get("label", "")
                    if t == "participant":
                        lab = n.split("participant:", 1)[1]
                    net.add_node(n, label=lab or n, title=f"type={t}", **style(t))

                for u, v, ed in H.edges(data=True):
                    net.add_edge(u, v, title=ed.get("rel", ""))

                html = net.generate_html()
                st.components.v1.html(html, height=760, scrolling=True)

    # ===== Inspect =====
    with tabI:
        st.subheader("Inspect Node")
        nid = st.text_input("Node ID (e.g., participant:123, domain:Health, subdomain:Exercise)")
        if st.button("Inspect"):
            if not G.has_node(nid):
                st.error("Not found.")
            else:
                st.write("**Attributes:**", G.nodes[nid])
                neighbors = list(G.neighbors(nid))
                st.write(f"**Neighbors ({len(neighbors)}):**", neighbors[:200] + (["..."] if len(neighbors) > 200 else []))
else:
    st.info("Upload your CSV, map the columns, then click **Build Knowledge Graph**.")
