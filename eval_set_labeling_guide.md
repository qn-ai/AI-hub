# EVAL-2 — Evaluation set labeling guide

Companion to `eval_set.jsonl`. The dataset is the ground truth every metric runner depends on: the libraries do the math, but the labels here are what make the numbers meaningful. One JSON object per line (JSONL). Fields prefixed with `_` are labeling helpers and are **ignored by the metric runners** — delete them or keep them, either is fine.

---

## Record schema

**Presence** = must the key exist on the record. **Nullable** = may its value be null/empty once present.
- `always` — key must be on every record.
- `conditional` — key must be present only when the trigger condition holds; omit otherwise.
- `never null` — if the key is present, a non-null/non-empty value is required.
- `null when <x>` — null/empty is the correct value in that case, non-null otherwise.

| Field | Presence | Nullable | Who fills | What it is |
|---|---|---|---|---|
| `query_id` | always | never null | pre-filled | Stable ID. Prefix by category: `V-` verbatim, `R-` reasoning, `F-` freshness, `O-` out-of-scope, `P-` probe. |
| `query` | always | never null | pre-filled / author | The reviewer's question, phrased as they would ask it. |
| `expected_path` | always | never null | author | `verbatim` or `reasoning`. Drives **routing accuracy**. |
| `scope` | always | never null | author | `in_scope` or `out_of_scope`. Drives **abstention accuracy**. |
| `source_tier` | always | `null` when `out_of_scope`, else never null | author | Trust label: `authoritative`, `sharepoint`, `web`, or a combination (`authoritative+sharepoint`). Drives **provenance** + **source-restriction**. |
| `relevant_chunk_ids` | always | empty `[]` when `out_of_scope`, else ≥1 required | labeler | Chunk IDs that contain the answer. Ground truth for **precision@k / recall@k / MRR / NDCG / hit rate**. |
| `gold_answer` | always | `null` when `out_of_scope`, else never null | labeler | Verbatim: the **exact** source span, char-for-char. Reasoning: a reference answer. |
| `k` | always | never null | pre-filled | Retrieval cutoff (5 verbatim, 8 reasoning by default). |
| `expected_behaviour` | conditional — present iff `out_of_scope` | never null when present (`abstain`) | author | The system must return None / "not found". |
| `time_sensitive` | conditional — present iff freshness case | never null when present (`true`) | author | Marks records where cited sources must be recent. |
| `freshness_window_days` | conditional — present iff `time_sensitive` | never null when present (integer) | author | Acceptable recency window in days. |
| `probe_type` | conditional — present iff a probe | never null when present | author | `information_integration` or `counterfactual`. |
| `conflicting_chunk_ids` | conditional — present iff `probe_type == counterfactual` | ≥1 required when present | labeler | The stale/incorrect lower-trust chunk the model must **not** follow. |
| `_status`, `_labeling_note` | optional | may be null/omitted | labeler | Helper fields; **ignored by runners**. |

**Cross-field rules a validator should enforce:**
- `scope == out_of_scope` ⟺ `relevant_chunk_ids == []` **and** `gold_answer == null` **and** `source_tier == null` **and** `expected_behaviour == "abstain"`.
- `scope == in_scope` ⟹ `len(relevant_chunk_ids) ≥ 1` **and** `gold_answer` is non-null.
- `probe_type == counterfactual` ⟹ `conflicting_chunk_ids` present with ≥1 ID.
- `time_sensitive == true` ⟹ `freshness_window_days` present.
- No record should reach a run with `_status == "TODO"`.

---

## The chunk-ID convention (decide this first)

Encode the **source in the ID prefix** so provenance and source-restriction are one-line metadata ratios, not separate lookups:

```
authdoc::<doc>#<section>::chunk_<NNN>      e.g. authdoc::access-policy#s3.2::chunk_014
sharepoint::<path>::chunk_<NNN>            e.g. sharepoint::updates/2026-05-evidence::chunk_003
web::<host>/<path>::chunk_<NNN>            e.g. web::ourguidelines.ndis.gov.au/access::chunk_011
```

Chunk IDs **must match the real IDs your Bedrock KB emits after ingestion** — so labeling can only be finalized once the KB is synced and the actual chunk boundaries exist. Agree the prefix scheme before anyone labels a record; changing it later invalidates every retrieval label.

---

## Decision rules

- **verbatim vs. reasoning** — if the reviewer needs the *exact words* of a source (a clause, definition, quoted figure), it's `verbatim`. If they need something *composed* (a summary, an explanation, a comparison across sources), it's `reasoning`. When unsure, ask: would a paraphrase be a failure? Yes → verbatim.
- **in_scope vs. out_of_scope** — is the answer actually present in one of the three sources? If not, it's `out_of_scope` and the correct output is abstention, not a best guess. Include a healthy share of these — they are the only thing that scores whether the system refuses to fabricate.
- **source_tier** — trust order is `authoritative` > `sharepoint` > `web`. For verbatim records this is always `authoritative` (only the authoritative doc is verbatim-eligible).

---

## The one thing that breaks metrics if done wrong

For **verbatim** records, `gold_answer` must be the exact post-chunking source span — same casing, punctuation, and whitespace as the retrieved chunk. If the gold text differs from the source by even a comma, normalized exact match and LCS will report false failures against your own labels. Copy the span from the actual chunk, don't retype it from the source document.

---

## Labeling workflow

1. Author writes `query`, `expected_path`, `scope`, `source_tier`, and any probe/freshness flags → set `_status: "TODO"`.
2. After KB sync, run the query through `retrieve` and identify which returned chunk(s) actually contain the answer → fill `relevant_chunk_ids`.
3. Fill `gold_answer` (exact span for verbatim; reference answer for reasoning). Out-of-scope stays `null`.
4. Set `_status: "FINAL"` and version the file (e.g. `eval_set.v1.jsonl`). The set is **frozen** once final — changes require a new version so scores stay comparable across runs.

## Quality checklist before freezing

- [ ] Both paths and out-of-scope cases represented; agree target counts with the Director.
- [ ] Every in-scope record has at least one `relevant_chunk_id`.
- [ ] Every verbatim `gold_answer` copied (not retyped) from the real chunk.
- [ ] Chunk-ID prefixes follow the agreed convention and match live KB IDs.
- [ ] Counterfactual probes have both `relevant_chunk_ids` and `conflicting_chunk_ids`.
- [ ] File is versioned and frozen.
