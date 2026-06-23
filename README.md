# RAG over Biomedical Literature — Sepsis & Critical Care

Retrieval-Augmented Generation pipeline over PubMed abstracts for clinical question answering in sepsis and critical care medicine. Answers are grounded exclusively in retrieved literature — every claim is traceable to a specific PMID.

This project is part of a deliberate portfolio sequence connecting clinical data modalities: [ICU mortality prediction](https://github.com/Patrick-Bonsu/icu-mortality-prediction) (structured time-series) → **RAG over sepsis literature** (unstructured biomedical text) → drug-target interaction prediction (molecular graphs). The ICU/sepsis topic was chosen intentionally to extend the clinical narrative from the prior project.

---

## Architecture

```
User Question
      │
      ▼
 PubMedBERT Embedding          ← domain-specific encoder
      │
      ▼
 ChromaDB Vector Search        ← cosine similarity over 925 abstracts
      │  (optional year filter)
      ▼
 Top-5 Abstracts + Metadata    ← PMID, year, journal, MeSH terms
      │
      ▼
 LangChain Prompt Template     ← "answer only from this context, cite PMIDs"
      │
      ▼
 Llama 3.1 8B (Ollama)         ← local inference, no API cost
      │
      ▼
 Grounded Answer + Sources
```

---

## Key Design Decisions

**Domain-specific embeddings.** General-purpose models (e.g. `text-embedding-ada-002`) underperform on biomedical text because the vocabulary is specialized — abbreviations like "MAP", "SOFA", and "CRRT" are ambiguous outside a clinical context. `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` was pretrained exclusively on PubMed abstracts and full-text articles, producing a semantic space that correctly clusters clinical concepts.

**Whole-abstract chunking.** PubMed abstracts average ~250 words — compact enough to serve as a single retrievable unit without splitting. This avoids the context fragmentation that affects full-text chunking strategies.

**Metadata filtering.** Each indexed document retains structured metadata (PMID, year, journal, MeSH terms) stored as typed fields in ChromaDB. Year-range filtering at query time is a production-relevant feature absent from most tutorial implementations.

**Constrained generation prompt.** The LLM is instructed to answer only from provided abstracts and to cite PMIDs explicitly. This makes hallucination auditable — any claim unsupported by the retrieved context is identifiable.

---

## Dataset

- **Source:** NCBI PubMed via Biopython Entrez API (no credentialing required)
- **Corpus:** 925 abstracts across four complementary search queries:
  - `"sepsis ICU mortality"`
  - `"septic shock vasopressors treatment"`
  - `"mechanical ventilation sepsis outcomes"`
  - `"sepsis-3 criteria diagnosis"`
- **Deduplication:** by PMID across queries
- **Coverage:** primarily 2020–2026, reflecting recent guideline evolution (Surviving Sepsis Campaign 2021, Sepsis-3)

---

## Evaluation

Evaluated on a 10-question hand-labeled test set written from a clinical perspective — the kind of questions an ICU clinician or critical care researcher would actually ask. Scored using an LLM-as-judge approach (Llama 3.1 8B) across three metrics.

| Metric | Score | Description |
|--------|-------|-------------|
| **Faithfulness** | **0.840** | Answer claims are grounded in retrieved context |
| **Answer Relevancy** | **0.750** | Answer directly addresses the question asked |
| **Context Precision** | **0.800** | Retrieved abstracts are relevant to the question |

**Findings:**

- Faithfulness is the strongest metric, confirming the constrained prompt is effective at preventing hallucination
- Answer relevancy is lowest for broad epidemiological questions (e.g. mortality outcomes, 0.60) because the corpus covers mortality as a secondary endpoint rather than a primary topic — retrieved abstracts are tangentially rather than directly relevant
- Context precision is uniform across questions (all 0.8), indicating consistent retrieval quality regardless of question type
- A hybrid retrieval strategy (dense + BM25 keyword matching) would likely improve relevancy on exact-terminology queries (drug names, score thresholds)

---

## Project Structure

```
rag-pubmed-sepsis/
├── app.py                          # Gradio demo with source citation and year filtering
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── abstracts.json          # 925 fetched PubMed abstracts
│   └── processed/
│       └── evaluation_results.json # per-question evaluation scores
├── src/
│   ├── __init__.py
│   ├── fetch_pubmed.py             # Entrez API data pipeline
│   ├── index.py                    # PubMedBERT embedding + ChromaDB indexing
│   ├── retriever.py                # semantic search with metadata filtering
│   ├── chain.py                    # LangChain RAG chain
│   └── evaluate.py                 # LLM-as-judge evaluation
└── notebooks/
```

---

## Setup

**Prerequisites:** [Ollama](https://ollama.com) installed and running (`brew install ollama && brew services start ollama`)

```bash
# Pull the local LLM
ollama pull llama3.1:8b

# Clone and set up environment
git clone https://github.com/Patrick-Bonsu/rag-pubmed-sepsis
cd rag-pubmed-sepsis

conda create -n rag-env python=3.11 -y
conda activate rag-env
pip install -r requirements.txt
```

**Build the index** (fetches abstracts from PubMed and embeds with PubMedBERT):

```bash
python src/fetch_pubmed.py   # ~3 min — downloads 925 abstracts
python src/index.py          # ~1 min on Apple Silicon MPS
```

**Run the demo:**

```bash
python app.py
# Open http://127.0.0.1:7860
```

**Run evaluation:**

```bash
python -m src.evaluate       # ~10–15 min
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Data access | Biopython Entrez |
| Embeddings | PubMedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`) |
| Vector database | ChromaDB (persistent, local) |
| Retrieval | Cosine similarity + metadata filtering |
| Orchestration | LangChain |
| LLM (local) | Llama 3.1 8B via Ollama |
| Evaluation | LLM-as-judge (Faithfulness, Answer Relevancy, Context Precision) |
| Demo | Gradio |

---

## Limitations

- **Corpus scope:** 925 abstracts covers core sepsis management well but has sparse coverage of sub-topics (paediatric sepsis, fungal sepsis, post-ICU syndrome)
- **Local LLM ceiling:** Llama 3.1 8B is the judge and generator — a larger model would produce higher-quality answers and more reliable evaluation scores
- **Dense retrieval only:** exact-terminology queries (specific drug names, score thresholds) would benefit from hybrid retrieval (BM25 + dense)
- **No reranking:** a cross-encoder reranker applied to top-20 candidates would improve precision for the tangential results currently in positions 4–5

## Future Work

- Hybrid retrieval (BM25 + dense) using `rank_bm25`
- Cross-encoder reranking with `sentence-transformers` CrossEncoder
- MeSH term filtering as an additional retrieval dimension
- Scale to full-text articles via PubMed Central OA dataset
- Swap Ollama for Claude API for deployment on HuggingFace Spaces

---

## Related Projects

- [Medical Image Segmentation — BraTS Pediatric](https://github.com/Patrick-Bonsu/medical-seg-brats-ped) — 3D U-Net on pediatric brain tumor MRI
- [ICU Mortality Prediction — MIMIC-IV](https://github.com/Patrick-Bonsu/icu-mortality-prediction) — clinical time-series with XGBoost + Temporal Transformer
