import json
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

RAW_DATA_PATH   = Path("data/raw/abstracts.json")
CHROMA_DB_PATH  = Path("chroma_db")
COLLECTION_NAME = "sepsis_abstracts"
EMBEDDING_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
BATCH_SIZE      = 64


def load_abstracts(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_document_text(record: dict) -> str:
    """Concatenate title and abstract for richer embedding context."""
    return f"{record['title']}\n\n{record['abstract']}"


def build_metadata(record: dict) -> dict:
    """Flatten record into ChromaDB-compatible metadata (no lists allowed)."""
    return {
        "pmid":       record["pmid"],
        "title":      record["title"],
        "year":       int(record["year"]) if record["year"].isdigit() else 0,
        "journal":    record["journal"],
        "mesh_terms": ", ".join(record["mesh_terms"]) if record["mesh_terms"] else "",
    }


def embed_in_batches(
    model: SentenceTransformer, texts: list[str], batch_size: int
) -> list[list[float]]:
    """Embed texts in batches and return as a flat list of vectors."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch      = texts[i : i + batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())
    return all_embeddings


def main():
    # Load abstracts
    print(f"Loading abstracts from {RAW_DATA_PATH}...")
    records = load_abstracts(RAW_DATA_PATH)
    print(f"Loaded {len(records)} records")

    # Load PubMedBERT — use MPS on Apple Silicon, fall back to CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nLoading embedding model on {device}: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # Prepare inputs
    documents = [build_document_text(r) for r in records]
    metadatas = [build_metadata(r) for r in records]
    ids       = [r["pmid"] for r in records]

    # Embed all documents
    print(f"\nEmbedding {len(documents)} documents in batches of {BATCH_SIZE}...")
    embeddings = embed_in_batches(model, documents, BATCH_SIZE)
    print(f"Done — {len(embeddings)} vectors, dim {len(embeddings[0])}")

    # Initialize persistent ChromaDB
    print(f"\nInitializing ChromaDB at {CHROMA_DB_PATH}/")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    # Drop existing collection for clean re-runs
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Dropped existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # Cosine similarity — standard for text
    )

    # Insert documents in batches
    print(f"Inserting {len(records)} documents into ChromaDB...")
    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Indexing"):
        collection.add(
            ids        = ids[i : i + BATCH_SIZE],
            documents  = documents[i : i + BATCH_SIZE],
            embeddings = embeddings[i : i + BATCH_SIZE],
            metadatas  = metadatas[i : i + BATCH_SIZE],
        )

    print(f"\n✓ Indexed {collection.count()} documents in '{COLLECTION_NAME}'")
    print(f"✓ Persisted to {CHROMA_DB_PATH}/")

    # Sanity check — run a real retrieval query
    print("\nSanity check — querying: 'vasopressor management septic shock'")
    query_vec = model.encode("vasopressor management septic shock").tolist()
    results   = collection.query(
        query_embeddings = [query_vec],
        n_results        = 3,
        include          = ["documents", "metadatas", "distances"],
    )

    for i, (meta, dist) in enumerate(zip(
        results["metadatas"][0],
        results["distances"][0],
    )):
        print(f"\n  Result {i + 1}  (cosine distance: {dist:.4f})")
        print(f"  PMID: {meta['pmid']} | Year: {meta['year']}")
        print(f"  {meta['title'][:90]}...")


if __name__ == "__main__":
    main()