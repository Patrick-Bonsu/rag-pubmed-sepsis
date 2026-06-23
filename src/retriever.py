from pathlib import Path

import torch
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH  = Path("chroma_db")
COLLECTION_NAME = "sepsis_abstracts"
EMBEDDING_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


class SepsisRetriever:
    """
    Retrieves relevant PubMed abstracts from ChromaDB using PubMedBERT embeddings.
    Supports optional year filtering for time-scoped queries.
    """

    def __init__(self, n_results: int = 5):
        self.n_results = n_results

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=device)

        client = PersistentClient(path=str(CHROMA_DB_PATH))
        self.collection = client.get_collection(COLLECTION_NAME)

    def retrieve(
        self,
        query: str,
        year_from: int | None = None,
        year_to:   int | None = None,
    ) -> list[dict]:
        """
        Embed the query and return the top-k most similar abstracts.

        Args:
            query:     Natural language clinical question.
            year_from: Optional lower bound on publication year (inclusive).
            year_to:   Optional upper bound on publication year (inclusive).

        Returns:
            List of result dicts with keys: pmid, title, abstract,
            year, journal, mesh_terms, distance.
        """
        query_embedding = self.model.encode(query).tolist()

        # Build ChromaDB metadata filter if year bounds are specified
        where_clause = self._build_year_filter(year_from, year_to)

        results = self.collection.query(
            query_embeddings = [query_embedding],
            n_results        = self.n_results,
            include          = ["documents", "metadatas", "distances"],
            where            = where_clause,
        )

        return self._parse_results(results)

    def _build_year_filter(
        self,
        year_from: int | None,
        year_to:   int | None,
    ) -> dict | None:
        """Construct a ChromaDB 'where' clause for year-based filtering."""
        if year_from is None and year_to is None:
            return None

        conditions = []
        if year_from:
            conditions.append({"year": {"$gte": year_from}})
        if year_to:
            conditions.append({"year": {"$lte": year_to}})

        return {"$and": conditions} if len(conditions) > 1 else conditions[0]

    def _parse_results(self, results: dict) -> list[dict]:
        """Unpack ChromaDB query output into a clean list of dicts."""
        parsed = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            parsed.append({
                "pmid":       meta["pmid"],
                "title":      meta["title"],
                "abstract":   doc,
                "year":       meta["year"],
                "journal":    meta["journal"],
                "mesh_terms": meta["mesh_terms"],
                "distance":   round(dist, 4),
            })
        return parsed

    def format_for_prompt(self, results: list[dict]) -> str:
        """
        Format retrieved abstracts into a prompt-ready context block.
        Each abstract is labelled with its PMID for source attribution.
        """
        sections = []
        for r in results:
            sections.append(
                f"[PMID: {r['pmid']} | {r['year']} | {r['journal']}]\n"
                f"Title: {r['title']}\n"
                f"{r['abstract']}"
            )
        return "\n\n---\n\n".join(sections)