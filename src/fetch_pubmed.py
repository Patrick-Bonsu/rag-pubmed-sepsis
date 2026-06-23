import json
import time
from pathlib import Path
from Bio import Entrez

# NCBI requires a registered email
Entrez.email = "pybonsu@gmail.com"

SEARCH_QUERIES = [
    "sepsis ICU mortality",
    "septic shock vasopressors treatment",
    "mechanical ventilation sepsis outcomes",
    "sepsis-3 criteria diagnosis",
]

RAW_DATA_PATH = Path("data/raw/abstracts.json")


def search_pubmed(query: str, max_results: int = 250) -> list[str]:
    """Return PMIDs matching a PubMed search query."""
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def fetch_abstracts(pmids: list[str], batch_size: int = 100) -> list[dict]:
    """Fetch and parse XML records for a list of PMIDs in batches."""
    records = []

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]

        handle = Entrez.efetch(db="pubmed", id=batch, rettype="xml", retmode="xml")
        batch_records = Entrez.read(handle)
        handle.close()

        for article in batch_records["PubmedArticle"]:
            parsed = parse_article(article)
            if parsed:
                records.append(parsed)

        print(f"  Fetched {min(i + batch_size, len(pmids))}/{len(pmids)} records...")
        time.sleep(0.4)  # Stay within NCBI's 3 requests/second limit

    return records


def parse_article(article: dict) -> dict | None:
    """Extract relevant fields from a PubMed XML article. Returns None if no abstract."""
    try:
        medline = article["MedlineCitation"]
        article_data = medline["Article"]

        # Skip records without an abstract — not useful for retrieval
        abstract_texts = article_data.get("Abstract", {}).get("AbstractText", [])
        if not abstract_texts:
            return None

        # Abstracts can be structured (list of labelled sections) or plain strings
        if isinstance(abstract_texts, list):
            abstract = " ".join(str(t) for t in abstract_texts)
        else:
            abstract = str(abstract_texts)

        if len(abstract.strip()) < 50:
            return None

        pmid  = str(medline["PMID"])
        title = str(article_data.get("ArticleTitle", ""))

        # Publication year — fall back to MedlineDate for older records
        pub_date = (
            article_data.get("Journal", {})
            .get("JournalIssue", {})
            .get("PubDate", {})
        )
        year = str(pub_date.get("Year", pub_date.get("MedlineDate", "Unknown")))[:4]

        journal    = str(article_data.get("Journal", {}).get("Title", "Unknown"))
        mesh_list  = medline.get("MeshHeadingList", [])
        mesh_terms = [str(h["DescriptorName"]) for h in mesh_list]

        return {
            "pmid":       pmid,
            "title":      title,
            "abstract":   abstract,
            "year":       year,
            "journal":    journal,
            "mesh_terms": mesh_terms,
        }

    except (KeyError, IndexError):
        return None


def main():
    all_pmids: set[str] = set()

    # Collect PMIDs across all queries and deduplicate
    for query in SEARCH_QUERIES:
        print(f"Searching: '{query}'")
        pmids = search_pubmed(query, max_results=250)
        all_pmids.update(pmids)
        print(f"  {len(pmids)} results ({len(all_pmids)} unique so far)")
        time.sleep(0.4)

    print(f"\nTotal unique PMIDs: {len(all_pmids)}")
    print("Fetching abstracts...")

    records = fetch_abstracts(list(all_pmids))

    print(f"\nRecords with abstracts: {len(records)}")

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_DATA_PATH, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved → {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()