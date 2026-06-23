from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from src.retriever import SepsisRetriever

LLM_MODEL = "llama3.1:8b"

# Prompt template — instructs the LLM to answer only from retrieved context
PROMPT_TEMPLATE = """You are a clinical research assistant specializing in sepsis and critical care medicine.

Use ONLY the following PubMed abstracts to answer the question. Do not use any prior knowledge.
Cite each abstract you draw from by its PMID at the end of your answer.
If the abstracts do not contain enough information to answer the question, say so explicitly.

---
{context}
---

Question: {question}

Answer:"""


class RAGChain:
    """
    Full RAG pipeline: retrieves relevant PubMed abstracts then
    generates a grounded answer using a local Ollama LLM.
    """

    def __init__(self, n_results: int = 5):
        self.retriever = SepsisRetriever(n_results=n_results)
        self.llm       = OllamaLLM(model=LLM_MODEL)
        self.prompt    = PromptTemplate(
            input_variables=["context", "question"],
            template=PROMPT_TEMPLATE,
        )
        self.chain = self.prompt | self.llm

    def run(
        self,
        question:  str,
        year_from: int | None = None,
        year_to:   int | None = None,
        verbose:   bool = False,
    ) -> dict:
        """
        Run the full RAG pipeline for a given question.

        Args:
            question:  Natural language clinical question.
            year_from: Optional lower bound on publication year.
            year_to:   Optional upper bound on publication year.
            verbose:   If True, print retrieved sources before the answer.

        Returns:
            Dict with keys: question, answer, sources.
        """
        # Step 1 — Retrieve relevant abstracts
        retrieved = self.retriever.retrieve(question, year_from=year_from, year_to=year_to)
        context   = self.retriever.format_for_prompt(retrieved)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"\nRetrieved {len(retrieved)} abstracts:")
            for r in retrieved:
                print(f"  [{r['distance']}] PMID {r['pmid']} ({r['year']}) — {r['title'][:65]}...")
            print(f"\n{'='*60}\n")

        # Step 2 — Generate answer from retrieved context
        answer = self.chain.invoke({"context": context, "question": question})

        # Step 3 — Package sources for downstream use (evaluation, display)
        sources = [
            {
                "pmid":     r["pmid"],
                "title":    r["title"],
                "year":     r["year"],
                "journal":  r["journal"],
                "distance": r["distance"],
            }
            for r in retrieved
        ]

        return {
            "question": question,
            "answer":   answer.strip(),
            "sources":  sources,
        }