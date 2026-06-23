import os
import gradio as gr
from src.retriever import SepsisRetriever
from src.chain import RAGChain

# Evaluation summary to display in the demo — from our evaluation run
EVAL_SUMMARY = {
    "faithfulness":      0.840,
    "answer_relevancy":  0.750,
    "context_precision": 0.800,
}

EXAMPLE_QUESTIONS = [
    "What is the recommended first-line vasopressor for septic shock?",
    "What is the Sepsis-3 definition of septic shock?",
    "What initial fluid resuscitation volume is recommended in septic shock?",
    "What is the role of lactate monitoring in sepsis management?",
    "When is vasopressin added to norepinephrine in septic shock treatment?",
    "How is fluid responsiveness assessed in critically ill sepsis patients?",
]

# Initialize pipeline once at startup
print("Loading RAG pipeline...")
chain = RAGChain(n_results=5)
print("Pipeline ready.")


def run_query(question: str, year_from: int, year_to: int) -> tuple[str, str]:
    """
    Run the full RAG pipeline and return formatted answer and sources.
    Returns a tuple of (answer_text, sources_markdown).
    """
    if not question.strip():
        return "Please enter a question.", ""

    year_from_val = int(year_from) if year_from else None
    year_to_val   = int(year_to)   if year_to   else None

    result = chain.run(
        question  = question,
        year_from = year_from_val,
        year_to   = year_to_val,
        verbose   = False,
    )

    # Format sources as a readable markdown table
    sources_md = "### Retrieved Sources\n\n"
    sources_md += "| # | PMID | Year | Journal | Relevance |\n"
    sources_md += "|---|------|------|---------|----------|\n"
    for i, s in enumerate(result["sources"], 1):
        relevance = f"{(1 - s['distance']):.3f}"
        sources_md += (
            f"| {i} | [{s['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{s['pmid']}) "
            f"| {s['year']} | {s['journal'][:40]} | {relevance} |\n"
        )

    # Add full titles below table
    sources_md += "\n**Abstract titles:**\n"
    for i, s in enumerate(result["sources"], 1):
        sources_md += f"\n{i}. {s['title']}"

    return result["answer"], sources_md


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Sepsis RAG — PubMed Literature Assistant") as demo:

        gr.Markdown("""
# 🏥 Sepsis Literature Assistant
**Retrieval-Augmented Generation over PubMed abstracts**

Ask clinical questions about sepsis and septic shock management. 
Answers are grounded exclusively in retrieved PubMed literature — 
every claim is traceable to a specific PMID.

**Corpus:** ~925 PubMed abstracts | **Embedding:** PubMedBERT | 
**Vector DB:** ChromaDB | **LLM:** Llama 3.1 8B via Ollama
        """)

        # Evaluation metrics banner
        gr.Markdown(f"""
| Metric | Score |
|--------|-------|
| Faithfulness | **{EVAL_SUMMARY['faithfulness']:.3f}** |
| Answer Relevancy | **{EVAL_SUMMARY['answer_relevancy']:.3f}** |
| Context Precision | **{EVAL_SUMMARY['context_precision']:.3f}** |
""")

        gr.Markdown("*Scores from 10-question evaluation set using LLM-as-judge (Llama 3.1 8B)*")

        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label       = "Clinical Question",
                    placeholder = "e.g. What is the first-line vasopressor for septic shock?",
                    lines       = 3,
                )
                with gr.Row():
                    year_from = gr.Number(
                        label   = "Published From (year)",
                        value   = None,
                        minimum = 1990,
                        maximum = 2026,
                    )
                    year_to = gr.Number(
                        label   = "Published To (year)",
                        value   = None,
                        minimum = 1990,
                        maximum = 2026,
                    )
                submit_btn = gr.Button("Ask", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("**Example questions:**")
                for q in EXAMPLE_QUESTIONS:
                    gr.Button(q, size="sm").click(
                        fn      = lambda x=q: x,
                        outputs = question_input,
                    )

        with gr.Row():
            with gr.Column():
                answer_output = gr.Textbox(
                    label      = "Generated Answer",
                    lines      = 10,
                    interactive= False,
                )
            with gr.Column():
                sources_output = gr.Markdown(label="Sources")

        submit_btn.click(
            fn      = run_query,
            inputs  = [question_input, year_from, year_to],
            outputs = [answer_output, sources_output],
        )
        question_input.submit(
            fn      = run_query,
            inputs  = [question_input, year_from, year_to],
            outputs = [answer_output, sources_output],
        )

        gr.Markdown("""
---
**How it works:** Questions are embedded with PubMedBERT and compared against 
925 indexed PubMed abstracts using cosine similarity. The top 5 most relevant 
abstracts are retrieved and passed as context to an LLM, which generates a 
grounded answer citing source PMIDs.

**Portfolio project by Patrick Bonsu** — biomedical engineer transitioning to AI/ML.
[GitHub](https://github.com/Patrick-Bonsu/rag-pubmed-sepsis)
        """)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=False, theme=gr.themes.Soft())