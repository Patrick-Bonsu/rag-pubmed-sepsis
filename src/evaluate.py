import json
import re
from pathlib import Path

import pandas as pd
from langchain_ollama import OllamaLLM

from src.chain import RAGChain
from src.retriever import SepsisRetriever

RESULTS_PATH = Path("data/processed/evaluation_results.json")
JUDGE_MODEL  = "llama3.1:8b"

TEST_QUESTIONS = [
    {
        "question": "What is the recommended first-line vasopressor for septic shock?",
        "ground_truth": "Norepinephrine is the recommended first-line vasopressor for septic shock per Surviving Sepsis Campaign guidelines.",
    },
    {
        "question": "What is the Sepsis-3 definition of septic shock?",
        "ground_truth": "Septic shock is defined as sepsis with circulatory and cellular/metabolic dysfunction, requiring vasopressors to maintain MAP ≥65 mmHg and serum lactate >2 mmol/L despite adequate fluid resuscitation.",
    },
    {
        "question": "What initial fluid resuscitation volume is recommended in septic shock?",
        "ground_truth": "Current guidelines recommend approximately 30 mL/kg of intravenous crystalloid fluid within the first 3 hours of sepsis resuscitation.",
    },
    {
        "question": "What are the main criteria used in the SOFA score for sepsis diagnosis?",
        "ground_truth": "The SOFA score assesses six organ systems: respiratory, coagulation, liver, cardiovascular, CNS, and renal function.",
    },
    {
        "question": "What is the role of lactate monitoring in sepsis management?",
        "ground_truth": "Lactate is used as a marker of tissue hypoperfusion. Lactate clearance guides resuscitation adequacy, and persistent elevation is associated with worse outcomes.",
    },
    {
        "question": "What are the known adverse effects of norepinephrine in septic shock?",
        "ground_truth": "Norepinephrine can cause peripheral vasoconstriction, limb ischemia, cardiac arrhythmias, and increased cardiac afterload.",
    },
    {
        "question": "When is vasopressin added to norepinephrine in septic shock treatment?",
        "ground_truth": "Vasopressin is added when norepinephrine doses exceed 0.25-0.5 mcg/kg/min as a catecholamine-sparing strategy.",
    },
    {
        "question": "What balanced crystalloid solutions are preferred for sepsis fluid resuscitation?",
        "ground_truth": "Balanced crystalloids such as lactated Ringer's solution and Plasma-Lyte are preferred over normal saline to avoid hyperchloremic metabolic acidosis.",
    },
    {
        "question": "How is fluid responsiveness assessed in critically ill sepsis patients?",
        "ground_truth": "Fluid responsiveness is assessed using dynamic measures such as pulse pressure variation, passive leg raise test, and end-expiratory occlusion test.",
    },
    {
        "question": "What are the mortality outcomes associated with septic shock in ICU patients?",
        "ground_truth": "Septic shock carries a hospital mortality rate of approximately 40-50%, varying by severity, organ dysfunction, and patient comorbidities.",
    },
]


def extract_score(text: str) -> float:
    """Extract a 0.0–1.0 score from LLM judge output using regex."""
    match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    if match:
        return round(float(match.group()), 3)
    return 0.0


def score_faithfulness(llm: OllamaLLM, answer: str, context: str) -> tuple[float, str]:
    """
    Faithfulness: Does the answer stay grounded in the retrieved context?
    Score of 1.0 = fully grounded, 0.0 = contradicts or fabricates beyond context.
    """
    prompt = f"""You are an evaluation judge. Your task is to assess whether an AI-generated
answer is faithful to the provided source context.

A faithful answer:
- Only makes claims that are directly supported by the context
- Does not introduce facts not present in the context
- Does not contradict the context

Context:
{context}

Answer to evaluate:
{answer}

Rate the faithfulness of this answer on a scale from 0.0 to 1.0, where:
1.0 = fully faithful, every claim is supported by the context
0.5 = partially faithful, some claims are unsupported
0.0 = unfaithful, contradicts or fabricates beyond the context

Respond with ONLY a number between 0.0 and 1.0, then one sentence of explanation."""

    response = llm.invoke(prompt)
    return extract_score(response), response.strip()


def score_answer_relevancy(
    llm: OllamaLLM, question: str, answer: str
) -> tuple[float, str]:
    """
    Answer relevancy: Does the answer actually address the question asked?
    Score of 1.0 = directly and completely answers the question.
    """
    prompt = f"""You are an evaluation judge. Assess whether the following answer
adequately addresses the question.

Question: {question}

Answer: {answer}

Rate how well the answer addresses the question on a scale from 0.0 to 1.0, where:
1.0 = directly and completely answers the question
0.5 = partially answers the question or includes significant irrelevant content
0.0 = does not answer the question at all

Respond with ONLY a number between 0.0 and 1.0, then one sentence of explanation."""

    response = llm.invoke(prompt)
    return extract_score(response), response.strip()


def score_context_precision(
    llm: OllamaLLM, question: str, context: str
) -> tuple[float, str]:
    """
    Context precision: Were the retrieved chunks relevant to the question?
    Score of 1.0 = all retrieved context is highly relevant.
    """
    prompt = f"""You are an evaluation judge. Assess whether the following retrieved
context is relevant and useful for answering the question.

Question: {question}

Retrieved context:
{context}

Rate the relevance of this context to the question on a scale from 0.0 to 1.0, where:
1.0 = all retrieved content is directly relevant to answering the question
0.5 = some retrieved content is relevant, some is off-topic
0.0 = retrieved content is not relevant to the question

Respond with ONLY a number between 0.0 and 1.0, then one sentence of explanation."""

    response = llm.invoke(prompt)
    return extract_score(response), response.strip()


def evaluate_sample(
    llm: OllamaLLM,
    retriever: SepsisRetriever,
    question: str,
    answer: str,
    sources: list[dict],
) -> dict:
    """Run all three metrics for a single question/answer pair."""
    retrieved   = retriever.retrieve(question)
    context_str = retriever.format_for_prompt(retrieved)

    faith_score,  faith_reason  = score_faithfulness(llm, answer, context_str)
    relev_score,  relev_reason  = score_answer_relevancy(llm, question, answer)
    prec_score,   prec_reason   = score_context_precision(llm, question, context_str)

    return {
        "faithfulness":        faith_score,
        "answer_relevancy":    relev_score,
        "context_precision":   prec_score,
        "faith_reason":        faith_reason,
        "relev_reason":        relev_reason,
        "prec_reason":         prec_reason,
    }


def main():
    print("Initializing RAG chain and judge LLM...")
    chain     = RAGChain(n_results=5)
    judge_llm = OllamaLLM(model=JUDGE_MODEL, temperature=0)
    retriever = SepsisRetriever(n_results=5)

    results = []

    for i, item in enumerate(TEST_QUESTIONS):
        print(f"\n[{i + 1}/{len(TEST_QUESTIONS)}] {item['question'][:70]}...")

        # Generate answer
        rag_result = chain.run(question=item["question"], verbose=False)
        answer     = rag_result["answer"]
        sources    = rag_result["sources"]
        print(f"  Answer generated ({len(answer)} chars). Scoring...")

        # Score with three metrics
        scores = evaluate_sample(
            judge_llm, retriever, item["question"], answer, sources
        )

        print(
            f"  faithfulness={scores['faithfulness']:.2f}  "
            f"relevancy={scores['answer_relevancy']:.2f}  "
            f"precision={scores['context_precision']:.2f}"
        )

        results.append({
            "question":          item["question"],
            "ground_truth":      item["ground_truth"],
            "answer":            answer,
            "faithfulness":      scores["faithfulness"],
            "answer_relevancy":  scores["answer_relevancy"],
            "context_precision": scores["context_precision"],
            "faith_reason":      scores["faith_reason"],
            "relev_reason":      scores["relev_reason"],
            "prec_reason":       scores["prec_reason"],
        })

    # Aggregate summary
    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        mean = df[metric].mean()
        print(f"  {metric:25s}: {mean:.3f}")

    print("\nPer-question scores:")
    pd.set_option("display.max_colwidth", 50)
    print(
        df[["question", "faithfulness", "answer_relevancy", "context_precision"]]
        .to_string(index=False)
    )

    # Save full results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(RESULTS_PATH, orient="records", indent=2)
    print(f"\nFull results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()