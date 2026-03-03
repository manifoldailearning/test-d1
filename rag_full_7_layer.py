from dotenv import load_dotenv

from src.evaluation import GOLDEN_QUERIES, evaluate_retrieval, evaluate_ragas


def main() -> None:
    # Load environment variables (API keys, etc.)
    load_dotenv()

    # Run retrieval evaluation
    retrieval_metrics = evaluate_retrieval(GOLDEN_QUERIES)

    print("=== RETRIEVAL METRICS (7-LAYER RAG) ===")
    print(f"Queries:        {retrieval_metrics['num_queries']}")
    print(f"Precision@5:    {retrieval_metrics['precision_at_5']:.2f}")
    print(f"Recall@5:       {retrieval_metrics['recall_at_5']:.2f}")
    print(f"MRR:            {retrieval_metrics['mrr']:.2f}")
    print()

    # Run (stubbed) RAGAS-style generation evaluation
    ragas_metrics = evaluate_ragas(GOLDEN_QUERIES)

    print("=== RAGAS METRICS (GENERATION) ===")
    print(f"Faithfulness:        {ragas_metrics['faithfulness']:.2f}")
    print(f"AnswerRelevancy:     {ragas_metrics['answer_relevancy']:.2f}")
    print(f"ContextPrecision:    {ragas_metrics['context_precision']:.2f}")
    print(f"ContextRecall:       {ragas_metrics['context_recall']:.2f}")
    print()
    print(f"Lowest metric:       {ragas_metrics['lowest_metric']}")
    print(f"Root cause layer:    {ragas_metrics['root_cause_layer']}")


if __name__ == "__main__":
    main()

