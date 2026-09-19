def calculate_metrics(retrieved_ids, relevant_ids):
    retrieved_ids = set(retrieved_ids)
    relevant_ids = set(relevant_ids)

    true_positives = len(retrieved_ids & relevant_ids)

    precision = (
        true_positives / len(retrieved_ids)
        if retrieved_ids
        else 0
    )

    recall = (
        true_positives / len(relevant_ids)
        if relevant_ids
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_retrieval(
    queries,
    chunks,
    vector_search_function,
    bm25_search_function,
    hybrid_search_function,
    embedding_model,
    embeddings,
    bm25_index,
    k=5
):
    results = {
        "BM25": [],
        "Semantic": [],
        "Hybrid": []
    }

    for item in queries:

        question = item["question"]
        query_type = item["query_type"]
        relevant_ids = item["relevant_chunk_ids"]

        # --------------------------------------------
        # Semantic Retrieval
        # --------------------------------------------

        semantic_results = vector_search_function(
            question,
            chunks,
            embeddings,
            embedding_model,
            k=k
        )

        semantic_ids = [
            result["chunk"]["chunk_id"]
            for result in semantic_results
        ]

        # --------------------------------------------
        # BM25 Retrieval
        # --------------------------------------------

        bm25_results = bm25_search_function(
            question,
            chunks,
            bm25_index,
            k=k
        )

        bm25_ids = [
            result["chunk"]["chunk_id"]
            for result in bm25_results
        ]

        # --------------------------------------------
        # Hybrid Retrieval
        # --------------------------------------------

        hybrid_results = hybrid_search_function(
            semantic_results,
            bm25_results
        )

        hybrid_ids = [
            result["chunk"]["chunk_id"]
            for result in hybrid_results[:k]
        ]

        # --------------------------------------------
        # Calculate Metrics
        # --------------------------------------------

        bm25_metrics = calculate_metrics(
            bm25_ids,
            relevant_ids
        )

        semantic_metrics = calculate_metrics(
            semantic_ids,
            relevant_ids
        )

        hybrid_metrics = calculate_metrics(
            hybrid_ids,
            relevant_ids
        )

        # --------------------------------------------
        # Store Query-Level Results
        # --------------------------------------------

        results["BM25"].append({
            "question": question,
            "query_type": query_type,
            **bm25_metrics
        })

        results["Semantic"].append({
            "question": question,
            "query_type": query_type,
            **semantic_metrics
        })

        results["Hybrid"].append({
            "question": question,
            "query_type": query_type,
            **hybrid_metrics
        })

    return results


def average_metrics(metrics_list):

    if not metrics_list:
        return {
            "precision": 0,
            "recall": 0,
            "f1": 0
        }

    return {
        "precision": sum(
            item["precision"]
            for item in metrics_list
        ) / len(metrics_list),

        "recall": sum(
            item["recall"]
            for item in metrics_list
        ) / len(metrics_list),

        "f1": sum(
            item["f1"]
            for item in metrics_list
        ) / len(metrics_list)
    }


def average_metrics_by_query_type(results):

    query_types = sorted(
        set(
            item["query_type"]
            for method_results in results.values()
            for item in method_results
        )
    )

    grouped_results = {}

    for query_type in query_types:

        grouped_results[query_type] = {}

        for method, method_results in results.items():

            filtered_results = [
                item
                for item in method_results
                if item["query_type"] == query_type
            ]

            grouped_results[query_type][method] = average_metrics(
                filtered_results
            )

    return grouped_results