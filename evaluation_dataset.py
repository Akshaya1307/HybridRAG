EVALUATION_QUERIES = [

    # ------------------------------------------------
    # Keyword-focused queries
    # ------------------------------------------------

    {
        "question": "How does the rule engine work?",
        "query_type": "Keyword-focused",
        "relevant_chunk_ids": [9]
    },

    {
        "question": "What machine learning techniques are used?",
        "query_type": "Keyword-focused",
        "relevant_chunk_ids": [9, 10]
    },

    # ------------------------------------------------
    # Semantic / conceptual queries
    # ------------------------------------------------

    {
        "question": "What problem does the system address?",
        "query_type": "Semantic / Conceptual",
        "relevant_chunk_ids": [4]
    },

    {
        "question": "What are the primary objectives of the system?",
        "query_type": "Semantic / Conceptual",
        "relevant_chunk_ids": [5]
    },

    {
        "question": "What are the limitations of the system?",
        "query_type": "Semantic / Conceptual",
        "relevant_chunk_ids": [13]
    },

    {
        "question": "What are the results of the hybrid system?",
        "query_type": "Semantic / Conceptual",
        "relevant_chunk_ids": [11]
    },

    # ------------------------------------------------
    # Mixed / technical queries
    # ------------------------------------------------

    {
        "question": "What are the two main components of the system architecture?",
        "query_type": "Mixed / Technical",
        "relevant_chunk_ids": [6]
    },

    {
        "question": "How does the hybrid system classify transactions?",
        "query_type": "Mixed / Technical",
        "relevant_chunk_ids": [7]
    }
]