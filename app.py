import os
import tempfile

import streamlit as st

from src.ingestion import extract_text_from_pdf
from src.chunking import create_chunks
from src.embeddings import load_embedding_model, create_embeddings
from src.vector_search import retrieve_top_k
from src.bm25_search import build_bm25_index, bm25_search
from src.hybrid_search import reciprocal_rank_fusion
from src.generator import generate_answer

from src.evaluation import (
    evaluate_retrieval,
    average_metrics,
    average_metrics_by_query_type
)
from evaluation_dataset import EVALUATION_QUERIES


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HybridRAG",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-top: 0;
        margin-bottom: 25px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .source-box {
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

SAMPLE_PDF = "data/documents/sample.pdf"
TOP_K = 3


# ============================================================
# SESSION STATE
# ============================================================

if "pages" not in st.session_state:
    st.session_state.pages = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "bm25" not in st.session_state:
    st.session_state.bm25 = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_ready" not in st.session_state:
    st.session_state.document_ready = False


# ============================================================
# CACHED EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<p class="main-title">🔎 HybridRAG</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Hybrid Retrieval-Augmented Generation for intelligent document search'
    '</p>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Document Settings")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"]
    )

    st.divider()

    st.markdown("### 🔀 Retrieval Pipeline")

    st.markdown(
        """
        **Keyword Retrieval**
        
        BM25 finds exact and important terms.

        **Semantic Retrieval**
        
        Embeddings find semantically related content.

        **Hybrid Retrieval**
        
        RRF combines both retrieval methods.

        **Generation**
        
        Qwen3 generates an answer from the retrieved context.
        """
    )

    st.divider()

    st.caption("HybridRAG • Local Qwen + Hybrid Retrieval")


# ============================================================
# DOCUMENT PROCESSING FUNCTION
# ============================================================

def process_document(file_path, document_name):

    with st.status(
        "Processing document...",
        expanded=True
    ) as status:

        # ----------------------------------------------------
        # Extract PDF text
        # ----------------------------------------------------

        st.write("📖 Extracting text from PDF...")

        pages = extract_text_from_pdf(file_path)

        if not pages:
            status.update(
                label="❌ No text found in PDF",
                state="error"
            )

            return False

        st.write(f"✓ Extracted {len(pages)} pages")


        # ----------------------------------------------------
        # Create chunks
        # ----------------------------------------------------

        st.write("✂️ Creating document chunks...")

        chunks = create_chunks(pages)

        st.write(f"✓ Created {len(chunks)} chunks")


        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        st.write("🧠 Loading embedding model...")

        model = get_embedding_model()

        st.write("✓ Embedding model loaded")


        # ----------------------------------------------------
        # Create embeddings
        # ----------------------------------------------------

        st.write("🔢 Creating embeddings...")

        embeddings = create_embeddings(
            chunks,
            model
        )

        st.write(
            f"✓ Embeddings created: {embeddings.shape}"
        )


        # ----------------------------------------------------
        # Build BM25 index
        # ----------------------------------------------------

        st.write("🔤 Building BM25 index...")

        bm25 = build_bm25_index(chunks)

        st.write("✓ BM25 index ready")


        # ----------------------------------------------------
        # Store everything in session state
        # ----------------------------------------------------

        st.session_state.pages = pages
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.bm25 = bm25
        st.session_state.document_name = document_name
        st.session_state.document_ready = True


        status.update(
            label="✅ Document ready!",
            state="complete"
        )

    return True


# ============================================================
# HANDLE UPLOADED PDF
# ============================================================

if uploaded_file is not None:

    # Only process when a new document is uploaded
    if (
        st.session_state.document_name
        != uploaded_file.name
    ):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name


        process_document(
            temp_path,
            uploaded_file.name
        )


        # Remove temporary file
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ============================================================
# SAMPLE PDF FALLBACK
# ============================================================

elif (
    not st.session_state.document_ready
    and os.path.exists(SAMPLE_PDF)
):

    st.info(
        "📄 No PDF uploaded. Using the bundled sample document."
    )

    process_document(
        SAMPLE_PDF,
        "sample.pdf"
    )


# ============================================================
# DOCUMENT STATUS
# ============================================================

if st.session_state.document_ready:

    st.success(
        f"📄 Document loaded: "
        f"**{st.session_state.document_name}**"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Pages",
            len(st.session_state.pages)
        )

    with col2:
        st.metric(
            "Chunks",
            len(st.session_state.chunks)
        )

    with col3:
        st.metric(
            "Embedding Size",
            st.session_state.embeddings.shape[1]
        )


# ============================================================
# QUESTION SECTION
# ============================================================

if st.session_state.document_ready:

    st.divider()

    st.subheader("💬 Ask your document")

    question = st.text_input(
        "Enter your question",
        placeholder="Example: How does the hybrid architecture classify financial transactions?"
    )

    search_button = st.button(
        "🔍 Search & Generate Answer",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # SEARCH + GENERATION
    # ========================================================

    if search_button:

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            # ------------------------------------------------
            # Retrieve components
            # ------------------------------------------------

            model = get_embedding_model()

            chunks = st.session_state.chunks
            embeddings = st.session_state.embeddings
            bm25 = st.session_state.bm25


            # ------------------------------------------------
            # Semantic Search
            # ------------------------------------------------

            with st.spinner(
                "🧠 Running semantic search..."
            ):

                vector_results = retrieve_top_k(
                    question,
                    chunks,
                    embeddings,
                    model,
                    k=5
                )


            # ------------------------------------------------
            # BM25 Search
            # ------------------------------------------------

            with st.spinner(
                "🔤 Running BM25 keyword search..."
            ):

                bm25_results = bm25_search(
                    question,
                    chunks,
                    bm25,
                    k=5
                )


            # ------------------------------------------------
            # Hybrid RRF
            # ------------------------------------------------

            with st.spinner(
                "🔀 Combining retrieval results..."
            ):

                hybrid_results = reciprocal_rank_fusion(
                    vector_results,
                    bm25_results
                )


            # ------------------------------------------------
            # Retrieval Comparison
            # ------------------------------------------------

            bm25_ids = {
                result["chunk"]["chunk_id"]
                for result in bm25_results
            }

            semantic_ids = {
                result["chunk"]["chunk_id"]
                for result in vector_results
            }

            hybrid_ids = {
                result["chunk"]["chunk_id"]
                for result in hybrid_results[:TOP_K]
            }

            common_ids = bm25_ids & semantic_ids
            bm25_only_ids = bm25_ids - semantic_ids
            semantic_only_ids = semantic_ids - bm25_ids


            # Top results for Qwen
            top_results = hybrid_results[:TOP_K]


            # ------------------------------------------------
            # Generate answer
            # ------------------------------------------------

            with st.spinner(
                "🤖 Qwen is generating the answer..."
            ):

                answer = generate_answer(
                    question,
                    top_results
                )


            # =================================================
            # ANSWER
            # =================================================

            st.subheader("💡 Answer")

            st.markdown(
                f"""
                <div class="answer-box">

                {answer}

                </div>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # SOURCES
            # =================================================

            source_pages = sorted(
                set(
                    result["chunk"]["page_number"]
                    for result in top_results
                )
            )

            st.subheader("📚 Sources")

            st.write(
                ", ".join(
                    f"Page {page}"
                    for page in source_pages
                )
            )


            # =================================================
            # RETRIEVAL DETAILS
            # =================================================

            st.divider()

            st.subheader("🔬 Retrieval Details")

            bm25_tab, semantic_tab, hybrid_tab = st.tabs(
                [
                    "🔤 BM25",
                    "🧠 Semantic",
                    "🔀 Hybrid"
                ]
            )


            # ------------------------------------------------
            # BM25 TAB
            # ------------------------------------------------

            with bm25_tab:

                st.caption(
                    "Keyword-based retrieval using BM25"
                )

                for rank, result in enumerate(
                    bm25_results[:TOP_K],
                    start=1
                ):

                    chunk = result["chunk"]

                    st.markdown(
                        f"""
                        **Rank {rank}**  
                        Page: {chunk['page_number']}  
                        Chunk: {chunk['chunk_id']}  
                        BM25 Score: {result['score']:.4f}
                        """
                    )

                    st.write(
                        chunk["text"]
                    )

                    st.divider()


            # ------------------------------------------------
            # SEMANTIC TAB
            # ------------------------------------------------

            with semantic_tab:

                st.caption(
                    "Semantic retrieval using embeddings and cosine similarity"
                )

                for rank, result in enumerate(
                    vector_results[:TOP_K],
                    start=1
                ):

                    chunk = result["chunk"]

                    st.markdown(
                        f"""
                        **Rank {rank}**  
                        Page: {chunk['page_number']}  
                        Chunk: {chunk['chunk_id']}  
                        Similarity: {result['score']:.4f}
                        """
                    )

                    st.write(
                        chunk["text"]
                    )

                    st.divider()


            # ------------------------------------------------
            # HYBRID TAB
            # ------------------------------------------------

            with hybrid_tab:

                st.markdown("### 🔀 Hybrid Retrieval Ranking")

                for result in hybrid_results[:TOP_K]:

                    chunk = result["chunk"]

                    st.markdown(
                        f"**Hybrid Rank {result['rank']}**"
                    )

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "BM25 Rank",
                            result["bm25_rank"]
                            if result["bm25_rank"] is not None
                            else "—"
                        )

                    with col2:
                        st.metric(
                            "Semantic Rank",
                            result["semantic_rank"]
                            if result["semantic_rank"] is not None
                            else "—"
                        )

                    with col3:
                        st.metric(
                            "RRF Score",
                            f"{result['rrf_score']:.4f}"
                        )

                    with col4:
                        st.metric(
                            "Page",
                            chunk["page_number"]
                        )

                    st.caption(
                        f"Chunk {chunk['chunk_id']}"
                    )

                    st.write(chunk["text"])

                    st.divider()


            # =================================================
            # RETRIEVAL COMPARISON
            # =================================================

            st.divider()

            st.subheader("📊 Retrieval Comparison")

            st.caption(
                "Comparison of chunks retrieved by BM25 and Semantic Search."
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("### 🔗 Common Chunks")

                if common_ids:
                    st.write(
                        ", ".join(
                            str(chunk_id)
                            for chunk_id in sorted(common_ids)
                        )
                    )
                else:
                    st.write("No common chunks.")

            with col2:

                st.markdown("### 🔤 BM25 Only")

                if bm25_only_ids:
                    st.write(
                        ", ".join(
                            str(chunk_id)
                            for chunk_id in sorted(bm25_only_ids)
                        )
                    )
                else:
                    st.write("No BM25-only chunks.")


            col3, col4 = st.columns(2)

            with col3:

                st.markdown("### 🧠 Semantic Only")

                if semantic_only_ids:
                    st.write(
                        ", ".join(
                            str(chunk_id)
                            for chunk_id in sorted(semantic_only_ids)
                        )
                    )
                else:
                    st.write("No Semantic-only chunks.")

            with col4:

                st.markdown("### 🔀 Hybrid Results")

                if hybrid_ids:
                    st.write(
                        ", ".join(
                            str(chunk_id)
                            for chunk_id in sorted(hybrid_ids)
                        )
                    )
                else:
                    st.write("No hybrid results.")


# ============================================================
# RETRIEVAL EVALUATION
# ============================================================

if st.session_state.document_ready:

    st.divider()

    st.subheader("📊 Retrieval Evaluation")

    st.write(
        "Evaluate BM25, Semantic, and Hybrid retrieval using "
        "Precision, Recall, and F1-score across different "
        "types of queries."
    )

    if st.button(
        "Run Retrieval Evaluation",
        type="secondary"
    ):

        with st.spinner(
            "Evaluating retrieval methods..."
        ):

            evaluation_results = evaluate_retrieval(
                queries=EVALUATION_QUERIES,
                chunks=st.session_state.chunks,
                vector_search_function=retrieve_top_k,
                bm25_search_function=bm25_search,
                hybrid_search_function=reciprocal_rank_fusion,
                embedding_model=get_embedding_model(),
                embeddings=st.session_state.embeddings,
                bm25_index=st.session_state.bm25,
                k=5
            )

        st.success(
            "Retrieval evaluation completed!"
        )

        # ====================================================
        # OVERALL PERFORMANCE
        # ====================================================

        st.markdown("### 📈 Overall Retrieval Performance")

        bm25_avg = average_metrics(
            evaluation_results["BM25"]
        )

        semantic_avg = average_metrics(
            evaluation_results["Semantic"]
        )

        hybrid_avg = average_metrics(
            evaluation_results["Hybrid"]
        )

        evaluation_table = {
            "Retrieval Method": [
                "BM25",
                "Semantic",
                "Hybrid"
            ],

            "Precision": [
                round(
                    bm25_avg["precision"],
                    3
                ),
                round(
                    semantic_avg["precision"],
                    3
                ),
                round(
                    hybrid_avg["precision"],
                    3
                )
            ],

            "Recall": [
                round(
                    bm25_avg["recall"],
                    3
                ),
                round(
                    semantic_avg["recall"],
                    3
                ),
                round(
                    hybrid_avg["recall"],
                    3
                )
            ],

            "F1-Score": [
                round(
                    bm25_avg["f1"],
                    3
                ),
                round(
                    semantic_avg["f1"],
                    3
                ),
                round(
                    hybrid_avg["f1"],
                    3
                )
            ]
        }

        st.dataframe(
            evaluation_table,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # QUERY TYPE PERFORMANCE
        # ====================================================

        st.markdown(
            "### 🔍 Performance by Query Type"
        )

        grouped_results = average_metrics_by_query_type(
            evaluation_results
        )

        query_type_rows = []

        for query_type, methods in grouped_results.items():

            for method in [
                "BM25",
                "Semantic",
                "Hybrid"
            ]:

                metrics = methods[method]

                query_type_rows.append({
                    "Query Type": query_type,
                    "Retrieval Method": method,
                    "Precision": round(
                        metrics["precision"],
                        3
                    ),
                    "Recall": round(
                        metrics["recall"],
                        3
                    ),
                    "F1-Score": round(
                        metrics["f1"],
                        3
                    )
                })

        st.dataframe(
            query_type_rows,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # QUERY-LEVEL RESULTS
        # ====================================================

        st.markdown(
            "### 📝 Query-Level Comparison"
        )

        query_comparison = []

        for i in range(
            len(evaluation_results["BM25"])
        ):

            bm25_item = evaluation_results["BM25"][i]
            semantic_item = evaluation_results["Semantic"][i]
            hybrid_item = evaluation_results["Hybrid"][i]

            query_comparison.append({
                "Query Type": bm25_item["query_type"],
                "Question": bm25_item["question"],

                "BM25 F1": round(
                    bm25_item["f1"],
                    3
                ),

                "Semantic F1": round(
                    semantic_item["f1"],
                    3
                ),

                "Hybrid F1": round(
                    hybrid_item["f1"],
                    3
                )
            })

        st.dataframe(
            query_comparison,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Metrics are calculated from the predefined "
            "evaluation questions, query categories, and "
            "their manually identified relevant chunk IDs."
        )