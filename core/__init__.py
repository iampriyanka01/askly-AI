"""
Askly core package - a RAG helpdesk pipeline.

Modules (in pipeline order):
    ingestion           - PDF / Markdown / text -> section-aware chunks with citation metadata
    vector_store        - dense retrieval: embeddings in FAISS (local) or Pinecone
    hybrid_indexer      - BM25 sparse retrieval fused with dense results (Reciprocal Rank Fusion)
    re_ranker           - cross-encoder / LLM / lexical reranking -> relevance 0-1
    query_understanding - intent detection + follow-up rewriting ("what about sick leave?")
    search_router       - shallow vs deep (multi-query) retrieval, personalisation boosts
    gap_detector        - knowledge-gap detection, escalation tickets, team routing
    feedback            - thumbs up/down store -> retrieval boosts
    memory_rag          - per-session conversation memory + user profile
    rag_pipeline        - AsklyAssistant: orchestrates one conversational turn
    evaluator           - RAGAS metrics (+ offline proxies) on a labelled test set
"""
