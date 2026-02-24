from __future__ import annotations
from typing import Dict, List, Optional, Union
from langchain_core.documents import Document
from langchain_chroma import Chroma


class Retrievers:
    """Semantic search over domain knowledge with optional metadata filtering.

    Supports three retrieval modes:
    - ``"dense"``: Chroma embedding similarity search only.
    - ``"bm25"``: BM25 keyword search only (requires ``documents`` arg).
    - ``"hybrid"``: BM25 + Dense fusion via Reciprocal Rank Fusion (RRF).

    하위 호환성: ``documents=None`` 이면 mode와 무관하게 dense-only fallback.
    """

    def __init__(
        self,
        vectorstore: Chroma,
        documents: Optional[List[Document]] = None,
        mode: str = "hybrid",
        rrf_k: int = 60,
    ):
        self.vectorstore = vectorstore
        self.documents = documents or []
        self.mode = mode
        self.rrf_k = rrf_k
        self._bm25 = None

        if self.documents:
            self._init_bm25()

    def _init_bm25(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "rank_bm25 is required for BM25/hybrid mode. "
                "Install with: uv pip install rank_bm25"
            ) from e
        tokenized = [doc.page_content.lower().split() for doc in self.documents]
        self._bm25 = BM25Okapi(tokenized)

    def filter_documents(
        self,
        dataset: Optional[Union[str, List[str]]] = None,
        category: Optional[Union[str, List[str]]] = None,
        defect_type: Optional[str] = None,
    ) -> List[Document]:
        """Python-side metadata filtering for BM25 candidate set."""
        filtered = []
        for doc in self.documents:
            meta = doc.metadata
            if dataset:
                allowed = [dataset] if isinstance(dataset, str) else dataset
                if meta.get("dataset") not in allowed:
                    continue
            if category:
                allowed = [category] if isinstance(category, str) else category
                if meta.get("category") not in allowed:
                    continue
            if defect_type:
                if meta.get("defect_type") != defect_type:
                    continue
            filtered.append(doc)
        return filtered

    def retrieve_bm25(
        self,
        query: str,
        dataset: Optional[Union[str, List[str]]] = None,
        category: Optional[Union[str, List[str]]] = None,
        defect_type: Optional[str] = None,
        k: int = 3,
    ) -> List[Document]:
        """BM25 search on metadata-filtered document subset."""
        if not self.documents:
            return []

        filtered = self.filter_documents(dataset=dataset, category=category, defect_type=defect_type)
        if not filtered:
            filtered = self.documents  # fallback: no filter match → search all

        from rank_bm25 import BM25Okapi
        tokenized = [doc.page_content.lower().split() for doc in filtered]
        bm25 = BM25Okapi(tokenized)
        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)

        ranked = sorted(zip(scores, filtered), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:k]]

    def fuse_rrf(
        self,
        dense_docs: List[Document],
        bm25_docs: List[Document],
    ) -> List[Document]:
        """Reciprocal Rank Fusion of dense and BM25 results.

        RRF score = Σ 1 / (rrf_k + rank_i).  Deduplication by page_content.
        """
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs, 1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_docs, 1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
            doc_map[key] = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in ranked]

    def retrieve(
        self,
        query: str,
        dataset: Optional[Union[str, List[str]]] = None,
        category: Optional[Union[str, List[str]]] = None,
        defect_type: Optional[str] = None,
        k: int = 3,
        mode: Optional[str] = None,
    ) -> List[Document]:
        """Search for relevant domain knowledge documents.

        Args:
            query: Free-text query describing the defect or context.
            dataset: Filter by dataset name(s). str or list of str.
            category: Filter by category name(s). str or list of str.
            defect_type: Exact defect type filter. ``None`` = no filter.
            k: Number of results to return.
            mode: Override instance-level mode for this call.

        Returns:
            List of matching Document objects.
        """
        effective_mode = mode or self.mode

        # documents 없으면 항상 dense fallback
        if not self.documents:
            effective_mode = "dense"

        if effective_mode == "dense":
            return self._retrieve_dense(query, dataset=dataset, category=category, defect_type=defect_type, k=k)
        elif effective_mode == "bm25":
            return self.retrieve_bm25(query, dataset=dataset, category=category, defect_type=defect_type, k=k)
        else:  # hybrid
            dense_docs = self._retrieve_dense(query, dataset=dataset, category=category, defect_type=defect_type, k=k)
            bm25_docs = self.retrieve_bm25(query, dataset=dataset, category=category, defect_type=defect_type, k=k)
            fused = self.fuse_rrf(dense_docs, bm25_docs)
            return fused[:k]

    def _retrieve_dense(
        self,
        query: str,
        dataset: Optional[Union[str, List[str]]] = None,
        category: Optional[Union[str, List[str]]] = None,
        defect_type: Optional[str] = None,
        k: int = 3,
    ) -> List[Document]:
        where_filter = self.build_filter(dataset=dataset, category=category, defect_type=defect_type)
        kwargs: Dict = {"k": k}
        if where_filter:
            kwargs["filter"] = where_filter
        return self.vectorstore.similarity_search(query, **kwargs)

    def build_filter(
        self,
        dataset: Optional[Union[str, List[str]]] = None,
        category: Optional[Union[str, List[str]]] = None,
        defect_type: Optional[str] = None,
    ) -> Optional[Dict]:
        """Build a Chroma metadata filter dict."""
        conditions = []
        if dataset:
            if isinstance(dataset, list):
                conditions.append({"dataset": {"$in": dataset}})
            else:
                conditions.append({"dataset": dataset})
        if category:
            if isinstance(category, list):
                conditions.append({"category": {"$in": category}})
            else:
                conditions.append({"category": category})
        if defect_type:
            conditions.append({"defect_type": defect_type})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def build_query(self, category: str, defect_type: str) -> str:
        """Build a retrieval query from image path components.

        good 이미지는 결함 쿼리 대신 정상 외관 쿼리를 사용한다.
        'carpet good defect anomaly' 같은 모순된 쿼리를 방지해
        정상 이미지에서 불필요한 결함 문서가 검색되는 것을 막는다.
        """
        if defect_type == "good":
            return f"{category} normal product appearance"
        return f"{category} {defect_type} defect anomaly"

    def build_generic_query(self, category: str) -> str:
        """Build a production-style query without ground-truth defect_type.

        MCQ 평가 시 데이터 유출(data leakage) 방지를 위해
        defect_type 없이 카테고리 기반 generic 쿼리를 생성한다.
        프로덕션 API가 defect_type을 모르는 상태와 동일한 조건으로 검색.
        """
        return f"{category} defect anomaly inspection"

    def format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into a text block for MLLM prompts.

        Args:
            docs: List of retrieved Document objects.

        Returns:
            Formatted string ready to inject into a prompt.
        """
        if not docs:
            return "No relevant domain knowledge found."

        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            header = (
                f"[{i}] Dataset: {meta.get('dataset', 'N/A')} | "
                f"Category: {meta.get('category', 'N/A')} | "
                f"Defect Type: {meta.get('defect_type', 'N/A')}"
            )
            parts.append(f"{header}\n{doc.page_content}")

        return "\n\n".join(parts)
