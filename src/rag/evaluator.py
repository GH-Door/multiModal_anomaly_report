from __future__ import annotations

from typing import Dict, List

from .retriever import Retrievers

# ── 외부 소스(PDF + Web) 검색 품질 테스트 쿼리 ─────────────────────────────────
# 실제 프로덕션 쿼리 형태: defect_type을 모르는 상태에서 카테고리 기반 generic 쿼리
TEST_QUERIES_EXTERNAL: List[Dict] = [
    {
        "query": "pouch seal integrity failure food package",
        "relevant_datasets": ["PackagingGuide", "CFIA_RetortPouch"],
        "relevant_category": None,
        "description": "패키지 씰 결함 (defect_type 미포함)",
    },
    {
        "query": "drink bottle closure defect anomaly inspection",
        "relevant_datasets": ["PackagingGuide"],
        "relevant_category": None,
        "description": "음료 병 캡 결함 (defect_type 미포함)",
    },
    {
        "query": "food packaging label placement defect anomaly",
        "relevant_datasets": ["PackagingGuide"],
        "relevant_category": None,
        "description": "라벨 결함 (defect_type 미포함)",
    },
    {
        "query": "foreign material contamination food surface defect",
        "relevant_datasets": ["PackagingGuide", "CFIA_RetortPouch"],
        "relevant_category": None,
        "description": "이물질/오염 결함 (defect_type 미포함)",
    },
    {
        "query": "retort pouch seal anomaly defect inspection",
        "relevant_datasets": ["CFIA_RetortPouch"],
        "relevant_category": None,
        "description": "CFIA 씰 결함 (defect_type 미포함)",
    },
    {
        "query": "product assembly packaging component defect anomaly",
        "relevant_datasets": ["PackagingGuide"],
        "relevant_category": None,
        "description": "어셈블리 결함 (defect_type 미포함)",
    },
    {
        "query": "sealed package abnormal shape defect inspection",
        "relevant_datasets": ["CFIA_RetortPouch"],
        "relevant_category": None,
        "description": "파우치 형상 이상 (defect_type 미포함)",
    },
    {
        "query": "product outer surface appearance defect anomaly",
        "relevant_datasets": ["PackagingGuide"],
        "relevant_category": None,
        "description": "외관 결함 (defect_type 미포함)",
    },
]

# ── MMAD JSON 지식 회귀 테스트 쿼리 ─────────────────────────────────────────────
# 프로덕션 형태 쿼리: build_generic_query(category) 와 동일한 패턴
# defect_type을 쿼리에 포함하지 않음 → 실제 평가 조건과 일치
TEST_QUERIES_MMAD: List[Dict] = [
    {
        "query": "food_package defect anomaly inspection",
        "relevant_datasets": ["GoodsAD"],
        "relevant_category": "food_package",
        "description": "GoodsAD food_package (generic query)",
    },
    {
        "query": "drink_bottle defect anomaly inspection",
        "relevant_datasets": ["GoodsAD"],
        "relevant_category": "drink_bottle",
        "description": "GoodsAD drink_bottle (generic query)",
    },
    {
        "query": "food_box defect anomaly inspection",
        "relevant_datasets": ["GoodsAD"],
        "relevant_category": "food_box",
        "description": "GoodsAD food_box (generic query)",
    },
    {
        "query": "cigarette_box defect anomaly inspection",
        "relevant_datasets": ["GoodsAD"],
        "relevant_category": "cigarette_box",
        "description": "GoodsAD cigarette_box (generic query)",
    },
    {
        "query": "food_bottle defect anomaly inspection",
        "relevant_datasets": ["GoodsAD"],
        "relevant_category": "food_bottle",
        "description": "GoodsAD food_bottle (generic query)",
    },
    {
        "query": "drink_can defect anomaly inspection",
        "relevant_datasets": ["GoodsAD"],
        "relevant_category": "drink_can",
        "description": "GoodsAD drink_can (generic query)",
    },
    {
        "query": "juice_bottle defect anomaly inspection",
        "relevant_datasets": ["MVTec-LOCO"],
        "relevant_category": "juice_bottle",
        "description": "MVTec-LOCO juice_bottle (generic query)",
    },
    {
        "query": "breakfast_box defect anomaly inspection",
        "relevant_datasets": ["MVTec-LOCO"],
        "relevant_category": "breakfast_box",
        "description": "MVTec-LOCO breakfast_box (generic query)",
    },
    {
        "query": "pushpins defect anomaly inspection",
        "relevant_datasets": ["MVTec-LOCO"],
        "relevant_category": "pushpins",
        "description": "MVTec-LOCO pushpins (generic query)",
    },
    {
        "query": "screw_bag defect anomaly inspection",
        "relevant_datasets": ["MVTec-LOCO"],
        "relevant_category": "screw_bag",
        "description": "MVTec-LOCO screw_bag (generic query)",
    },
]


class RAGEvaluator:
    """RAG 검색 품질 평가 — Hit Rate, MRR.

    Hit 판정 우선순위:
    1. ``relevant_category``가 있으면 category 수준으로 판정 (더 엄격)
    2. 없으면 ``relevant_datasets`` 수준으로 판정 (느슨)

    Args:
        retriever: Retrievers 인스턴스.

    Example::

        ev = RAGEvaluator(Retrievers(vectorstore))
        result = ev.evaluate(TEST_QUERIES_MMAD, k=5)
        print(result["hit_rate"], result["mrr"])

        comparison = ev.compare(
            baseline=RAGEvaluator(Retrievers(baseline_vs)),
            test_queries=TEST_QUERIES_MMAD,
        )
    """

    def __init__(self, retriever: Retrievers):
        self.retriever = retriever

    def evaluate(self, test_queries: List[Dict], k: int = 5) -> Dict:
        """Hit Rate와 MRR을 계산해 반환.

        Args:
            test_queries: ``[{"query", "relevant_datasets", "relevant_category"(optional), "description"}, ...]``
            k: 쿼리당 검색할 최대 문서 수.

        Returns:
            hit_rate, mrr, n_queries, per_query 결과 포함 dict.
        """
        per_query = []
        for item in test_queries:
            docs = self.retriever.retrieve(item["query"], k=k)
            relevant_category = item.get("relevant_category")

            # category 수준 판정 (relevant_category 있을 때)
            if relevant_category:
                retrieved_categories = [d.metadata.get("category", "") for d in docs]
                hit = any(cat == relevant_category for cat in retrieved_categories)
                rr = 0.0
                for rank, cat in enumerate(retrieved_categories, 1):
                    if cat == relevant_category:
                        rr = 1.0 / rank
                        break
                criterion = "category"
            else:
                # dataset 수준 판정 (fallback)
                retrieved_datasets = [d.metadata.get("dataset", "") for d in docs]
                hit = any(ds in item["relevant_datasets"] for ds in retrieved_datasets)
                rr = 0.0
                for rank, ds in enumerate(retrieved_datasets, 1):
                    if ds in item["relevant_datasets"]:
                        rr = 1.0 / rank
                        break
                criterion = "dataset"

            per_query.append({
                "query": item["query"],
                "description": item["description"],
                "hit": hit,
                "reciprocal_rank": rr,
                "criterion": criterion,
                "retrieved_categories": [d.metadata.get("category", "") for d in docs],
                "retrieved_datasets": [d.metadata.get("dataset", "") for d in docs],
                "relevant_datasets": item["relevant_datasets"],
                "relevant_category": relevant_category,
            })

        n = len(per_query)
        hit_rate = sum(r["hit"] for r in per_query) / n
        mrr = sum(r["reciprocal_rank"] for r in per_query) / n

        return {
            "hit_rate": round(hit_rate, 4),
            "mrr": round(mrr, 4),
            "n_queries": n,
            "per_query": per_query,
        }

    def compare(
        self,
        baseline: RAGEvaluator,
        test_queries: List[Dict],
        k: int = 5,
        label_self: str = "Extended",
        label_baseline: str = "Baseline",
    ) -> Dict:
        """두 retriever 성능 비교.

        Args:
            baseline: 비교 기준이 되는 RAGEvaluator.
            test_queries: 공통 테스트 쿼리셋.
            k: 검색 문서 수.
            label_self: 이 evaluator의 이름 (결과 dict 키).
            label_baseline: baseline의 이름 (결과 dict 키).

        Returns:
            두 결과와 delta(차이)를 포함한 dict.
        """
        result_self = self.evaluate(test_queries, k=k)
        result_base = baseline.evaluate(test_queries, k=k)
        return {
            label_self: result_self,
            label_baseline: result_base,
            "delta": {
                "hit_rate": round(result_self["hit_rate"] - result_base["hit_rate"], 4),
                "mrr": round(result_self["mrr"] - result_base["mrr"], 4),
            },
        }
