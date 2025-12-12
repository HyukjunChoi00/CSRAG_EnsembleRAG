import hashlib
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field


class EnsembleRetriever(BaseRetriever):
    """
    BM25와 FAISS를 결합한 Ensemble Retriever (LangChain Runnable 호환)
    """

    # Pydantic Field로 멤버 변수 선언
    bm25_retriever: Any = Field(description="BM25 Retriever 객체")
    faiss_retriever: Any = Field(description="FAISS Retriever 객체")
    bm25_weight: float = Field(default=0.4, description="BM25 가중치")
    faiss_weight: float = Field(default=0.6, description="FAISS 가중치")
    k: int = Field(default=60, description="RRF 파라미터 상수")
    top_k: int = Field(default=6, description="최종 반환 문서 개수")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        # [수정 1] None을 허용하도록 타입 힌트 수정 ( | None 추가)
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """
        LangChain 내부에서 invoke 호출 시 실제로 실행되는 메서드
        """

        # 1. 후보군 개수 설정
        candidate_k = self.top_k * 3

        # 2. BM25 검색
        self.bm25_retriever.k = candidate_k
        bm25_docs = self.bm25_retriever.invoke(query)

        # 3. FAISS 검색
        if self.faiss_retriever.search_kwargs:
            self.faiss_retriever.search_kwargs.update({"k": candidate_k})
        else:
            self.faiss_retriever.search_kwargs = {"k": candidate_k}

        faiss_docs = self.faiss_retriever.invoke(query)

        # 4. RRF (Reciprocal Rank Fusion) 로직
        # [수정 2] 빈 딕셔너리의 타입을 명시적으로 선언 (dict[str, dict[str, Any]])
        doc_scores: dict[str, dict[str, Any]] = {}

        # BM25 점수 합산
        for rank, doc in enumerate(bm25_docs):
            doc_id = self._get_doc_id(doc)
            rrf_score = self.bm25_weight / (self.k + rank + 1)

            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0.0}
            doc_scores[doc_id]["score"] += rrf_score

        # FAISS 점수 합산
        for rank, doc in enumerate(faiss_docs):
            doc_id = self._get_doc_id(doc)
            rrf_score = self.faiss_weight / (self.k + rank + 1)

            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0.0}
            doc_scores[doc_id]["score"] += rrf_score

        # 점수 기준 내림차순 정렬
        # doc_scores의 타입이 명확해졌으므로 sorted 에러도 사라짐
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: float(x["score"]),  # float으로 확실하게 변환
            reverse=True,
        )

        # item['doc']가 Document 타입임을 보장하기 위해 cast 사용 (선택사항이나 안전함)
        return [cast(Document, item["doc"]) for item in sorted_docs[: self.top_k]]

    def _get_doc_id(self, doc: Document) -> str:
        """
        문서 고유 ID 생성 (hashlib 사용으로 재실행 시 안정성 확보)
        """
        content_str = doc.page_content

        metadata = cast(dict[str, Any], getattr(doc, "metadata", {}))

        source = str(metadata.get("source", ""))
        row = str(metadata.get("row", ""))

        # 내용을 합쳐서 MD5 해시 생성
        unique_str = f"{content_str}_{source}_{row}"
        return hashlib.md5(unique_str.encode("utf-8")).hexdigest()
