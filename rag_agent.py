"""
Multi-Agent RAG System with LangGraph

시스템 구조:
1. 쿼리 분석 및 다각화 Agent: 질문을 분석하고 검색에 최적화된 쿼리 생성
2. Multi-step Reasoning Agent: 검색 결과를 바탕으로 단계적 추론
3. 최종 답변 Agent: 객관식 답을 숫자로 출력 (A→1, B→2, C→3, D→4)

사용법:
    python evaluate.py --input_path data/dev.csv --output_path data/predictions.csv
"""

import argparse
import os
from operator import add
from typing import Annotated, Any, TypedDict, cast

import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pandas import Series

from ensemble_retriever import EnsembleRetriever

# ═══════════════════════════════════════════════════════════════════════════
# 환경 설정
# ═══════════════════════════════════════════════════════════════════════════
# 회사 api
os.environ["OPENAI_API_KEY"] = (
    "YOUR_API_KEY"
)

llm = ChatOpenAI(model="gpt-4o-mini")

print("LLM 초기화 완료")

# ═══════════════════════════════════════════════════════════════════════════
# 데이터 로드 및 Knowledge Base 구축
# ═══════════════════════════════════════════════════════════════════════════


def setup_knowledge_base(csv_path: str = "train_processed.csv"):
    """CSV 파일로부터 Knowledge Base 구축"""
    print("\nKnowledge Base 구축 시작...")

    # CSV 파일 로드
    loader = CSVLoader(file_path=csv_path, encoding="utf-8")
    documents = loader.load()
    print(f"  ✓ 문서 로드 완료: {len(documents)}개")

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        length_function=len,
    )

    splits = text_splitter.split_documents(documents)
    print(f"문서 분할 완료: {len(splits)}개 청크")
    # splits = splits[:20] # for testing

    print("임베딩 생성 중...")
    # Retriever 구축
    bm25_retriever = BM25Retriever.from_documents(splits)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    faiss_db = FAISS.from_documents(splits, embeddings)
    faiss_retriever = faiss_db.as_retriever(
        search_type="similarity",  # Top-k 방식, threshold 없음
        # k는 EnsembleRetriever 내부에서 설정되므로 여기서는 지정하지 않음 (candidate_k 사용)
    )
    ensemble_retriever = EnsembleRetriever(
        bm25_retriever=bm25_retriever,
        faiss_retriever=faiss_retriever,
        bm25_weight=0.3,  # BM25 30%
        faiss_weight=0.7,  # FAISS 70%
        k=60,
    )

    return ensemble_retriever, splits


# 전역 변수로 retriever 저장 (나중에 초기화)
ensemble_retriever = None
splits = None

# 전역 verbose 플래그 (Agent 출력 제어용)
_verbose_mode = True

# ═══════════════════════════════════════════════════════════════════════════
# State 정의
# ═══════════════════════════════════════════════════════════════════════════


class AgentState(TypedDict):
    """Multi-Agent RAG 시스템의 상태"""

    # 입력
    original_query: str  # 원본 질문

    # Agent 1: 쿼리 분석 및 다각화
    analyzed_query: str  # 분석된 쿼리
    refined_queries: list[str]  # 다각화된 쿼리들

    # 검색 결과
    retrieved_documents: list[str]  # 검색된 문서들

    # Agent 2: Multi-step Reasoning
    reasoning_steps: list[str]  # 추론 단계들
    reasoning_result: str  # 추론 결과

    # Agent 3: 최종 답변
    final_answer: str  # 최종 답변 (1, 2, 3, 4)

    # 메타데이터
    messages: Annotated[list[str], add]  # 로그 메시지


# ═══════════════════════════════════════════════════════════════════════════
# Agent 1: 쿼리 분석 및 다각화
# ═══════════════════════════════════════════════════════════════════════════


def query_analysis_agent(state: AgentState) -> AgentState:
    """
    쿼리를 분석하고 검색에 최적화된 다각화된 쿼리를 생성
    원본 1개 + 다각화 2개 = 총 3개 쿼리 사용
    """
    global _verbose_mode

    if _verbose_mode:
        print("\n [Agent 1] 쿼리 분석 및 다각화 시작...")

    original_query = state["original_query"]

    # 프롬프트 정의 (다각화 쿼리 2개만 생성)
    prompt = ChatPromptTemplate.from_template(
        """당신은 검색 쿼리 최적화 전문가입니다.

주어진 질문을 분석하고, 더 효과적인 검색을 위해 다각화된 쿼리 2개를 생성하세요.
원본 질문은 그대로 유지하고, 다른 표현이나 확장된 질문 2개를 추가로 만드세요.

### 지시사항:
1. 핵심 키워드를 추출하세요
2. 유사한 의미의 표현으로 변환하세요
3. 더 구체적이거나 일반적인 질문으로 확장하세요

### 원본 질문:
{query}

### 출력 형식 (정확히 따르세요):
분석: [질문의 핵심 분석]
쿼리1: [첫 번째 다각화 쿼리]
쿼리2: [두 번째 다각화 쿼리]
"""
    )

    chain = prompt | llm | StrOutputParser()  # type: ignore
    result = chain.invoke({"query": original_query})  # type: ignore

    # 결과 파싱
    lines = result.strip().split("\n")
    analyzed_query = ""
    refined_queries: list[str] = []

    for line in lines:
        if line.startswith("분석:"):
            analyzed_query = line.replace("분석:", "").strip()
        elif line.startswith("쿼리1:"):
            refined_queries.append(line.replace("쿼리1:", "").strip())
        elif line.startswith("쿼리2:"):
            refined_queries.append(line.replace("쿼리2:", "").strip())

    if _verbose_mode:
        print(f"  ✓ 분석: {analyzed_query[:50]}...")
        print("  ✓ 원본 쿼리: 1개 (유지)")
        print(f"  ✓ 다각화 쿼리: {len(refined_queries)}개 생성")
        print(f"  ✓ 총 사용 쿼리: {1 + len(refined_queries)}개")

    return {
        **state,
        "analyzed_query": analyzed_query,
        "refined_queries": refined_queries,
        "messages": ["[Agent 1] 쿼리 분석 완료"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 검색 노드
# ═══════════════════════════════════════════════════════════════════════════


def retrieval_node(state: AgentState) -> AgentState:
    """
    원본 쿼리 + 다각화된 쿼리로 문서 검색 수행
    원본 1개 + 다각화 2개 = 총 3개 쿼리로 검색
    """
    global _verbose_mode

    if _verbose_mode:
        print("[검색] 문서 검색 시작...")

    original_query = state["original_query"]
    refined_queries = state.get("refined_queries", [])

    # 원본 쿼리 1개 + 다각화 쿼리 2개
    all_queries = [original_query] + refined_queries
    all_docs: list[str] = []
    seen_docs: set[str] = set()  # 중복 제거용

    if _verbose_mode:
        print(
            f"  → 총 {len(all_queries)}개 쿼리로 검색 (원본 1개 + 다각화 {len(refined_queries)}개)"
        )

    assert ensemble_retriever is not None, "Ensemble Retriever가 초기화되지 않았습니다."
    for idx, query in enumerate(all_queries, 1):
        if _verbose_mode:
            query_type = "원본" if idx == 1 else f"다각화{idx - 1}"
            print(f"    [{query_type}] {query[:40]}...")

        docs = ensemble_retriever.invoke(query)
        for doc in docs:
            metadata = cast(dict[str, Any], getattr(doc, "metadata", {}))
            # row 값이 없으면 None일 수 있으므로 안전하게 처리
            row_val = metadata.get("row")
            doc_id = str(row_val) if row_val is not None else doc.page_content[:50]
            if doc_id not in seen_docs:
                all_docs.append(doc.page_content)
                seen_docs.add(doc_id)

    if _verbose_mode:
        print(f"  ✓ 검색된 문서: {len(all_docs)}개 (중복 제거 후)")

    return {
        **state,
        "retrieved_documents": all_docs[:10],  # 상위 10개만 사용
        "messages": ["[검색] 문서 검색 완료"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Agent 2: Multi-step Reasoning
# ═══════════════════════════════════════════════════════════════════════════


def reasoning_and_answer_agent(state: AgentState) -> AgentState:
    """
    검색된 문서를 바탕으로 단계적 추론 수행 후 최종 답변을 숫자로 출력
    (기존 Agent 2 + Agent 3 통합)
    """
    global _verbose_mode

    if _verbose_mode:
        print("[Agent 2] Multi-step Reasoning & Answer 시작...")

    original_query = state["original_query"]
    documents = state["retrieved_documents"]

    # 문서들을 컨텍스트로 결합
    context = "\n\n---\n\n".join(documents[:5])  # 상위 5개 문서만 사용

    # 프롬프트 정의 (추론 + 답변 통합)
    prompt = ChatPromptTemplate.from_template(
        """당신은 객관식 문제를 푸는 전문가입니다.

주어진 문서들을 분석하여 질문에 대한 답을 단계적으로 추론한 후, 최종 답을 숫자로 출력하세요.

### 질문:
{query}

### 문서들:
{context}

### 지시사항:
1. 문서에서 관련 정보를 추출하세요
2. 단계별로 논리적인 추론을 진행하세요
3. 최종 결론을 도출하세요
4. **마지막에 답을 1, 2, 3, 4 중 하나의 숫자로만 출력하세요**

### 중요:
- 문서에 'answer: X' 형태로 답이 명시되어 있다면 그 값을 사용하세요
- 객관식 보기가 A, B, C, D로 제시되면: A→1, B→2, C→3, D→4로 변환하세요
- 최종 출력은 반드시 1, 2, 3, 4 중 하나의 **숫자만** 출력하세요

### 출력 형식:
단계1: [첫 번째 추론 단계]
단계2: [두 번째 추론 단계]
단계3: [세 번째 추론 단계]
결론: [최종 결론]
최종답변: [1, 2, 3, 또는 4 중 하나의 숫자만]
"""
    )

    chain = prompt | llm | StrOutputParser()  # type: ignore
    result = chain.invoke({"query": original_query, "context": context})  # type: ignore

    # 결과 파싱
    lines = result.strip().split("\n")
    reasoning_steps: list[str] = []
    reasoning_result = ""
    final_answer = ""

    for line in lines:
        if line.startswith("단계"):
            reasoning_steps.append(line)
        elif line.startswith("결론:"):
            reasoning_result = line.replace("결론:", "").strip()
        elif line.startswith("최종답변:"):
            final_answer = line.replace("최종답변:", "").strip()

    # 숫자 추출 (최종답변 라인에서 찾기)
    if final_answer:
        final_answer = "".join(filter(str.isdigit, final_answer))

    # 최종답변 라인이 없으면 전체 결과에서 마지막 숫자 찾기
    if not final_answer:
        all_digits = "".join(filter(str.isdigit, result))
        if all_digits:
            final_answer = all_digits[-1]  # 마지막 숫자

    # 그래도 없으면 0
    final_answer = "0" if not final_answer else final_answer[0]

    if _verbose_mode:
        print(f"  ✓ 추론 단계: {len(reasoning_steps)}개")
        print(f"  ✓ 결론: {reasoning_result[:50]}...")
        print(f"  ✓ 최종 답변: {final_answer}")

    return {
        **state,
        "reasoning_steps": reasoning_steps,
        "reasoning_result": reasoning_result,
        "final_answer": final_answer,
        "messages": ["[Agent 2] 추론 및 답변 완료"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph 구축
# ═══════════════════════════════════════════════════════════════════════════


def build_agent_graph():
    """LangGraph 워크플로우 구축"""
    # StateGraph 생성
    workflow = StateGraph(AgentState)

    # 노드 추가 (Agent 2와 3을 통합)
    workflow.add_node("analyze_query", query_analysis_agent)  # type: ignore
    workflow.add_node("retrieve_docs", retrieval_node)  # type: ignore
    workflow.add_node("reason_and_answer", reasoning_and_answer_agent)  # type: ignore

    # 엣지 정의 (순서대로 실행)
    workflow.set_entry_point("analyze_query")  # type: ignore
    workflow.add_edge("analyze_query", "retrieve_docs")  # type: ignore
    workflow.add_edge("retrieve_docs", "reason_and_answer")  # type: ignore
    workflow.add_edge("reason_and_answer", END)  # type: ignore

    # 그래프 컴파일
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)  # type: ignore

    print("  1. 쿼리 분석 및 다각화 (원본 1개 + 다각화 2개)")
    print("  2. 문서 검색 (3개 쿼리로 검색)")
    print("  3. Multi-step Reasoning & Answer (추론 + 답변 통합)\n")

    return app


# ═══════════════════════════════════════════════════════════════════════════
# 실행 함수
# ═══════════════════════════════════════════════════════════════════════════


def run_agent(app: Any, query: str) -> dict[str, Any]:
    """
    단일 질문에 대해 Agent 실행
    """
    print(f"\n{'=' * 70}")
    print("Multi-Agent RAG 실행")
    print(f"{'=' * 70}")
    print(f"질문: {query}")

    # 초기 상태
    initial_state: AgentState = {
        "original_query": query,
        "analyzed_query": "",
        "refined_queries": [],
        "retrieved_documents": [],
        "reasoning_steps": [],
        "reasoning_result": "",
        "final_answer": "",
        "messages": [],
    }

    # Agent 실행
    config = {"configurable": {"thread_id": "1"}}
    result = app.invoke(initial_state, config)

    # 결과 출력
    print(f"\n{'=' * 70}")
    print("실행 결과")
    print(f"{'=' * 70}")
    print(f"분석된 쿼리: {result['analyzed_query']}")
    print("다각화 쿼리:")
    for i, q in enumerate(result["refined_queries"], 1):
        print(f"  {i}. {q}")
    print("추론 단계:")
    for step in result["reasoning_steps"]:
        print(f"  • {step}")
    print(f"추론 결과: {result['reasoning_result']}")
    print(f"최종 답변: {result['final_answer']}")
    print(f"\n{'=' * 70}")

    return result


def format_test_question(row: Series) -> str:  # type: ignore[type-arg]
    """
    테스트용 질문 포맷팅 (답변 없이 질문과 보기만)

    Args:
        row: DataFrame의 한 행 (question, A, B, C, D 컬럼 포함)

    Returns:
        포맷팅된 질문 문자열
    """
    question: str = str(row["question"])  # type: ignore[arg-type]
    options: list[str] = []

    for label in ["A", "B", "C", "D"]:
        if label in row:
            val = row[label]  # type: ignore[has-type]
            if pd.notna(val):  # type: ignore[call-overload, arg-type]
                options.append(f"{label}: {val}")

    formatted = f"질문: {question}\n\n객관식 보기:\n" + "\n".join(options)
    return formatted


def evaluate_agent_on_dev(
    app: Any,
    dev_csv_path: str = "dev.csv",
    output_path: str = "predictions.csv",
    verbose_limit: int = 10,
):
    """
    dev.csv로 Agent 평가 (답변을 보지 않고 질문과 보기만 제공)

    Args:
        app: LangGraph 앱
        dev_csv_path: 평가용 CSV 파일 경로
        output_path: 예측 결과를 저장할 CSV 파일 경로
        verbose_limit: 상세 출력할 질문 개수 (기본값: 10)

    Returns:
        결과 DataFrame
    """
    # dev.csv 로드
    df_dev = pd.read_csv(dev_csv_path, encoding="utf-8")  # type: ignore[call-overload]
    total_queries = len(df_dev)

    results: list[dict[str, Any]] = []

    print(f"\n{'=' * 70}")
    print(f"Dev Set 평가 시작 (전체 {total_queries}개 질문)")
    print(f"{'=' * 70}")
    print("Knowledge Base: train_processed.csv")
    print(f"Test Set: {dev_csv_path}")
    print(f"상세 출력: 첫 {verbose_limit}개 질문만")
    print(f"{'=' * 70}\n")

    for i in range(total_queries):
        row: Series = df_dev.iloc[i]  # type: ignore[type-arg]

        # verbose 모드 설정 (첫 verbose_limit개만)
        global _verbose_mode
        _verbose_mode = i < verbose_limit

        # 테스트용 질문 포맷팅 (답변 제외)
        test_query = format_test_question(row)  # type: ignore[arg-type]
        original_question: str = str(row["question"])  # type: ignore[arg-type]

        # 상세 출력 (첫 verbose_limit개만)
        if i < verbose_limit:
            print(f"\n[{i + 1}/{total_queries}] {original_question[:50]}...")
        elif i == verbose_limit:
            print(
                f"\n... 이후 {total_queries - verbose_limit}개 질문은 백그라운드 처리 중 (Agent 출력 숨김) ...\n"
            )

        correct_answer = "N/A"
        try:
            # Ground Truth (정답)
            correct_answer = str(row["answer"]).strip()  # type: ignore[arg-type]

            # Agent 실행
            initial_state: AgentState = {
                "original_query": test_query,  # 질문 + 보기만 제공
                "analyzed_query": "",
                "refined_queries": [],
                "retrieved_documents": [],
                "reasoning_steps": [],
                "reasoning_result": "",
                "final_answer": "",
                "messages": [],
            }

            config = {"configurable": {"thread_id": str(i)}}
            result = app.invoke(initial_state, config)

            agent_answer = result["final_answer"]
            is_correct = agent_answer == correct_answer

            # 상세 출력 (첫 verbose_limit개만)
            if i < verbose_limit:
                status = "정답" if is_correct else "오답"
                print(f"  정답: {correct_answer} | Agent: {agent_answer} | {status}")

            results.append(
                {
                    "query": original_question,
                    "test_input": test_query[:100] + "...",  # 로그용
                    "correct_answer": correct_answer,
                    "agent_answer": agent_answer,
                    "is_correct": is_correct,
                }
            )

        except Exception as e:
            if i < verbose_limit:
                print(f"   오류: {e}")

            results.append(
                {
                    "query": original_question,
                    "test_input": test_query[:100] + "...",
                    "correct_answer": correct_answer
                    if "correct_answer" in locals()
                    else "N/A",
                    "agent_answer": f"Error: {str(e)[:30]}",
                    "is_correct": False,
                }
            )

        # 진행률 표시 (매 50개마다)
        if (i + 1) % 50 == 0 and i >= verbose_limit:
            correct_so_far: int = sum(1 for r in results if r["is_correct"])
            accuracy_so_far: float = correct_so_far / len(results)
            print(
                f"진행률: {i + 1}/{total_queries} ({(i + 1) / total_queries * 100:.1f}%) | 현재 Accuracy: {accuracy_so_far:.4f}"
            )

    # 결과 분석
    df_results = pd.DataFrame(results)
    total = len(df_results)
    correct: int = int(df_results["is_correct"].sum())  # type: ignore[call-overload]
    accuracy: float = correct / total if total > 0 else 0.0

    print(f"\n{'=' * 70}")
    print("평가 요약")
    print(f"{'=' * 70}")
    print(f"평가된 질문 수: {total}개")
    print(f"정답: {correct}개")
    print(f"오답: {total - correct}개")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"{'=' * 70}\n")

    # 오답 분석 (상위 10개만)
    print("\n오답 목록 (상위 10개):")
    wrong_answers = df_results[~df_results["is_correct"]]  # type: ignore[arg-type]
    if len(wrong_answers) == 0:  # type: ignore[arg-type]
        print("  없음 (모두 정답!)")
    else:
        for _idx, row_item in wrong_answers.head(10).iterrows():  # type: ignore[call-overload, union-attr]
            row_series = cast(Series, row_item)  # type: ignore[type-arg]
            print(f"\n질문: {str(row_series['query'])[:60]}...")  # type: ignore[arg-type]
            print(
                f"  정답: {row_series['correct_answer']} | Agent: {row_series['agent_answer']}"
            )

        if len(wrong_answers) > 10:  # type: ignore[arg-type]
            print(f"\n... 외 {len(wrong_answers) - 10}개 오답 더 있음")  # type: ignore[arg-type]

    # ═══════════════════════════════════════════════════════════════════════════
    # predictions.csv 저장
    # ═══════════════════════════════════════════════════════════════════════════

    # 원본 dev.csv에 predicted_answer와 is_correct 컬럼 추가
    df_predictions = df_dev.copy()
    df_predictions["predicted_answer"] = df_results["agent_answer"].values  # type: ignore[attr-defined]
    df_predictions["is_correct"] = (
        df_results["is_correct"]
        .apply(lambda x: "correct" if x else "incorrect")  # type: ignore[call-overload, misc]
        .values
    )  # type: ignore[attr-defined]

    # predictions.csv 저장
    df_predictions.to_csv(output_path, index=False, encoding="utf-8")  # type: ignore[call-overload]

    print(f"\n{'=' * 70}")
    print(f"예측 결과 저장 완료: {output_path}")
    print(f"{'=' * 70}")
    print("저장된 컬럼:")
    print(f"  - 원본 컬럼: {list(df_dev.columns)}")
    print("  - 추가 컬럼: ['predicted_answer', 'is_correct']")
    print(f"{'=' * 70}\n")

    return df_results


# ═══════════════════════════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════════════════════════


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent RAG System Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python evaluate.py --input_path data/dev.csv --output_path data/predictions.csv
    python evaluate.py --input_path dev.csv --output_path predictions.csv --verbose 5
        """,
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="평가할 입력 CSV 파일 경로 (예: data/dev.csv)",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="예측 결과를 저장할 CSV 파일 경로 (예: data/predictions.csv)",
    )

    parser.add_argument(
        "--knowledge_base",
        type=str,
        default="train_processed.csv",
        help="Knowledge Base로 사용할 CSV 파일 경로 (기본값: train_processed.csv)",
    )

    parser.add_argument(
        "--verbose", type=int, default=10, help="상세 출력할 질문 개수 (기본값: 10)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # 커맨드라인 인자 파싱
    args = parse_args()

    print("\n" + "=" * 70)
    print("Multi-Agent RAG System Evaluation")
    print("=" * 70)
    print(f"입력 파일: {args.input_path}")
    print(f"출력 파일: {args.output_path}")
    print(f"Knowledge Base: {args.knowledge_base}")
    print(f"Verbose 출력: 첫 {args.verbose}개")
    print("=" * 70 + "\n")

    # 1. Knowledge Base 구축
    print("Step 1: Knowledge Base 구축")
    print("=" * 70)
    ensemble_retriever, splits = setup_knowledge_base(args.knowledge_base)

    # 2. Agent Graph 구축
    print("\nStep 2: Agent Graph 구축")
    print("=" * 70)
    app = build_agent_graph()

    # 3. Dev Set 평가 및 predictions.csv 저장
    print("\nStep 3: 평가 및 예측 결과 저장")
    print("=" * 70)
    dev_results = evaluate_agent_on_dev(
        app,
        dev_csv_path=args.input_path,
        output_path=args.output_path,
        verbose_limit=args.verbose,
    )

    print("\n모든 작업 완료!")
