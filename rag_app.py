import streamlit as st
import boto3
import os
from dotenv import load_dotenv
from strands import Agent, tool

# =========================
# ENV 로드
# =========================
load_dotenv()

BEDROCK_REGION = os.getenv("AWS_REGION")
KB_ID = os.getenv("KNOWLEDGE_BASE_ID")

# bedrock 실행 API
bedrock_rt = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
# bedrock 지식기반 API
bedrock_kb = boto3.client("bedrock-agent-runtime", region_name=BEDROCK_REGION)


# 검색함수
def retrieve_from_kb(question: str, top_k: int = 8, score_threshold: float = 0.50):
    resp = bedrock_kb.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": top_k}
        },
        retrievalQuery={"text": question},
    )

    chunks = []
    for r in resp.get("retrievalResults", []):
        score = r.get("score", 0)
        if score < score_threshold:
            continue

        text = r["content"]["text"]
        uri = r.get("location", {}).get("s3Location", {}).get("uri", "")
        file_name = uri.split("/")[-1] if uri else "Unknown"

        chunks.append({
            "text": text,
            "source": file_name,
            "score": score,
        })

    return chunks

# 동일 참조 문서 병합
def merge_chunks_by_file(chunks):
    grouped = {}
    for c in chunks:
        file = c["source"]
        if file not in grouped:
            grouped[file] = {"texts": [], "scores": [], "source": file}

        grouped[file]["texts"].append(c["text"])
        grouped[file]["scores"].append(c["score"])

    merged = []
    for file, data in grouped.items():
        avg_score = sum(data["scores"]) / len(data["scores"])
        weighted = avg_score * (1 + (len(data["scores"]) - 1) * 0.25)

        merged.append({
            "source": file,
            "score": weighted,
            "text": "\n".join(data["texts"]),
        })

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged



# agent 도구
@tool
def kb_retrieve(question: str) -> dict:
    chunks = retrieve_from_kb(question)
    merged = merge_chunks_by_file(chunks)
    return {"documents": merged}



# Agent 프롬프트
AGENT_PROMPT = """
너는 전문 시험 문제 생성 Agent이다.

다음은 검색된 참고 문서들이다:
<documents>
{documents}
</documents>

이 문서들을 기반으로 아래 단계를 수행한다:

[Step 1] 문서에서 핵심 개념 5~10개 추출
[Step 2] 각 개념별 문제 후보 생성
[Step 3] 객관식 4지선다로 변환
[Step 4] 정답은 하나만 유지
[Step 5] 품질 점검

[출력 형식 예시]
1. 문제
A. 보기
B. 보기
C. 보기
D. 보기
정답: A
참조문서: file1.pdf, file2.txt
"""

exam_agent = Agent(
    model="us.amazon.nova-lite-v1:0",
    system_prompt=AGENT_PROMPT,
    tools=[kb_retrieve]
)



# RAG 답변 함수
def rag_answer(question: str):
    chunks = retrieve_from_kb(question)
    merged = merge_chunks_by_file(chunks)

    response = exam_agent(
        question,
        documents=merged
    )

    final_text = str(response)
    return final_text, merged


# UI
st.set_page_config(page_title="RAG", layout="wide")
st.title("WinDos Application")

# 사이드 바 넓이
SIDEBAR_WIDTH = 350
st.markdown(
    f"""
    <style>
        [data-testid="stSidebar"] {{
            width: {SIDEBAR_WIDTH}px !important;
        }}
        [data-testid="stSidebarNav"] {{
            width: {SIDEBAR_WIDTH}px !important;
        }}
    </style>
    """, unsafe_allow_html=True
)

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_answer" not in st.session_state:
    st.session_state.pending_answer = None
if "footnotes" not in st.session_state:
    st.session_state.footnotes = []
if "selected_footnote" not in st.session_state:
    st.session_state.selected_footnote = None


query = st.chat_input("질문을 입력하세요...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.pending_answer = query


# 메세지
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("\n", "  \n"), unsafe_allow_html=True)


# 답변 생성
if st.session_state.pending_answer:
    q = st.session_state.pending_answer

    with st.chat_message("assistant"):
        with st.spinner("시험 문제 생성 중..."):
            answer, contexts = rag_answer(q)

    formatted = answer.replace("\n", "  \n")
    st.session_state.messages.append({"role": "assistant", "content": formatted})

    st.session_state.footnotes = [
        {"index": i, "text": ctx["text"], "source": ctx["source"], "score": ctx["score"]}
        for i, ctx in enumerate(contexts, 1)
    ]

    st.session_state.pending_answer = None
    st.rerun()


# 사이드바
with st.sidebar:
    st.header("참고 문서")

    footnotes = st.session_state.footnotes

    if footnotes:
        cols = st.columns(min(5, len(footnotes)))
        for i, f in enumerate(footnotes):
            if cols[i % 5].button(f"[{f['index']}]", key=f"foot_{i}"):
                st.session_state.selected_footnote = f

        st.markdown("---")

    if st.session_state.selected_footnote:
        f = st.session_state.selected_footnote
        st.markdown(f"### {f['source']} (score: {f['score']:.2f})")
        st.markdown(f["text"].replace("\n", "  \n"), unsafe_allow_html=True)
    else:
        st.write("선택된 참고 문서가 없습니다.")
