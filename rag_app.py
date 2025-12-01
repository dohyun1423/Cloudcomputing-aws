import streamlit as st
from strands_tools import http_request
import boto3
import os
from dotenv import load_dotenv


load_dotenv()
# bedrock 설정
BEDROCK_REGION = os.getenv("AWS_REGION")
KB_ID = os.getenv("KNOWLEDGE_BASE_ID")

# LLM API 호출
bedrock_rt = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
# 벡터 검색 API
bedrock_kb = boto3.client("bedrock-agent-runtime", region_name=BEDROCK_REGION)

# 벡터 db에서 관련 문서 검색
def retrieve_from_kb(question: str, top_k: int = 8, score_threshold: float = 0.50):
    resp = bedrock_kb.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": top_k}
        },
        retrievalQuery={"text": question},
    )
    # 관련 문서 메타정보 저장
    chunks = []
    for r in resp.get("retrievalResults", []):
        score = r.get("score", 0)
        if score < score_threshold:
            continue

        text = r["content"]["text"]

        uri = (
            r.get("location", {})
             .get("s3Location", {})
             .get("uri", "")
        )
        file_name = uri.split("/")[-1] if uri else "Unknown document"

        chunks.append({
            "text": text,
            "source": file_name,
            "score": score,
        })

    return chunks

# 관련 문서 병합 - 동일 문서 여러번 저장하는걸 대비
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
        weighted_score = avg_score * (1 + (len(data["scores"]) - 1) * 0.25)

        merged.append({
            "source": file,
            "score": weighted_score,
            "text": "\n".join(data["texts"]),
        })

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged

# LLM 답변 생성
def rag_answer(question: str):
    chunks = retrieve_from_kb(question)
    contexts = merge_chunks_by_file(chunks)

    if contexts:
        context_text = "\n\n".join(
            f"[{i+1}] FILE: {ctx['source']}\n{ctx['text']}"
            for i, ctx in enumerate(contexts)
        )
    else:
        context_text = "No relevant documents found."

    messages = [
        {
            "role": "user",
            "content": [
                {"text": f"Context:\n{context_text}\n\nQuestion: {question}"}
            ]
        }
    ]

    resp = bedrock_rt.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=messages,
        inferenceConfig={"maxTokens": 4096, "temperature": 0},
    )

    answer = resp["output"]["message"]["content"][0]["text"]
    return answer, contexts


# UI 설정
st.set_page_config(page_title="RAG", layout="wide")
st.title("WinDos application")

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            width: 350px !important;
        }
        [data-testid="stSidebarNav"] {
            width: 350px !important;
        }
    </style>
""", unsafe_allow_html=True)


# 세션
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_answer" not in st.session_state:
    st.session_state.pending_answer = None

if "selected_footnote" not in st.session_state:
    st.session_state.selected_footnote = None

if "footnotes" not in st.session_state:
    st.session_state.footnotes = []


# 질문 입력 처리
query = st.chat_input("질문을 입력하세요...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.pending_answer = query


# 메세지 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# 답변 생성
if st.session_state.pending_answer:
    pending_q = st.session_state.pending_answer

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            answer, contexts = rag_answer(pending_q)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.session_state.footnotes = [
        {
            "index": i,
            "text": ctx["text"],
            "source": ctx["source"],
            "score": ctx["score"],
        }
        for i, ctx in enumerate(contexts, 1)
    ]

    st.session_state.pending_answer = None

    st.rerun()


# 사이드바 참고 문서 표시
with st.sidebar:
    st.header("참고 문서")

    footnotes = st.session_state.footnotes

    if footnotes:
        cols = st.columns(min(5, len(footnotes)))
        for i, foot in enumerate(footnotes):
            col = cols[i % 5]
            if col.button(f"[{foot['index']}]", key=f"side_{foot['index']}"):
                st.session_state.selected_footnote = foot

        st.markdown("---")

    if st.session_state.selected_footnote:
        f = st.session_state.selected_footnote
        st.markdown(f"### {f['source']} (score: {f['score']:.2f})")
        st.write(f["text"])
    else:
        st.write("선택된 참고 문서가 없습니다.")
