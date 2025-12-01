import streamlit as st
import boto3
import os
import re
from dotenv import load_dotenv
from strands import Agent

load_dotenv()

BEDROCK_REGION = os.getenv("AWS_REGION")
KB_ID = os.getenv("KNOWLEDGE_BASE_ID")

# 대화 기록 유지 개수 - 이 숫자를 넘으면 오래된 것부터 삭제
MAX_HISTORY = 20

# 리소스 캐싱 - streamlit 재실행에도 유지
@st.cache_resource
def get_bedrock_client():
    return boto3.client("bedrock-agent-runtime", region_name=BEDROCK_REGION)

@st.cache_resource
def get_agents():
    # 검색 쿼리 변환
    query_prompt = """
    당신은 '검색어 최적화 도구'입니다. 
    <history>와 <question>을 참고하여, 사용자가 실제로 의도하는 바를 포함한 '검색 가능한 문장' 하나만 출력하세요.
    설명이나 인사말은 생략하세요.

    예시:
    히스토리: "S3 스토리지 클래스 알려줘"
    질문: "관련된 문제 내줘"
    출력: AWS S3 스토리지 클래스 객관식 시험 문제
    """

    # 초안 생성
    drafter_prompt = """
    당신은 '시험 문제 출제 위원'입니다. 
    <documents>와 <history>를 분석하여 요청에 맞는 객관식 문제 '초안'을 작성하세요.

    [지침]
    1. **생성 제한:** 한 번의 답변에 **최대 5문제**까지만 생성하세요. 사용자가 그 이상을 요청하면 5개만 생성하고 "나머지는 나눠서 요청해주세요"라고 덧붙이세요. (토큰 초과 방지)
    2. 사용자가 개수를 지정하지 않았다면 기본 3문제를 생성하세요.
    3. <history>에 이미 나온 문제와 중복되지 않도록 구성하세요.
    4. 각 문제의 근거가 되는 문서 ID를 반드시 기록하세요.
    5. **주의:** 제공된 문서 ID(예: [ID: 1]) 외의 번호를 절대 사용하지 마세요.

    [형식]
    1. 핵심 개념
    2. 질문
    3. 정답 및 오답 후보
    4. 출처 문서 번호 (예: [1])
    """
    
    # 검수 및 편집
    editor_prompt = """
    당신은 '최종 검수자'입니다. 초안을 검토하고 아래 형식으로 변환하세요.
    초안의 '출처 문서 번호'를 '참고' 란에 반드시 포함하세요.

    [최종 형식]
    Q1. [문제]
    A. [보기] ... D. [보기]
    정답: [알파벳]
    해설: [해설]
    참고: [출처 번호]

    ---
    (반복)
    """

    query_agent = Agent(model="us.amazon.nova-lite-v1:0", system_prompt=query_prompt)
    drafter = Agent(model="us.amazon.nova-lite-v1:0", system_prompt=drafter_prompt)
    editor = Agent(model="us.amazon.nova-lite-v1:0", system_prompt=editor_prompt)
    
    return query_agent, drafter, editor

# 관련 문서 검색
def retrieve_from_kb(question: str, top_k: int = 8, score_threshold: float = 0.60):
    client = get_bedrock_client()
    resp = client.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": top_k}},
        retrievalQuery={"text": question},
    )

    chunks = []
    for r in resp.get("retrievalResults", []):
        score = r.get("score", 0)
        if score < score_threshold: continue
        
        uri = r.get("location", {}).get("s3Location", {}).get("uri", "")
        chunks.append({
            "text": r["content"]["text"],
            "source": uri.split("/")[-1] if uri else "Unknown",
            "score": score,
        })
    return chunks

# 동일한 참고문서 내용 병합
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
    
    return sorted(merged, key=lambda x: x["score"], reverse=True)

# 각주번호 할루시네이션 후처리
def clean_hallucinated_references(text: str, max_doc_count: int) -> str:
    def validator(match):
        try:
            if int(match.group(1)) > max_doc_count: return "" 
            return match.group(0)
        except ValueError: return match.group(0)

    cleaned = re.sub(r'\[(\d+)\]', validator, text)
    return re.sub(r'참고:\s*$', '참고: (없음)', cleaned, flags=re.MULTILINE)

# 대화 포맷
def format_history(messages):
    recent_msgs = messages[-6:] 
    history_text = ""
    for msg in recent_msgs:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"
    return history_text if history_text else "없음"

# 대화 히스토리 제한 관리 - 오래된 순 삭제
def manage_history_limit():
    if len(st.session_state.messages) > MAX_HISTORY:
        diff = len(st.session_state.messages) - MAX_HISTORY
        st.session_state.messages = st.session_state.messages[diff:]

def rag_answer_chain(question: str, messages: list):
    query_agent, drafter, editor = get_agents()
    
    history_text = format_history(messages)

    search_query_response = query_agent(f"<history>\n{history_text}\n</history>\n<question>\n{question}\n</question>")
    search_query = str(search_query_response)
    
    chunks = retrieve_from_kb(search_query)
    merged = merge_chunks_by_file(chunks)
    
    if not merged:
        return f"'{search_query}'에 대한 관련 문서를 찾을 수 없습니다.", []

    doc_count = len(merged)
    context_text = f"총 문서 개수: {doc_count}개\n\n" + "\n\n".join(
        [f"[ID: {i+1}] 파일명: {doc['source']}\n내용:\n{doc['text']}" for i, doc in enumerate(merged)]
    )

    draft_input = f"""
    <history>
    {history_text}
    </history>
    
    <documents>
    {context_text}
    </documents>

    사용자 요청: {question}
    (주의: 문서 ID는 1부터 {doc_count}까지만 유효)
    """
    draft_response = drafter(draft_input)
    
    final_response = editor(f"다음 초안을 검수하세요:\n\n{draft_response}")
    final_text = clean_hallucinated_references(str(final_response), doc_count)

    return final_text, merged

# 세션 초기화
def init_session_state():
    defaults = {"messages": [], "pending_answer": None, "footnotes": [], "selected_footnote": None}
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

# streamlit UI
st.set_page_config(page_title="WinDos Exam", layout="wide")
st.title("클라우드 컴퓨팅 문제 생성기")
init_session_state()

st.markdown("""<style>[data-testid="stSidebar"], [data-testid="stSidebarNav"] { width: 350px !important; }</style>""", unsafe_allow_html=True)

if query := st.chat_input("주제 또는 생성할 문제 수를 입력하세요 (예: S3 문제 3개 줘)"):
    st.session_state.messages.append({"role": "user", "content": query})
    # 히스토리 길이 관리 실행
    manage_history_limit()
    st.session_state.pending_answer = query

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("\n", "  \n"), unsafe_allow_html=True)

if st.session_state.pending_answer:
    with st.chat_message("assistant"):
        with st.spinner("대화 맥락 분석 및 문제 생성 중..."):
            answer, contexts = rag_answer_chain(st.session_state.pending_answer, st.session_state.messages)
    
    st.session_state.messages.append({"role": "assistant", "content": answer.replace("\n", "  \n")})
    # 답변 추가 후에도 히스토리 관리
    manage_history_limit()
    
    st.session_state.footnotes = [
        {"index": i, "text": ctx["text"], "source": ctx["source"], "score": ctx["score"]} for i, ctx in enumerate(contexts, 1)
    ]
    st.session_state.pending_answer = None
    st.rerun()

# 사이드바
with st.sidebar:
    st.header(f"참고 문서 ({len(st.session_state.footnotes)}개)")
    if footnotes := st.session_state.footnotes:
        cols = st.columns(min(5, len(footnotes)))
        for i, f in enumerate(footnotes):
            if cols[i % 5].button(f"[{f['index']}]", key=f"foot_{i}"):
                st.session_state.selected_footnote = f
        st.markdown("---")
        
    if f := st.session_state.selected_footnote:
        st.markdown(f"### [{f['index']}] {f['source']}")
        st.caption(f"Score: {f['score']:.4f}")
        st.info(f['text'])
    else:
        st.write("번호를 클릭하면 문서 내용이 표시됩니다.")