import streamlit as st
import boto3
import os
import re
from dotenv import load_dotenv
from strands import Agent, tool

load_dotenv()

BEDROCK_REGION = os.getenv("AWS_REGION")
KB_ID = os.getenv("KNOWLEDGE_BASE_ID")

# 대화 기록 유지 개수
MAX_HISTORY = 20

# 스레드 간 데이터 공유 컨텍스트
class SharedContext:
    def __init__(self):
        self.docs = []

@st.cache_resource
def get_shared_context():
    return SharedContext()

# 리소스 캐싱
@st.cache_resource
def get_bedrock_client():
    return boto3.client("bedrock-agent-runtime", region_name=BEDROCK_REGION)

# 검색 도구
@tool
def search_knowledge_base(query: str) -> str:
    """
    AWS 지식 기반(Knowledge Base)에서 관련 문서를 검색합니다.
    시험 문제 출제에 필요한 정보를 찾을 때 사용하세요.
    
    Args:
        query: 검색할 키워드나 문장 (예: "AWS S3 스토리지 클래스")
    """
    try:
        client = boto3.client("bedrock-agent-runtime", region_name=BEDROCK_REGION)
        
        resp = client.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 8}},
            retrievalQuery={"text": query},
        )
        # 각주 처리 메타정보
        chunks = []
        for r in resp.get("retrievalResults", []):
            score = r.get("score", 0)
            # 임계값 설정
            if score < 0.60:
                continue

            uri = r.get("location", {}).get("s3Location", {}).get("uri", "")
            chunks.append({
                "text": r["content"]["text"],
                "source": uri.split("/")[-1] if uri else "Unknown",
                "score": score,
            })
        # 동일 문서 내용 병합 + 가중치 계산
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
        
        sorted_docs = sorted(merged, key=lambda x: x["score"], reverse=True)
        
        ctx = get_shared_context()
        ctx.docs = sorted_docs

        if not sorted_docs:
            return "검색 결과가 없습니다 (관련도 낮음)."

        formatted_text = f"총 {len(sorted_docs)}개의 문서가 검색되었습니다.\n\n"
        formatted_text += "\n\n".join(
            [f"[ID: {i+1}] 파일명: {doc['source']}\n내용:\n{doc['text']}" for i, doc in enumerate(sorted_docs)]
        )
        return formatted_text

    except Exception as e:
        return f"검색 중 오류 발생: {str(e)}"

# 에이전트 설정
@st.cache_resource
def get_agents():
    query_prompt = """
    당신은 '검색어 최적화 도구'입니다. 
    <history>와 <question>을 참고하여, '검색 가능한 문장' 하나만 출력하세요.
    """

    drafter_prompt = """
    당신은 '시험 문제 출제 위원'입니다. 
    **반드시 `search_knowledge_base` 도구를 먼저 사용하여 정보를 검색하세요.**
    검색된 정보를 바탕으로 객관식 문제 '초안'을 작성하세요.

    [지침]
    1. **도구 사용 필수:** 지식 없이 문제를 내지 말고, 도구 검색 결과를 근거로 사용하세요.
    2. **생성 제한:** 한 번에 **최대 5문제**까지만 생성하세요.
    3. 각 문제의 근거가 되는 문서 ID를 반드시 기록하세요.
    4. **형식 준수:** 출처는 반드시 **[1], [2]** 와 같은 형식으로만 표기하세요. (ID: 1 X, [ID: 1] X)

    [형식]
    1. 핵심 개념
    2. 질문
    3. 정답 및 오답 후보
    4. 출처 문서 번호 (예: [1], [3])
    """
    
    editor_prompt = """
    당신은 '최종 검수자'입니다. 초안을 검토하고 아래 형식으로 변환하세요.
    초안의 '출처 문서 번호'를 [번호]의 형태로 '참고' 란에 반드시 포함하세요.
    
    [주의사항]
    1. **마크다운 헤더(#) 사용 금지:** 글자 크기가 커지므로 절대 사용하지 마세요.
    2. 강조는 **굵게**만 사용하세요.
    3. **출처 형식 통일:** [ID: 1] 같은 형식이 있다면 **[1]**로 변경하세요. 여러 개일 경우 **[1], [2]**로 표기하세요.

    [최종 형식]
    **Q1. [문제]**
    A. [보기] ... D. [보기]
    **정답: [알파벳]**
    **해설:** [해설]
    참고: [출처 번호]

    ---
    (반복)
    """

    query_agent = Agent(model="us.amazon.nova-lite-v1:0", system_prompt=query_prompt)
    
    drafter = Agent(
        model="us.amazon.nova-lite-v1:0", 
        system_prompt=drafter_prompt,
        tools=[search_knowledge_base]
    )
    
    editor = Agent(model="us.amazon.nova-lite-v1:0", system_prompt=editor_prompt)
    
    return query_agent, drafter, editor

# 마크다운 헤더 제거
def remove_markdown_headers(text: str) -> str:
    return re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

# 참조 형식 정규화
def normalize_references(text: str) -> str:
    return re.sub(r'\[ID:\s*(\d+)\]', r'[\1]', text, flags=re.IGNORECASE)

# 각주번호 할루시네이션 후처리
def clean_hallucinated_references(text: str, max_doc_count: int) -> str:
    def validator(match):
        try:
            if int(match.group(1)) > max_doc_count: return "" 
            return match.group(0)
        except ValueError: return match.group(0)

    normalized_text = normalize_references(text)
    cleaned = re.sub(r'\[(\d+)\]', validator, normalized_text)
    cleaned = re.sub(r'참고:\s*[, ]*$', '참고: (없음)', cleaned, flags=re.MULTILINE)
    return cleaned

# 대화 포맷
def format_history(messages):
    recent_msgs = messages[-6:] 
    history_text = ""
    for msg in recent_msgs:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"
    return history_text if history_text else "없음"

# 대화 히스토리 제한 관리
def manage_history_limit():
    if len(st.session_state.messages) > MAX_HISTORY:
        diff = len(st.session_state.messages) - MAX_HISTORY
        st.session_state.messages = st.session_state.messages[diff:]

# 메인 체인
def rag_answer_chain(question: str, messages: list):
    query_agent, drafter, editor = get_agents()
    
    ctx = get_shared_context()
    ctx.docs = [] 

    history_text = format_history(messages)

    search_query_response = query_agent(f"<history>\n{history_text}\n</history>\n<question>\n{question}\n</question>")
    optimized_query = str(search_query_response)
    
    draft_input = f"""
    <history>
    {history_text}
    </history>

    사용자 요청 주제: {optimized_query}
    위 주제에 대해 `search_knowledge_base` 도구를 사용하여 문서를 찾고, 문제를 출제하세요.
    """
    
    draft_response = drafter(draft_input)
    
    final_response = editor(f"다음 초안을 검수하세요:\n\n{draft_response}")
    
    contexts = ctx.docs
    doc_count = len(contexts)
    
    final_text_str = str(final_response)
    final_text_str = remove_markdown_headers(final_text_str)
    final_text = clean_hallucinated_references(final_text_str, doc_count)

    return final_text, contexts

# 세션 초기화
def init_session_state():
    defaults = {
        "messages": [], 
        "pending_answer": None, 
        "footnotes": [], 
        "selected_footnote": None
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

# UI 설정
st.set_page_config(page_title="Exam", layout="wide")
st.title("클라우드 컴퓨팅 문제 생성")
init_session_state()

st.markdown("""<style>[data-testid="stSidebar"], [data-testid="stSidebarNav"] { width: 350px !important; }</style>""", unsafe_allow_html=True)

if query := st.chat_input("주제 또는 생성할 문제 수를 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": query})
    manage_history_limit()
    st.session_state.pending_answer = query
    
    st.session_state.footnotes = []
    st.session_state.selected_footnote = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("\n", "  \n"), unsafe_allow_html=True)

if st.session_state.pending_answer:
    with st.chat_message("assistant"):
        with st.spinner("문서 검색 및 문제 생성 중..."):
            answer, contexts = rag_answer_chain(st.session_state.pending_answer, st.session_state.messages)
    
    st.session_state.messages.append({"role": "assistant", "content": answer.replace("\n", "  \n")})
    manage_history_limit()
    
    if contexts:
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
        if st.session_state.footnotes:
            st.write("번호를 클릭하면 문서 내용이 표시됩니다.")
        else:
            st.write("생성된 문제의 참고 문서가 여기에 표시됩니다.")