import streamlit as st
import boto3
import os
import re
import json
from dotenv import load_dotenv
from strands import Agent, tool

load_dotenv()

BEDROCK_REGION = os.getenv("AWS_REGION")
KB_ID = os.getenv("KNOWLEDGE_BASE_ID")
MAX_HISTORY = 20


# ===================================================================
# JSON 파싱
# ===================================================================
def extract_json(text: str):
    try:
        s = text.find("{")
        e = text.rfind("}")
        if s == -1 or e == -1:
            return None
        return json.loads(text[s:e+1])
    except:
        return None


# ===================================================================
# Shared Context
# ===================================================================
class SharedContext:
    def __init__(self):
        self.docs = []


@st.cache_resource
def get_shared_context():
    return SharedContext()


@st.cache_resource
def get_bedrock_client():
    return boto3.client("bedrock-agent-runtime", region_name=BEDROCK_REGION)


# ===================================================================
# KB 검색 도구
# ===================================================================
@tool
def search_knowledge_base(query: str) -> str:
    try:
        client = boto3.client("bedrock-agent-runtime", region_name=BEDROCK_REGION)
        resp = client.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 8}},
            retrievalQuery={"text": query},
        )

        chunks = []
        for r in resp.get("retrievalResults", []):
            score = r.get("score", 0)
            if score < 0.60:
                continue

            uri = r.get("location", {}).get("s3Location", {}).get("uri", "")
            chunks.append({
                "text": r["content"]["text"],
                "source": uri.split("/")[-1] if uri else "Unknown",
                "score": score,
            })

        grouped = {}
        for c in chunks:
            if c["source"] not in grouped:
                grouped[c["source"]] = {"texts": [], "scores": [], "source": c["source"]}
            grouped[c["source"]]["texts"].append(c["text"])
            grouped[c["source"]]["scores"].append(c["score"])

        merged = []
        for file, d in grouped.items():
            avg = sum(d["scores"]) / len(d["scores"])
            w = avg * (1 + (len(d["scores"]) - 1) * 0.25)
            merged.append({
                "source": file,
                "score": w,
                "text": "\n".join(d["texts"]),
            })

        sorted_docs = sorted(merged, key=lambda x: x["score"], reverse=True)
        ctx = get_shared_context()
        ctx.docs = sorted_docs

        if not sorted_docs:
            return "검색 결과 없음."

        out = []
        for i, doc in enumerate(sorted_docs):
            out.append(f"[ID: {i+1}] 파일: {doc['source']}\n내용:\n{doc['text']}")
        return "\n\n".join(out)

    except Exception as e:
        return f"오류: {str(e)}"


# ===================================================================
# Agents
# ===================================================================
@st.cache_resource
def get_agents():
    query_prompt = """검색어 최적화 도구. 질문에서 핵심 검색어만 추출."""

    drafter_prompt = """시험 문제 출제자. search_knowledge_base 도구로 문서를 찾아 간결한 문제 초안 작성."""

    editor_prompt = """문제 편집자. 초안을 JSON으로 변환.

규칙:
1. 100% 한국어
2. 객관식: 보기 4개(A,B,C,D), answer는 "A"/"B"/"C"/"D" 중 하나
3. OX: 보기 2개(A:O, B:X), answer는 "A" 또는 "B"
4. 단답형: 
   - options는 반드시 빈 객체 {}
   - answer는 A/B/C/D가 아닌 실제 정답 키워드 (예: "Amazon S3", "VPC", "로드밸런서")
   - explanation의 wrong는 빈 객체 {}
5. JSON만 출력, 다른 텍스트 금지

단답형 예시:
{
    "questions": [{
        "number": 1,
        "question": "AWS의 객체 스토리지 서비스 이름은?",
        "options": {},
        "answer": "Amazon S3",
        "explanation": {
            "correct": "Amazon S3는 AWS의 대표적인 객체 스토리지 서비스입니다.",
            "wrong": {}
        },
        "related_concepts": ["S3", "객체 스토리지"]
    }]
}

객관식 예시:
{
    "questions": [{
        "number": 1,
        "question": "문제",
        "options": {"A":"보기1", "B":"보기2", "C":"보기3", "D":"보기4"},
        "answer": "A",
        "explanation": {
            "correct": "해설",
            "wrong": {"A":"A해설", "B":"B해설", "C":"C해설", "D":"D해설"}
        },
        "related_concepts": ["개념1"]
    }]
}
"""

    query_agent = Agent(
        model="us.amazon.nova-lite-v1:0", 
        system_prompt=query_prompt
    )
    
    drafter = Agent(
        model="us.amazon.nova-lite-v1:0", 
        system_prompt=drafter_prompt, 
        tools=[search_knowledge_base]
    )
    
    editor = Agent(
        model="us.amazon.nova-lite-v1:0", 
        system_prompt=editor_prompt
    )

    return query_agent, drafter, editor

# ===================================================================
# 텍스트 정규화 함수
# ===================================================================
def normalize_references(t: str):
    return re.sub(r'\[ID:\s*(\d+)\]', r'[\1]', t, flags=re.IGNORECASE)


def remove_markdown_headers(t: str):
    return re.sub(r'^#+\s*', '', t, flags=re.MULTILINE)


# ===================================================================
# 단답형 답변 검증 함수
# ===================================================================
def normalize_answer(text: str) -> str:
    """답변을 정규화 (공백 제거, 소문자 변환)"""
    return re.sub(r'\s+', '', text.strip().lower())


def get_synonyms(text: str) -> list:
    """동의어 목록 반환"""
    synonym_map = {
        # AWS 서비스
        "iam정책": ["아이엠정책", "iam정책", "아이엠 정책", "iam 정책"],
        "ec2": ["이씨투", "ec2", "이씨2", "일라스틱컴퓨트클라우드"],
        "s3": ["에스쓰리", "s3", "에스3", "심플스토리지서비스"],
        "vpc": ["브이피씨", "vpc", "가상프라이빗클라우드", "버츄얼프라이빗클라우드"],
        "rds": ["알디에스", "rds", "관계형데이터베이스서비스", "릴레이셔널데이터베이스서비스"],
        "elb": ["이엘비", "elb", "일래스틱로드밸런서", "엘라스틱로드밸런서"],
        "lambda": ["람다", "lambda", "람다함수", "람다 함수"],
        "cloudfront": ["클라우드프론트", "cloudfront"],
        "route53": ["라우트53", "route53", "라우트 53", "라우트피프티쓰리"],
        "dynamodb": ["다이나모db", "dynamodb", "다이나모디비", "다이나모 db"],
        "sns": ["에스엔에스", "sns", "심플노티피케이션서비스"],
        "sqs": ["에스큐에스", "sqs", "심플큐서비스"],
        "efs": ["이에프에스", "efs", "일래스틱파일시스템"],
        "ebs": ["이비에스", "ebs", "일래스틱블록스토어"],
        "cloudwatch": ["클라우드워치", "cloudwatch"],
        "iam": ["아이엠", "iam", "아이디엔티티액세스매니지먼트"],
        
        # 클라우드 개념
        "로드밸런서": ["로드밸런싱", "로드 밸런서", "로드 밸런싱", "부하분산"],
        "오토스케일링": ["자동확장", "자동 확장", "오토 스케일링", "autoscaling"],
        "가용영역": ["availability zone", "az", "에이지", "가용 영역"],
        "리전": ["region", "지역", "리전"],
        "스토리지": ["저장소", "storage", "스토리지"],
        "인스턴스": ["instance", "인스턴스"],
        "버킷": ["bucket", "버킷"],
        "스냅샷": ["snapshot", "스냅샷", "스냅 샷"],
        "엔드포인트": ["endpoint", "종단점", "엔드 포인트"],
        "보안그룹": ["security group", "시큐리티그룹", "보안 그룹"],
    }
    
    normalized = normalize_answer(text)
    
    # 정확한 매칭 찾기
    for key, synonyms in synonym_map.items():
        if normalized in [normalize_answer(s) for s in synonyms]:
            return [normalize_answer(s) for s in synonyms]
    
    return [normalized]


def check_short_answer(user_answer: str, correct_answer: str) -> tuple:
    """
    단답형 답변 검증
    Returns: (is_correct: bool, match_type: str, message: str)
        - match_type: "exact" (완전 일치), "synonym" (동의어), "partial" (부분 일치), "wrong" (오답)
    """
    user_normalized = normalize_answer(user_answer)
    correct_normalized = normalize_answer(correct_answer)
    
    # 1. 완전 일치 확인
    if user_normalized == correct_normalized:
        return True, "exact", "정답입니다!"
    
    # 2. 동의어 확인
    user_synonyms = get_synonyms(user_answer)
    correct_synonyms = get_synonyms(correct_answer)
    
    for us in user_synonyms:
        if us in correct_synonyms:
            return True, "synonym", "정답입니다!"
    
    # 3. 부분 일치 확인 (키워드 일부 포함)
    # 사용자가 정답의 일부를 포함하거나, 정답이 사용자 답의 일부를 포함하는 경우
    if user_normalized in correct_normalized or correct_normalized in user_normalized:
        # 너무 짧은 답변(1-2글자)은 부분 일치로 인정하지 않음
        if len(user_normalized) >= 2 and len(correct_normalized) >= 2:
            if user_normalized != correct_normalized:  # 완전 일치는 이미 체크했으므로
                return False, "partial", f"아쉽습니다! 정답은 '{correct_answer}'입니다."
    
    # 4. 오답
    return False, "wrong", f"오답입니다. 정답은 '{correct_answer}'입니다."

# ===================================================================
# RAG Chain
# ===================================================================
def rag_answer_chain(question: str, messages: list, num_questions: int = 1, difficulty: str = "보통", question_type: str = "객관식"):
    query_agent, drafter, editor = get_agents()
    ctx = get_shared_context()
    ctx.docs = []

    # 난이도별 지시사항
    difficulty_guide = {
        "쉬움": "기본 개념과 정의를 묻는 쉬운 문제",
        "보통": "개념의 적용과 이해를 묻는 중간 난이도 문제",
        "어려움": "심화 개념과 복잡한 시나리오를 포함한 어려운 문제"
    }

    # 문제 유형별 지시사항
    type_guide = {
        "객관식": "4개의 선택지(A, B, C, D)가 있는 객관식 문제. answer는 A/B/C/D 중 하나",
        "OX": "참/거짓을 판단하는 문제 (선택지 A: O, B: X). answer는 A 또는 B",
        "단답형": "1~3단어 이내의 짧은 키워드로 답하는 문제. options는 빈 객체 {}, answer는 실제 정답 단어"
    }

    optimized = str(query_agent(question, max_tokens=1000))
    
    enhanced_prompt = f"""
    주제: {optimized}
    문제 개수: {num_questions}개
    난이도: {difficulty} - {difficulty_guide[difficulty]}
    문제 유형: {question_type} - {type_guide[question_type]}
    
    중요: {question_type} 유형에 맞게 정확히 생성하세요!
    """
    
    draft = drafter(enhanced_prompt, max_tokens=2000)
    
    # max_tokens를 충분히 크게 설정
    required_tokens = 2000 + (num_questions * 800)
    final_raw = editor(str(draft), max_tokens=required_tokens)

    txt = normalize_references(remove_markdown_headers(str(final_raw)))

    js = extract_json(txt)
    
    # 고유 ID 생성 (타임스탬프 기반)
    import time
    unique_id = int(time.time() * 1000)
    
    if js is None:
        js = create_error_question(question, difficulty, question_type, num_questions, unique_id)
    else:
        # 메타데이터 추가 및 검증
        for idx, q in enumerate(js.get("questions", []), 1):
            q["difficulty"] = difficulty
            q["topic"] = question
            q["type"] = question_type
            q["number"] = f"{unique_id}_{idx}"  # 고유 ID_번호 형식
            q["display_number"] = idx  # 화면 표시용 번호
            
            if "related_concepts" not in q:
                q["related_concepts"] = []
            
            # 단답형 검증 및 수정
            if question_type == "단답형":
                if "options" not in q or q["options"]:
                    q["options"] = {}
                
                # answer가 A/B/C/D면 오류로 처리
                if q.get("answer", "").upper() in ["A", "B", "C", "D"]:
                    q["answer"] = "정답 생성 오류"
                
                # wrong explanation은 빈 객체로
                if "explanation" not in q:
                    q["explanation"] = {"correct": "", "wrong": {}}
                else:
                    q["explanation"]["wrong"] = {}
            
            # 객관식/OX 검증
            else:
                if "options" not in q or not q["options"]:
                    q["options"] = {"A": "-", "B": "-", "C": "-", "D": "-"}
                
                if "explanation" not in q:
                    q["explanation"] = {"correct": "", "wrong": {}}

    return js, ctx.docs


def create_error_question(topic, difficulty, question_type, num_questions, unique_id):
    """오류 발생 시 기본 문제 생성"""
    return {
        "questions": [
            {
                "number": f"{unique_id}_{i}",
                "display_number": i,
                "question": f"문제 생성 오류 ({i}/{num_questions})",
                "options": {} if question_type == "단답형" else {"A": "-", "B": "-", "C": "-", "D": "-"},
                "answer": "오류" if question_type == "단답형" else "A",
                "difficulty": difficulty,
                "topic": topic,
                "type": question_type,
                "explanation": {
                    "correct": "JSON 파싱 오류",
                    "wrong": {} if question_type == "단답형" else {"A": "-", "B": "-", "C": "-", "D": "-"}
                },
                "related_concepts": []
            }
            for i in range(1, num_questions + 1)
        ]
    }

# ===================================================================
# 저장/분석 기능
# ===================================================================
def toggle_bookmark(qid, question_data):
    if "bookmarks" not in st.session_state:
        st.session_state.bookmarks = []
    
    bookmark_key = f"bookmark_{qid}"
    if bookmark_key in st.session_state.bookmarks:
        st.session_state.bookmarks.remove(bookmark_key)
        if "bookmark_data" in st.session_state:
            st.session_state.bookmark_data.pop(bookmark_key, None)
    else:
        st.session_state.bookmarks.append(bookmark_key)
        if "bookmark_data" not in st.session_state:
            st.session_state.bookmark_data = {}
        st.session_state.bookmark_data[bookmark_key] = question_data


def record_answer(qid, question_data, is_correct):
    """답변을 기록합니다. 같은 문제는 한 번만 기록됩니다."""
    if "answer_history" not in st.session_state:
        st.session_state.answer_history = []
    
    if "answered_questions" not in st.session_state:
        st.session_state.answered_questions = {}
    
    # 문제의 고유 ID 생성 (문제 내용 기반)
    question_hash = hash(question_data["question"])
    
    # 이미 푼 문제인지 확인
    if question_hash in st.session_state.answered_questions:
        # 이미 기록된 답변 업데이트 (정답/오답 상태만 갱신)
        for record in st.session_state.answer_history:
            if record.get("question_hash") == question_hash:
                record["correct"] = is_correct
                break
        st.session_state.answered_questions[question_hash] = is_correct
        return
    
    # 새로운 문제 기록
    record = {
        "qid": qid,
        "question_hash": question_hash,
        "topic": question_data.get("topic", ""),
        "difficulty": question_data.get("difficulty", "보통"),
        "correct": is_correct,
        "question": question_data
    }
    st.session_state.answer_history.append(record)
    st.session_state.answered_questions[question_hash] = is_correct


def get_statistics():
    if "answer_history" not in st.session_state or not st.session_state.answer_history:
        return None
    
    history = st.session_state.answer_history
    total = len(history)
    correct = sum(1 for h in history if h["correct"])
    
    # 주제별 정답률
    topic_stats = {}
    for h in history:
        topic = h["topic"]
        if topic not in topic_stats:
            topic_stats[topic] = {"total": 0, "correct": 0}
        topic_stats[topic]["total"] += 1
        if h["correct"]:
            topic_stats[topic]["correct"] += 1
    
    # 난이도별 정답률
    diff_stats = {"쉬움": {"total": 0, "correct": 0}, 
                  "보통": {"total": 0, "correct": 0}, 
                  "어려움": {"total": 0, "correct": 0}}
    for h in history:
        diff = h["difficulty"]
        diff_stats[diff]["total"] += 1
        if h["correct"]:
            diff_stats[diff]["correct"] += 1
    
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total * 100) if total > 0 else 0,
        "topic_stats": topic_stats,
        "diff_stats": diff_stats
    }


def get_weak_topics():
    stats = get_statistics()
    if not stats or not stats["topic_stats"]:
        return []
    
    weak = []
    for topic, data in stats["topic_stats"].items():
        if data["total"] >= 2:
            accuracy = (data["correct"] / data["total"] * 100) if data["total"] > 0 else 0
            if accuracy < 60:
                weak.append({
                    "topic": topic,
                    "accuracy": accuracy,
                    "total": data["total"]
                })
    
    return sorted(weak, key=lambda x: x["accuracy"])


def get_wrong_questions():
    """틀린 문제만 반환합니다. 중복 제거됨."""
    if "answer_history" not in st.session_state:
        return []
    
    wrong = []
    seen = set()
    
    # 최신 기록부터 확인 (같은 문제를 나중에 맞췄을 수도 있음)
    for h in reversed(st.session_state.answer_history):
        question_hash = h.get("question_hash")
        if question_hash and question_hash not in seen:
            seen.add(question_hash)
            if not h["correct"]:
                wrong.append(h)
    
    return wrong



# ===================================================================
# 보기 클릭형 UI 컴포넌트
# ===================================================================
def render_question(qid, question_data, context="main"):
    question = question_data["question"]
    options = question_data.get("options", {})
    correct = question_data["answer"]
    explanation = question_data["explanation"]
    difficulty = question_data.get("difficulty", "보통")
    topic = question_data.get("topic", "")
    q_type = question_data.get("type", "객관식")
    related = question_data.get("related_concepts", [])
    display_num = question_data.get("display_number", qid)  # 표시용 번호

    # 난이도 색상
    diff_colors = {
        "쉬움": "#10b981",
        "보통": "#f59e0b",
        "어려움": "#ef4444"
    }

    # 북마크 상태 확인
    bookmark_key = f"bookmark_{qid}"
    is_bookmarked = bookmark_key in st.session_state.get("bookmarks", [])

    # 질문 카드 (display_num 사용)
    st.markdown(f"""
        <div style="
            padding: 28px;
            border-radius: 16px;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border: 2px solid #e9ecef;
            margin-bottom: 32px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            position: relative;
        ">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px;">
                <h3 style="margin: 0; color: #1e293b; font-size: 1.25rem; flex: 1; padding-right: 20px;">Q{display_num}. {question}</h3>
                <div style="display: flex; gap: 8px; align-items: center; flex-shrink: 0;">
                    <span style="
                        background-color: {diff_colors.get(difficulty, '#6b7280')};
                        color: white;
                        padding: 6px 14px;
                        border-radius: 20px;
                        font-size: 0.8rem;
                        font-weight: 600;
                        white-space: nowrap;
                    ">{difficulty}</span>
                    <span style="
                        background-color: #6366f1;
                        color: white;
                        padding: 6px 14px;
                        border-radius: 20px;
                        font-size: 0.8rem;
                        font-weight: 600;
                        white-space: nowrap;
                    ">{q_type}</span>
                </div>
            </div>
    """, unsafe_allow_html=True)

    # context를 포함한 고유 key 생성
    sel_key = f"selected_{context}_{qid}"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = None

    selected = st.session_state[sel_key]
    is_answered = selected is not None  # 답변 여부 확인

    # 단답형인 경우 텍스트 입력
    if q_type == "단답형":
        answer_key = f"answer_{context}_{qid}"
        
        if is_answered:
            st.text_input("답변을 입력하세요:", value=selected, key=answer_key, disabled=True, placeholder="답변 완료")
        else:
            user_answer = st.text_input("답변을 입력하세요:", key=answer_key, placeholder="핵심 키워드를 입력하세요")
            
            submit_key = f"submit_{context}_{qid}"
            if st.button("제출", key=submit_key) and user_answer:
                # 여기서만 record_answer 호출
                is_correct, match_type, feedback = check_short_answer(user_answer, correct)
                st.session_state[sel_key] = user_answer
                st.session_state[f"feedback_{sel_key}"] = feedback
                st.session_state[f"match_type_{sel_key}"] = match_type  # match_type 저장
                record_answer(qid, question_data, is_correct)  # ✅ 제출 버튼 클릭 시에만
                st.rerun()

    # OX 문제인 경우
    elif q_type == "OX":
        ox_options = {"O": "O (참)", "X": "X (거짓)"}
        for key, text in ox_options.items():
            button_key = f"btn_{context}_{qid}_{key}"
            actual_correct = "O" if correct == "A" else "X"
            
            if is_answered:
                if selected == key:
                    st.button(f"✓ {text}", key=button_key, use_container_width=True, disabled=True, type="primary")
                else:
                    st.button(text, key=button_key, use_container_width=True, disabled=True)
            else:
                if st.button(text, key=button_key, use_container_width=True):
                    st.session_state[sel_key] = key
                    record_answer(qid, question_data, key == actual_correct)  # ✅ 버튼 클릭 시에만
                    st.rerun()

    # 객관식인 경우
    else:
        for key, text in options.items():
            button_key = f"btn_{context}_{qid}_{key}"
            
            if is_answered:
                if selected == key:
                    st.button(f"✓ {key}. {text}", key=button_key, use_container_width=True, disabled=True, type="primary")
                else:
                    st.button(f"{key}. {text}", key=button_key, use_container_width=True, disabled=True)
            else:
                if st.button(f"{key}. {text}", key=button_key, use_container_width=True):
                    st.session_state[sel_key] = key
                    record_answer(qid, question_data, key == correct)  # ✅ 버튼 클릭 시에만
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 정오답 피드백
    if selected:
        if q_type == "단답형":
            is_correct, match_type, feedback_msg = check_short_answer(selected, correct)
            feedback_key = f"feedback_{sel_key}"
            match_type_key = f"match_type_{sel_key}"
            
            # 저장된 피드백이 있으면 사용
            if feedback_key in st.session_state:
                feedback_msg = st.session_state[feedback_key]
            if match_type_key in st.session_state:
                match_type = st.session_state[match_type_key]
        elif q_type == "OX":
            actual_correct = "O" if correct == "A" else "X"
            is_correct = selected == actual_correct
            feedback_msg = "정답입니다!" if is_correct else "오답입니다."
            match_type = "exact" if is_correct else "wrong"
        else:
            is_correct = selected == correct
            feedback_msg = "정답입니다!" if is_correct else "오답입니다."
            match_type = "exact" if is_correct else "wrong"
            
        # 피드백 표시
        if is_correct:
            st.success(feedback_msg)
            with st.expander("해설 보기", expanded=False):
                st.info(explanation.get("correct", "정답입니다."))
                
                if related:
                    st.markdown("**관련 학습 자료**")
                    for concept in related:
                        search_url = f"https://www.google.com/search?q={concept}+클라우드+컴퓨팅"
                        st.markdown(f"- [{concept}]({search_url})")
        else:
            # match_type에 따라 다른 색상 표시
            if q_type == "단답형" and match_type == "partial":
                st.warning(feedback_msg)  # 부분 일치는 노란색으로
            else:
                st.error(feedback_msg)  # 오답은 빨간색으로
            
            with st.expander("해설 보기", expanded=False):
                if q_type == "단답형":
                    st.info(f"**정답:** {correct}\n\n{explanation.get('correct', '')}")
                elif q_type == "OX":
                    actual_correct = "O" if correct == "A" else "X"
                    st.info(f"**정답:** {actual_correct}\n\n{explanation.get('correct', '')}")
                else:
                    st.warning(f"**선택한 보기 ({selected})**\n\n{explanation['wrong'].get(selected, '-')}")
                    st.info(f"**정답 ({correct})**\n\n{explanation['correct']}")
                
                if related:
                    st.markdown("**복습 자료**")
                    for concept in related:
                        search_url = f"https://www.google.com/search?q={concept}+클라우드+컴퓨팅"
                        st.markdown(f"- [{concept}]({search_url})")

    # 북마크 버튼 (고유 key 사용)
    bm_button_key = f"bookmark_btn_{context}_{qid}"
    bookmark_label = "★ 저장됨" if is_bookmarked else "☆ 저장하기"
    
    col1, col2 = st.columns([2, 8])
    with col1:
        if st.button(bookmark_label, key=bm_button_key, help="중요 문제로 저장"):
            toggle_bookmark(qid, question_data)
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ===================================================================
# UI 세션 초기화
# ===================================================================
def init_session_state():
    defaults = {
        "messages": [],
        "pending_answer": None,
        "quiz_data": None,
        "footnotes": [],
        "bookmarks": [],
        "bookmark_data": {},
        "answer_history": [],
        "answered_questions": {},  # 문제 해시 -> 정답여부 매핑
        "num_questions": 1,
        "difficulty": "보통",
        "question_type": "객관식",
        "current_mode": "generate"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ===================================================================
# Main UI
# ===================================================================
st.set_page_config(page_title="Exam Generator", layout="wide", initial_sidebar_state="expanded")

# 전체 페이지 스타일
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 900px;
        }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
        }
        
        .main .block-container {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
        }
        
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        div[data-testid="stButton"] button {
            text-align: left;
            padding: 16px 20px;
            border-radius: 12px;
            background-color: #ffffff;
            border: 2px solid #e2e8f0;
            color: #334155;
            transition: all 0.3s ease;
            font-weight: 500;
            font-size: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        div[data-testid="stButton"] button:hover {
            background-color: #f1f5f9;
            border-color: #6366f1;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(99,102,241,0.1);
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f8f9fa;
            padding: 8px;
            border-radius: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }
    </style>
""", unsafe_allow_html=True)

st.title("클라우드 컴퓨팅 학습 시스템")
st.markdown("<p style='color: #64748b; font-size: 1.1rem; margin-top: -10px;'>AI 기반 맞춤형 문제 생성 및 학습 분석</p>", unsafe_allow_html=True)

init_session_state()

# 탭 네비게이션
tab1, tab2, tab3, tab4 = st.tabs(["✏️ 문제 생성", "⭐ 저장한 문제", "❌ 오답 노트", "📊 학습 통계"])

with tab1:
    # 설정 패널
    with st.expander("설정", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            num_q = st.selectbox(
                "문제 개수",
                options=[1, 2, 3, 4, 5],
                index=0,
                key="num_q_select"
            )
            st.session_state.num_questions = num_q
        
        with col2:
            diff = st.selectbox(
                "난이도",
                options=["쉬움", "보통", "어려움"],
                index=1,
                key="diff_select"
            )
            st.session_state.difficulty = diff
        
        with col3:
            q_type = st.selectbox(
                "문제 유형",
                options=["객관식", "OX", "단답형"],
                index=0,
                key="type_select"
            )
            st.session_state.question_type = q_type

    # 입력창
    if query := st.chat_input("문제를 생성할 주제를 입력하세요 (예: EC2 인스턴스 유형, S3 버킷 정책)"):
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.pending_answer = query
        st.session_state.quiz_data = None
        st.rerun()

    # 히스토리
    for msg in st.session_state.messages[-6:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 문제 생성
    if st.session_state.pending_answer:
        with st.chat_message("assistant"):
            with st.spinner("문서 검색 및 문제 생성 중…"):
                js, docs = rag_answer_chain(
                    st.session_state.pending_answer, 
                    st.session_state.messages,
                    num_questions=st.session_state.num_questions,
                    difficulty=st.session_state.difficulty,
                    question_type=st.session_state.question_type
                )
        st.session_state.quiz_data = js
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"'{st.session_state.pending_answer}' 주제에 대한 {st.session_state.num_questions}개의 {st.session_state.difficulty} 난이도 문제를 생성했습니다."
        })
        st.session_state.pending_answer = None
        st.rerun()

    ## 문제 출력
    if st.session_state.quiz_data:
        st.markdown("---")
        st.header("생성된 시험 문제")
        for q in st.session_state.quiz_data["questions"]:
            render_question(q["number"], q, context="generate")  # q["number"]는 고유 ID

with tab2:
    st.header("저장한 중요 문제")
    if st.session_state.get("bookmark_data"):
        for bookmark_key, q_data in st.session_state.bookmark_data.items():
            qid = int(bookmark_key.split("_")[1])
            render_question(qid, q_data, context="bookmark")
    else:
        st.info("중요한 문제는 ☆ 버튼을 눌러 저장하세요")

with tab3:
    st.header("오답 노트")
    wrong_qs = get_wrong_questions()
    if wrong_qs:
        st.warning(f"총 {len(wrong_qs)}개의 틀린 문제가 있습니다. 복습하세요!")
        for idx, w in enumerate(wrong_qs):
            render_question(w["qid"], w["question"], context=f"wrong_{idx}")
    else:
        st.success("아직 틀린 문제가 없습니다!")

with tab4:
    st.header("학습 통계 및 분석")
    stats = get_statistics()
    
    if stats:
        # 전체 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 문제 수", f"{stats['total']}문제")
        with col2:
            st.metric("정답 수", f"{stats['correct']}문제")
        with col3:
            st.metric("정답률", f"{stats['accuracy']:.1f}%")
        
        st.markdown("---")
        
        # 난이도별 통계
        st.subheader("난이도별 정답률")
        diff_data = stats["diff_stats"]
        for diff, data in diff_data.items():
            if data["total"] > 0:
                acc = (data["correct"] / data["total"] * 100)
                st.progress(acc / 100, text=f"{diff}: {acc:.1f}% ({data['correct']}/{data['total']})")
        
        st.markdown("---")
        
        # 취약 주제
        st.subheader("취약 주제 분석")
        weak = get_weak_topics()
        if weak:
            st.warning("다음 주제들을 집중 학습하세요:")
            for w in weak:
                st.markdown(f"""
                <div style="
                    padding: 12px;
                    background-color: #fff3cd;
                    border-left: 4px solid #ffc107;
                    border-radius: 8px;
                    margin-bottom: 8px;
                ">
                    <strong>{w['topic']}</strong><br>
                    정답률: {w['accuracy']:.1f}% ({w['total']}문제 풀이)
                    <a href="https://www.google.com/search?q={w['topic']}+클라우드+컴퓨팅+강의" target="_blank" style="margin-left: 10px;">학습하기</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("취약한 주제가 없습니다! 모든 주제를 잘 이해하고 있어요.")
    else:
        st.info("문제를 풀면 학습 통계가 표시됩니다.")

# 사이드바
with st.sidebar:
    st.markdown("""
        <div style="
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            margin-bottom: 20px;
            text-align: center;
        ">
            <h2 style="color: white; margin: 0;">학습 대시보드</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # 빠른 통계
    st.subheader("빠른 통계")
    stats = get_statistics()
    if stats:
        st.metric("총 풀이 수", f"{stats['total']}문제")
        st.metric("정답률", f"{stats['accuracy']:.1f}%")
        st.metric("저장한 문제", f"{len(st.session_state.get('bookmarks', []))}개")
        
        wrong_count = len(get_wrong_questions())
        if wrong_count > 0:
            st.metric("복습 필요", f"{wrong_count}문제", delta=f"-{wrong_count}", delta_color="inverse")
    else:
        st.info("문제를 풀면 통계가 표시됩니다.")
    
    st.markdown("---")
    
    # 참고 문서
    st.subheader("참고 문서 (RAG Source)")
    docs = get_shared_context().docs
    if docs:
        for i, d in enumerate(docs, 1):
            with st.expander(f"[{i}] {d['source']}", expanded=False):
                st.caption(f"관련도: {d['score']:.4f}")
                st.code(d["text"][:500] + "..." if len(d["text"]) > 500 else d["text"], language="markdown")
    else:
        st.caption("문제 생성 시 참고 문서가 표시됩니다.")
    
    st.markdown("---")
    
    # 학습 팁
    st.subheader("학습 팁")
    st.markdown("""
        <div style="
            background-color: #f0f9ff;
            padding: 15px;
            border-radius: 12px;
            border-left: 4px solid #0ea5e9;
        ">
            <ul style="margin: 0; padding-left: 15px;">
                <li style="margin-bottom: 8px;">난이도를 점진적으로 높여보세요</li>
                <li style="margin-bottom: 8px;">틀린 문제는 오답노트에서 복습하세요</li>
                <li style="margin-bottom: 8px;">취약 주제를 집중 학습하세요</li>
                <li style="margin-bottom: 8px;">중요한 문제는 저장하세요</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
# 데이터 초기화
    if st.button("학습 데이터 초기화", type="secondary", use_container_width=True):
        st.session_state.answer_history = []
        st.session_state.answered_questions = {}
        st.session_state.bookmarks = []
        st.session_state.bookmark_data = {}
        st.success("초기화 완료!")
        st.rerun()