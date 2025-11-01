# -*- coding: utf-8 -*-
"""
Streamlit 서술형 과제 제출 + GPT 채점/피드백 (천문 단원: 절대등급/거리지수)
단일 파일 실행:  streamlit run app.py
필수: .streamlit/secrets.toml 에 OPENAI_API_KEY 저장
"""

import math
import streamlit as st
from openai import OpenAI, OpenAIError

# -------------------------------
# 페이지 기본 설정
# -------------------------------
st.set_page_config(page_title="별의 등급과 거리", page_icon="✨", layout="centered")

# -------------------------------
# 1) 수업 제목 및 안내
# -------------------------------
st.title("별의 등급을 이용하여 거리 구하기")
st.markdown(
    "> **안내**  \n"
    "> 1) 학번과 각 문항의 답안을 작성한 뒤 **[제출]** 버튼을 누르세요.  \n"
    "> 2) **[GPT 피드백 확인]**을 누르면 문항별 O/X와 200자 이내 피드백이 생성됩니다.  \n"
    "> 3) 아래 **학습 도우미**에서 절대등급/거리 계산을 실습할 수 있습니다."
)

# -------------------------------
# 2) 학번 입력
# -------------------------------
student_id = st.text_input("학번", help="학생의 학번을 작성하세요. (예: 10130)")

# -------------------------------
# 3) 문항 렌더링 유틸: 문항 리스트 → 개별 답안 칸 생성
# -------------------------------
def render_questions(questions, base_key="q"):
    """문항 리스트를 받아 각 문항 아래에 개별 답안 칸을 생성하고, 답안 리스트를 반환합니다."""
    answers = []
    for i, q in enumerate(questions, start=1):
        st.markdown(f"#### 서술형 문제 {i}")
        st.write(q.strip())
        ans = st.text_area(
            "답안을 입력하세요",
            key=f"{base_key}_answer_{i}",
            height=150,
            placeholder="자신의 말로 서술형 답안을 작성해 보세요."
        )
        answers.append(ans or "")
    return answers

# -------------------------------
# 4) 문항 정의 (절대 등급/겉보기 등급/거리지수)
# -------------------------------
questions = [
    "별의 **절대 등급(M)** 은 어떻게 정의되며, 어떻게 구하나요?",
    "별의 **겉보기 등급(m)** 은 무엇이며 어떻게 정해지나요?",
    "별의 **겉보기 등급(m)** 과 **절대 등급(M)** 으로 **거리 d(pc)** 를 어떻게 구하나요? (성간 소광 고려 여부도 언급)"
]

answers = render_questions(questions, base_key="astro")

# -------------------------------
# (선택) 예시 답안/정의 — 교사용 참고
# -------------------------------
with st.expander("예시 답안 보기 (교사용 참고)"):
    st.markdown(
        "**문항 1 예시 — 절대 등급**  \n"
        "절대 등급(M)은 별을 **지구로부터 10파섹(10 pc)** 에 두었을 때의 밝기 등급입니다.  "
        "관측된 겉보기 등급(m)과 거리가 주어지면 **거리지수식** *m − M = 5\\log_{10}(d/10)* 으로 M을 계산합니다."
    )
    st.markdown(
        "**문항 2 예시 — 겉보기 등급**  \n"
        "겉보기 등급(m)은 **지구에서 보이는 밝기**를 로그 눈금으로 나타낸 값으로, 값이 **작을수록 밝습니다**.  "
        "전통적으로 기준성(베가 등)과 광도비의 로그 관계를 사용하며, 관측 대역(예: V-band)에 따라 값이 정해집니다."
    )
    st.markdown(
        "**문항 3 예시 — 거리 구하기(거리지수)**  \n"
        "겉보기 등급 m 과 절대 등급 M 사이에는 **거리지수(distance modulus)** 관계인  "
        "*m − M = 5\\log_{10}(d/10)* 가 성립합니다.  "
        "따라서 거리 *d(pc) = 10^{\\frac{m − M + 5}{5}}* 로 구합니다.  "
        "성간 소광이 큰 영역에서는 소광량 *A* 를 고려해 *m − M − A = 5\\log_{10}(d/10)* 로 보정합니다."
    )

# -------------------------------
# 5) 학습 도우미: 절대등급/거리 계산기
# -------------------------------
with st.expander("학습 도우미: 절대등급/거리 계산기"):
    st.markdown("**공식 요약**")
    st.latex(r"m - M = 5\log_{10}\!\left(\frac{d}{10\,\mathrm{pc}}\right) \quad(\text{소광 무시})")
    st.latex(r"m - M - A = 5\log_{10}\!\left(\frac{d}{10\,\mathrm{pc}}\right) \quad(\text{소광 }A\text{ 고려})")

    mode = st.radio(
        "계산 모드 선택",
        ["M 구하기 (m, d → M)", "d 구하기 (m, M → d)"],
        horizontal=True,
        key="calc_mode",
    )
    Av = st.number_input("소광량 A (mag, 선택 입력)", min_value=0.0, value=0.0, step=0.01, help="성간 소광을 모르면 0으로 두세요.")

    col1, col2 = st.columns(2)
    if mode.startswith("M 구하기"):
        with col1:
            m_val = st.number_input("겉보기 등급 m", value=10.0, step=0.1, format="%.3f")
        with col2:
            d_pc = st.number_input("거리 d (pc)", value=100.0, min_value=0.0001, step=1.0, format="%.3f")

        if st.button("계산하기", key="calc_M"):
            # M = m - 5*log10(d/10) - A
            M = m_val - 5.0 * math.log10(d_pc / 10.0) - Av
            st.success(f"계산 결과 → **절대 등급 M = {M:.3f}**")
    else:
        with col1:
            m_val = st.number_input("겉보기 등급 m", value=10.0, step=0.1, format="%.3f")
        with col2:
            M_val = st.number_input("절대 등급 M", value=5.0, step=0.1, format="%.3f")

        if st.button("계산하기", key="calc_d"):
            # m - M - A = 5 log10(d/10)  →  d = 10^((m - M - A)/5) * 10
            exponent = (m_val - M_val - Av) / 5.0
            d_pc = (10.0 ** exponent) * 10.0
            st.success(f"계산 결과 → **거리 d = {d_pc:.3f} pc**")

# -------------------------------
# 6) 제출 버튼
# -------------------------------
submitted = False
if st.button("제출"):
    if not student_id.strip():
        st.warning("학번을 입력하세요.")
    elif any(ans.strip() == "" for ans in answers):
        st.warning("모든 답안을 작성하세요.")
    else:
        st.success(f"제출 완료! 학번: {student_id}")
        submitted = True

# -------------------------------
# 7) OpenAI 클라이언트
# -------------------------------
def get_client():
    try:
        return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except KeyError:
        st.error("⚠️ .streamlit/secrets.toml 에 OPENAI_API_KEY를 설정하세요.")
        return None

# -------------------------------
# 8) 채점 기준 (교사가 자유롭게 수정)
# -------------------------------
GRADING_GUIDELINES = {
    1: "절대 등급(M)의 정의(10 pc 기준)를 명확히 쓰고, m−M=5log10(d/10)로 M을 계산하는 과정을 설명.",
    2: "겉보기 등급(m)의 정의(지구에서 보이는 밝기의 로그 척도), 값이 작을수록 밝음, 관측 대역/기준성 언급.",
    3: "거리지수식 m−M=5log10(d/10) 제시, d=10^((m−M+5)/5) 유도, 소광 보정(m−M−A) 가능성 언급.",
}

# -------------------------------
# 9) GPT 피드백 생성 버튼
# -------------------------------
if st.button("GPT 피드백 확인"):
    if not student_id.strip():
        st.warning("학번을 먼저 입력하세요.")
        st.stop()
    if any(ans.strip() == "" for ans in answers):
        st.warning("모든 답안을 작성한 뒤 다시 시도하세요.")
        st.stop()
    if not submitted:
        st.info("참고: 제출을 누르지 않아도 피드백 생성은 가능합니다. (그래도 제출을 권장합니다.)")

    client = get_client()
    if client is None:
        st.stop()

    feedbacks = []
    for idx, ans in enumerate(answers, start=1):
        criterion = GRADING_GUIDELINES.get(idx, "채점 기준이 없습니다.")
        prompt = (
            f"[채점 지시]\n"
            f"- 문항 번호: {idx}\n"
            f"- 채점 기준: {criterion}\n"
            f"- 학생 답안: {ans}\n"
            f"- 출력 형식: 한국어로 'O:' 또는 'X:'로 시작하고, 이어서 **200자 이내**의 구체적 피드백을 작성.\n"
            f"- 점수는 출력하지 말 것. 핵심 개념 누락/오개념이 있으면 간단히 짚고 보완 힌트를 제시."
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",   # 필요 시 모델 변경 가능
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=250,
            )
            text = resp.choices[0].message.content.strip()
        except OpenAIError as e:
            text = f"API 오류: {e}"

        feedbacks.append(text)

    st.markdown("---")
    st.subheader("문항별 GPT 피드백")
    for i, fb in enumerate(feedbacks, start=1):
        st.markdown(f"**▶ 서술형 문제 {i}**")
        st.info(fb)

    st.success("모든 피드백이 생성되었습니다. 교사 확인 후 학생에게 전달하세요.")
