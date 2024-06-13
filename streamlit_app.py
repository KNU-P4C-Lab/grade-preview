import pandas as pd
import altair as alt
import streamlit as st
import numpy as np
from scipy import stats


def load_data(course: str, student_id: str, email: str) -> tuple:
    if course == '문제 해결 프로그래밍':
        path = './data/problem-solving.csv'
    elif course == '알고리즘':
        path = './data/algorithm.csv'
    elif course == '데이터베이스':
        path = './data/database.csv'
    else:
        raise ValueError('해당하는 과목 성적이 없습니다.')

    try:
        grade = pd.read_csv(path)
        ind = grade.loc[lambda x: (x['student_id'] == int(student_id)) & (x['email'] == email), :]
        #ind = grade.loc[lambda x: (x['student_id'] == 202013294) & (x['email'] == 'chsm0403@naver.com'), :]
    except Exception as e:
        print(e)
        raise ValueError('해당하는 학번과 이메일이 없습니다.')

    if len(ind) == 0:
        raise ValueError('해당하는 학번과 이메일이 없습니다.')

    return grade, ind

def draw_plot(d: np.array) -> alt.Chart:
    avg, std = np.mean(d), np.std(d, ddof=1)
    d_pdf = pd.DataFrame(
        dict(
            Score=np.linspace(0, 100, 1000),
            PDF=stats.t.pdf(np.linspace(0, 100, 1000), len(d) - 1, avg, std),
        )
    )
    y, x = np.histogram(d, bins=np.linspace(0, 100, 11), density=True)
    d_hist = pd.DataFrame(dict(Score=x[:-1], PDF=y))

    p_pdf = alt.Chart(d_pdf).encode(
        x='Score:Q', y='PDF:Q'
    ).mark_line(
        color='red'
    )
    p_hist = alt.Chart(d_hist).encode(
        x='Score:Q', y='PDF:Q'
    ).mark_bar(
        width=20, opacity=0.7
    )
    p = alt.layer(p_hist + p_pdf).resolve_scale(
        y='independent'
    )
    return p


st.title('성적 미리보기')
with st.form(key='info', clear_on_submit=False):
    option_course = st.selectbox(
        '과목',
        ('문제 해결 프로그래밍', '알고리즘', '데이터베이스'),
        placeholder="과목을 선택하세요",
         index=None
    )
    option_student_id = st.text_input(
        '학번',
        placeholder='학번을 입력하세요',
        value=None
    )
    option_email = st.text_input(
        '이메일',
        placeholder='이루리에 입력한 이메일을 입력하세요.',
        value=None
    )
    is_submitted = st.form_submit_button('제출')

try:
    if is_submitted:
        grade_total, grade_ind = load_data(option_course, option_student_id, option_email)
        grade_ind = grade_ind.to_dict('list')
        is_foreign = grade_ind['foreign_std'][0] == 1
        with st.container(border=True):
            st.markdown(f'## {grade_ind["name"][0]} 학생의 성적')
            ccs = st.columns(3)
            ccs[0].metric('학생 구분', f'{"외국인" if is_foreign else "한국인"}')
            ccs[1].metric('석차', f'{grade_ind["order"][0]}/{len(grade_total)}')
            ccs[2].metric('학점', f'{grade_ind["grade"][0]}')

        with st.container(border=True):
            st.markdown('## 환산 점수')
            ocs = st.columns(5)
            ocs[0].metric('출석', f'{grade_ind["w_atnd"][0]:.2f}')
            ocs[1].metric('과제', f'{grade_ind["w_asmt"][0]:.2f}')
            ocs[2].metric('중간고사', f'{grade_ind["w_mid"][0]:.2f}')
            ocs[3].metric('기말고사', f'{grade_ind["w_final"][0]:.2f}')
            ocs[4].metric('계', f'{grade_ind["w_total"][0]:.2f}')

        with st.container(border=True):
            st.markdown('## 성적 분포')
            w_total = grade_total['w_total'].values
            dc1, dc2 = st.columns([2, 1])
            chart = draw_plot(w_total)
            dc1.altair_chart(chart)
            dc2.metric('평균', f'{np.mean(w_total):.2f}')
            dc2.metric('표준 편차', f'{np.std(w_total, ddof=1):.2f}')
            dc2.metric('중간값', f'{np.median(w_total):.2f}')

        with st.container(border=True):
            st.markdown('## 세부 성적')
            st.markdown('### 과제')
            rcs1 = st.columns(4)

            grade_detail = {
                k: v for k, v in grade_ind.items()
                if k.startswith('asmt_') or k in ['mid', 'final', 'atnd']
            }
            for i in range(1, 100):
                k = f'asmt_{i:02d}'
                if k in grade_ind:
                    rcs1[(i - 1) % 4].metric(f'과제 #{i:02d}', f'{grade_ind[k][0]:.2f}')
                else:
                    break

            st.markdown('### 출석')
            rcs2 = st.columns(4)
            rcs2[0].metric('출석 횟수', grade_ind['atnd'][0])
            rcs2[1].metric('결석 횟수', grade_ind['absnt'][0])
            rcs2[2].metric('지각 횟수', grade_ind['lateness'][0])
            rcs2[3].metric('공결 횟수', grade_ind['offabsnt'][0])

            st.markdown('### 시험')
            rcs3 = st.columns(2)
            rcs3[0].metric('중간고사', grade_ind['mid'][0])
            rcs3[1].metric('기말고사', grade_ind['final'][0])


except Exception as e:
    st.error(e)



