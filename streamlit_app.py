import time
import re
import pandas as pd
import altair as alt
import streamlit as st
import numpy as np
from scipy import stats
from cryptography.fernet import Fernet
import pickle


def load_data(key: str, course: str, student_id: str, email: str, phone: str) -> tuple:
    if course == '인간-컴퓨터 상호작용':
        path = './data/hci.res'
    elif course == '기계학습':
        path = './data/ml.res'
    else:
        raise ValueError('해당하는 과목 성적이 없습니다.')

    try:
        with open(path, mode='rb') as f:
            data = f.read()
        fernet = Fernet(str.encode(key))
        data = fernet.decrypt(data)
        grade = pickle.loads(data)
        ind = grade.loc[
            lambda x: (x['ID'] == student_id) & (x['Email'] == email) & (x['Phone'] == phone), :
        ]
        if len(ind) == 0:
            raise ValueError()
        ind = ind.squeeze(axis=0)
        return grade, ind

    except Exception:
        raise ValueError('해당하는 학번, 이메일, 휴대폰 번호가 없습니다.')


def draw_plot(d: np.array) -> alt.LayerChart:
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
        x=alt.X('Score:Q', title='점수'), y=alt.X('PDF:Q', title='추정 분포')
    ).mark_line(
        color='red'
    )
    p_hist = alt.Chart(d_hist).encode(
        x=alt.X('Score:Q', title='점수'), y=alt.X('PDF:Q', title='실제 분포')
    ).mark_bar(
        width=20, opacity=0.7
    )
    p = alt.layer(p_hist + p_pdf).resolve_scale(
        y='independent'
    )
    return p


def generate_page():
    with st.form(key='info', clear_on_submit=False):
        option_course = st.selectbox(
            label='과목',
            options=('인간-컴퓨터 상호작용', '기계학습'),
            placeholder='과목을 선택하세요.',
            index=None
        )

        option_id = st.text_input(
            label='학번',
            placeholder='학번을 입력하세요.',
            value=None
        )

        option_email = st.text_input(
            label='이메일',
            placeholder='이루리에 입력한 이메일을 입력하세요.',
            value=None
        )

        option_phone = st.text_input(
            label='휴대폰 번호',
            placeholder='이루리에 입력한 휴대폰 번호를 입력하세요.',
            value=None
        )
        is_submitted = st.form_submit_button('확인')
    if not is_submitted:
        return
    option_id = re.sub(r'\s', '', option_id)
    option_phone = re.sub('[^0-9]', '', option_phone)
    option_email = re.sub(r'\s', '', option_email)
    with st.spinner('성적 확인 중...'):
        # g_total, g_ind = load_data(st.secrets['shared_key'], '인간-컴퓨터 상호작용', '201811614', 'keroro06108@gmail.com', '01055364640')
        g_total, g_ind = load_data(st.secrets['shared_key'], option_course, option_id, option_email, option_phone)
        time.sleep(1)

    n = g_total.shape[0]
    common = ['ID', 'Email', 'Name', 'Phone', 'Total', 'Feedback', 'Rank', 'Grade', 'Absence', 'Lateness']
    with st.container(border=True):
        st.markdown(f'## {g_ind["Name"]} 학생의 성적')
        cols = st.columns(3)
        cols[0].metric(
            label='총점', value=f'{g_ind["Total"]:.2f} / 100'
        )
        cols[1].metric(
            label='석차', value=f'{g_ind["Rank"]} / {n}'
        )
        cols[2].metric(
            label='학점', value=g_ind['Grade']
        )

    with st.container(border=True):
        st.markdown('## 세부 점수')
        cols = st.columns(4)
        i = 0
        for k in g_ind.index:
            if k in common:
                continue
            v = g_ind[k]
            if type(v) is str:
                cols[i % 4].metric(label=k, value=v)
            else:
                cols[i % 4].metric(label=k, value=f'{v:.2f}')
            i = i + 1

    with st.container(border=True):
        st.markdown('## 피드백')
        for l in g_ind['Feedback'].split('; '):
            st.markdown(f'* {l}')

    with st.container(border=True):
        st.markdown('## 출결')

    with st.container(border=True):
        st.markdown('## 성적 분포')
        col1, col2 = st.columns((2, 1))
        v = g_total['Total'].values
        chart = draw_plot(v)
        col1.altair_chart(chart)
        col2.metric(label='평균', value=f'{np.nanmean(v):.2f}')
        col2.metric(label='표준 편차', value=f'{np.nanstd(v, ddof=1):.2f}')
        col2.metric(label='중간값', value=f'{np.nanmedian(v):.2f}')


st.title('성적 미리보기 (2024 봄)')

if st.secrets['is_ready']:
    try:
        generate_page()
    except Exception as e:
        st.error(e)
else:
    st.subheader('성적 입력이 완료될 때까지 기다려주세요.')